
# drives robot from start to just short of dock - records all sensors through driver layer
# Path (your ward: trolley at x 2.8-3.4, y 0.8-1.2, directly in the way):

    #   y 2.9  .  <- stop (dock is at y = 3.0, 0.40 m in front of the marker)
    #          |  north 1.3 m up the corridor between bed_A and bed_B
    #   y 1.6  +-----------------------------+   east 2.725 m, above the trolley
    #          |                                (robot sides at y 1.45-1.75:
    #   y 0.6  S  north 1.0 m                    trolley top 1.2, beds start at 2.0)
    #         x 0.6                          x 3.325

# segment ends when own odometry (endcoders) says has driven/turned far enough

# marker is out of view until final northward segment (EKF runs on odometry alone for most of run - convariencde will grow then shrink once marker updates)



# Rates (physics dt = 2 ms); the camera and LiDAR rates are multiples of the
# encoder rate, so every measurement lands exactly on an encoder sample:
#     encoders  every 10 steps  -> 50 Hz  (= Person 2's EKF_EVERY; also the control rate)
#     camera    every 20 steps  -> 25 Hz
#     LiDAR     every 100 steps -> 5 Hz   (the LDS-01 rotation rate)

# Saved file: logs/run_01.npz
#   enc_t  (N,)          time of each encoder sample [s]; sample 0 is at t = 0
#   enc    (N, 2)        cumulative measured wheel angles [phiR, phiL] [rad]  -> EKF predict
#   cmd    (N, 2)        commanded wheel speeds [wR, wL] [rad/s] (targets, not actual)
#   cam_k  (M,)          index into enc_t of each camera sample
#   cam_z  (M, 2)        measured [range m, bearing rad]; NaN = not detected  -> EKF update
#   cam_view_angle (M,)  marker viewing angle [rad] (for the HD study)
#   scan_k (K,)          index into enc_t of each scan
#   scan_angles (B,)     beam angles in the lidar frame [rad]
#   scan_ranges (K, B)   ranges [m]; inf = no return, NaN = invalid
#   map_logodds, map_origin, map_res                                        -> Person 3
#   --- EVALUATION ONLY (truth from the simulator; never fed to the EKF or controller) ---
#   true_pose  (N, 3)    true [x, y, theta] at each encoder sample
#   cam_z_true (M, 2)    true range and bearing computed from the true pose
#   meta                 JSON: rates, noise parameters, marker, start pose, trolley, plan
"""
How Person 2 replays it (predict, then update, in time order):

    log = load_run("logs/run_01.npz"); m = log["meta"]
    ekf = EKF(m["start_pose"], P0, m["enc_noise_k"], R_RB)    # known start: the pharmacy
    cam_at = {int(k): z for k, z in zip(log["cam_k"], log["cam_z"])}
    for i in range(1, len(log["enc_t"])):
        dR, dL = log["enc"][i] - log["enc"][i - 1]
        ekf.predict(dR, dL)
        z = cam_at.get(i)
        if z is not None and np.all(np.isfinite(z)):
            ekf.update_range_bearing(z, m["marker_xy"])
        # compare ekf.x with log["true_pose"][i] for the plots
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import mujoco

import frames as F
from Build_scene import MARKER_X, MARKER_Y, MARKER_ID, MARKER_SIDE, DOCK_POSITION
from Sensing.driver import TurtleBotDriver
from Sensing.marker_detection import marker_range_bearing
from Sensing.lidar_map import OccupancyGrid

ENC_EVERY, CAM_EVERY, LIDAR_EVERY = 10, 20, 100
DEFAULT_LOG = F.ROOT / "logs" / "run_01.npz"
T_MAX = 90.0                        # safety stop [s]

# (kind, amount): "turn" in rad (+ = left), "drive" in m, "pause" in s
PLAN = [
    ("pause", 0.5),
    ("turn", +np.pi / 2),           # face north
    ("drive", 1.0),                 # y 0.6 -> 1.6, west of the trolley
    ("turn", -np.pi / 2),           # face east
    ("drive", MARKER_X - F.START_POSE[0]),               # x 0.6 -> 3.325, above the trolley
    ("turn", +np.pi / 2),           # face north, marker ahead
    ("drive", (MARKER_Y - DOCK_POSITION - 0.1) - 1.6),   # stop 0.1 m short of the dock point
    ("pause", 1.0),
]


class OdometrySequencer:
    """
    Run PLAN using wheel odometry only.
    Inside each segment: distance driven = r/2 * sum(dphi) and heading change
    = r/b * difference(dphi), with both increments measured since the segment
    started. These are the same kinematics as Person 2's motion model.
    """
    V_MAX, V_MIN = 0.20, 0.06       # m/s (the minimum keeps the wheels above the friction deadband)
    W_MAX, W_MIN = 1.0, 0.5         # rad/s
    TOL_D, TOL_TH = 0.005, 0.01     # m, rad
    SETTLE = 0.3                    # s of zero command after each move
    SEG_TIMEOUT = 25.0              # s; a segment taking longer than this means the robot is stalled

    def __init__(self, plan):
        self.plan, self.i = list(plan), -1
        self.done = False
        self.stalled = None             # (segment index, segment) if the plan was aborted
        self._enc0 = self._t0 = None
        self._settling_until = None

    def _start_next(self, enc, t):
        self.i += 1
        self._settling_until = None
        if self.i >= len(self.plan):
            self.done = True
        self._enc0, self._t0 = enc.copy(), t

    def command(self, enc, t):
        """Return (v, omega) for the current encoder reading."""
        if self._enc0 is None:
            self._start_next(enc, t)
        if self.done:
            return 0.0, 0.0
        if self._settling_until is not None:
            if t >= self._settling_until:
                self._start_next(enc, t)
                return self.command(enc, t)
            return 0.0, 0.0

        kind, amount = self.plan[self.i]
        if kind != "pause" and t - self._t0 > self.SEG_TIMEOUT:
            # Stalled: stop instead of grinding against something for the rest of the run
            self.stalled = (self.i, self.plan[self.i])
            self.done = True
            return 0.0, 0.0
        dR, dL = enc - self._enc0
        dist = F.WHEEL_RADIUS / 2 * (dR + dL)
        dth = F.WHEEL_RADIUS / F.WHEEL_BASE * (dR - dL)

        if kind == "pause":
            if t - self._t0 >= amount:
                self._start_next(enc, t)
                return self.command(enc, t)
            return 0.0, 0.0
        if kind == "drive":
            rem = amount - dist
            if rem <= self.TOL_D:
                self._settling_until = t + self.SETTLE
                return 0.0, 0.0
            v = float(np.clip(0.8 * rem, self.V_MIN, self.V_MAX))
            return v, -2.0 * dth             # keep the odometry heading straight
        # turn
        rem = amount - dth
        if abs(rem) <= self.TOL_TH:
            self._settling_until = t + self.SETTLE
            return 0.0, 0.0
        return 0.0, float(np.sign(rem) * np.clip(1.5 * abs(rem), self.W_MIN, self.W_MAX))


def true_range_bearing(pose):
    """The noise-free version of Person 2's measurement model h(x). EVALUATION ONLY."""
    dx, dy = MARKER_X - pose[0], MARKER_Y - pose[1]
    return np.array([np.hypot(dx, dy), F.wrap(np.arctan2(dy, dx) - pose[2])])


def record(out=DEFAULT_LOG, seed=0, **driver_kw):
    drv = TurtleBotDriver(seed=seed, **driver_kw)
    drv.reset(F.START_POSE)
    calib = (drv.K, drv.R_BC, drv.t_BC)
    grid = OccupancyGrid(max_range=drv.lidar_max)
    seq = OdometrySequencer(PLAN)

    keys = ("enc_t", "enc", "cmd", "true_pose", "cam_k", "cam_z", "cam_z_true",
            "cam_view_angle", "scan_k", "scan_ranges")
    L = {k: [] for k in keys}

    def log_encoders():
        enc = drv.read_encoders()
        L["enc_t"].append(drv.time)
        L["enc"].append(enc)
        L["true_pose"].append(drv.true_pose())          # evaluation only
        return len(L["enc_t"]) - 1, enc

    i, enc = log_encoders()                             # sample 0 at t = 0
    drv.set_velocity(*seq.command(enc, drv.time))
    L["cmd"].append(drv.last_cmd.copy())
    step = 0
    while not seq.done and drv.time < T_MAX:
        drv.step()
        step += 1
        if step % ENC_EVERY:
            continue                                    # control and sensing run at 50 Hz
        i, enc = log_encoders()
        drv.set_velocity(*seq.command(enc, drv.time))   # odometry only, no truth
        L["cmd"].append(drv.last_cmd.copy())
        pose_true = L["true_pose"][-1]

        if step % CAM_EVERY == 0:
            z, det = marker_range_bearing(drv.read_camera(), calib=calib, return_details=True)
            L["cam_k"].append(i)
            L["cam_z"].append(z if z is not None else np.full(2, np.nan))
            L["cam_view_angle"].append(det["viewing_angle"] if det else np.nan)
            L["cam_z_true"].append(true_range_bearing(pose_true))

        if step % LIDAR_EVERY == 0:
            angles, ranges = drv.read_lidar()
            L["scan_k"].append(i)
            L["scan_ranges"].append(ranges)
            # Mapping with KNOWN poses for this offline Stage 1 test.
            # In Stage 2, pass F.lidar_pose(ekf.x) instead.
            grid.update(angles, ranges, F.lidar_pose(pose_true))

    if seq.stalled is not None:
        print(f"WARNING: robot stalled in segment {seq.stalled[0]}: {seq.stalled[1]} "
              f"- the wheel motors cannot overcome the load (check kv in Build_scene)")
    elif not seq.done:
        print(f"WARNING: plan not finished by T_MAX (stuck at segment {seq.i}: {PLAN[seq.i]})")

    gid = mujoco.mj_name2id(drv.model, mujoco.mjtObj.mjOBJ_GEOM, "trolley")
    meta = {
        "dt": drv.dt,
        "rates_hz": {"enc": 1 / (ENC_EVERY * drv.dt), "cam": 1 / (CAM_EVERY * drv.dt),
                     "lidar": 1 / (LIDAR_EVERY * drv.dt)},
        "enc_noise_k": drv.enc_noise_k, "wheel_radius": F.WHEEL_RADIUS,
        "wheel_base": F.WHEEL_BASE, "marker_xy": [MARKER_X, MARKER_Y],
        "marker_id": MARKER_ID, "marker_side": MARKER_SIDE,
        "start_pose": list(F.START_POSE), "plan": [[k, float(a)] for k, a in PLAN],
        "seed": seed,
        "trolley_xy": drv.model.geom_pos[gid][:2].tolist() if gid >= 0 else None,
        "lidar": {"min": drv.lidar_min, "max": drv.lidar_max,
                  "noise": drv.lidar_noise, "dropout": drv.lidar_dropout},
    }
    arrays = {k: np.asarray(v) for k, v in L.items()}
    arrays.update(scan_angles=drv.lidar_angles, map_logodds=grid.L,
                  map_origin=grid.origin, map_res=np.array(grid.res),
                  meta=np.array(json.dumps(meta)))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    drv.close()

    det = np.isfinite(arrays["cam_z"][:, 0])
    print(f"Saved {out}")
    print(f"  {len(arrays['enc_t'])} encoder samples over {arrays['enc_t'][-1]:.1f} s")
    print(f"  {det.sum()}/{det.size} camera frames with marker {MARKER_ID} detected")
    print(f"  {len(arrays['scan_k'])} LiDAR scans")
    print(f"  final true pose {np.round(arrays['true_pose'][-1], 3)} "
          f"(dock point {MARKER_X}, {MARKER_Y - DOCK_POSITION:.3f})")
    return out


def load_run(path=DEFAULT_LOG):
    """Load a run into a dict, with meta parsed back from JSON."""
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_LOG))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-noise", action="store_true",
                    help="perfect encoders and LiDAR (useful when debugging the EKF)")
    a = ap.parse_args()
    kw = {"enc_noise_k": 0.0, "lidar_noise": False} if a.no_noise else {}
    record(a.out, seed=a.seed, **kw)
