
# run_delivery.py - the whole medication round, all three parts together

#the robot visits every stop in the ward's MISSION in order, parking in front of each
#stop's ArUco marker, then finishes at the last one:
#   paediatric ward (default): pharmacy -> bed 1 -> bed 3 -> bed 4 -> bed 2 -> charging dock
#   classic ward (--ward classic): pharmacy = the one bedside (the original single delivery)
#
#run it from the repo folder (after python Build_scene.py once):
#   python run_delivery.py              #run, print results, save delivery_result.png
#   python run_delivery.py --view       #also watch it in the mujoco viewer
#   python run_delivery.py --map known  #plan on the obstacle list instead of the lidar map
#   python run_delivery.py --condition push --view    #try one of the robustness conditions
#   python run_delivery.py --ward classic              #the original 6 x 4 m ward
#
#every 20 ms the loop does the same things:
#   1. read the encoders = EKF predict
#   2. every 100 ms, look for markers = chi-squared gate -> EKF update (each marker seen)
#   3. every 200 ms, read a lidar scan= update the map, replan if the path is now blocked
#   4. LQR picks v and w from the EKF ESTIMATE
#   5. send v, w to the wheels and let 20 ms of physics happen
#the simulator's true pose is only used for logging scoring and injecting disturbances

import argparse
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import mujoco
import numpy as np

import frames as F
from Build_scene import SCENE_OUT
from Sensing.driver import TurtleBotDriver
from Sensing.lidar_map import OccupancyGrid as LidarMap
from Sensing.marker_detection import detect_markers
from Estimation.ekf import EKF
from Control.nav_settings import (PHARMACY_POSE, BEDSIDE_PARK_POSE, MARKER_XY, DT, STOPS,
                                  MARKER_POSITIONS, MARKER_FACING,
                                  PHYSICS_STEPS_PER_CONTROL, LIDAR_EVERY, CAMERA_EVERY,
                                  EKF_P0, EKF_K_ENC, NIS_GATE, marker_noise,
                                  correct_marker_reading, PROCESS_NOISE_SCALE, PROCESS_NOISE_BASE,
                                  GATE_RECOVERY_AFTER, GATE_RECOVERY_P)
from Control.planner import (build_grid_from_lidar, build_known_grid, plan_to_bedside,
                             path_blocked)
from Control.trajectory import build_ghost_robot
from Control.lqr_tracker import (LQRTracker, controllability_rank, tracking_error,
                                 Q_TRANSIT, R_TRANSIT, Q_DOCK, R_DOCK)


# =====================================================================================#
# ------------------------------- TRIAL CONDITIONS ------------------------------------ #
# =====================================================================================#

@dataclass
class TrialConditions:
    """
    everything that can change between trials. the defaults are the nominal run.
    disturbances are applied to the WORLD (using the true pose where needed, like a
    real push would), never handed to the estimator or controller
    """
    name: str = "nominal"
    seed: int = 0
    start_pose: tuple = None              #None = the pharmacy pad
    map_source: str = "lidar"             #"lidar" or "known"
    enc_noise_scale: float = 1.0          #real encoder noise x this (EKF still assumes 1x)
    lidar_dropout: float = 0.0            #chance each lidar beam returns nothing
    wheel_radius_error: float = 0.0       #EKF thinks r is 0.033 * (1 + this)
    push_force_n: float = 0.0             #sideways shove on the robot (N), 0 = none
    push_at_s: float = 8.0                #when, counted from the start of tracking
    push_duration_s: float = 0.3
    move_trolley_to: tuple = None         #(x, y) the trolley is wheeled to once tracking starts
    marker_hidden_within_m: float = 0.0   #someone stands in front of the marker this close in
    camera_noise: bool = False            #grainy, blurry, darker image like a real pi camera
    outlier_rate: float = 0.0             #chance a marker reading is badly wrong
    gating: bool = True                   #chi-squared rejection of bad marker readings
    gate_recovery: bool = True            #after GATE_RECOVERY_AFTER rejections in a row, assume the
                                          #EKF is lost (not the camera) and widen its uncertainty
    r_model: str = "fitted"               #"fitted" (measured) or "fixed" (Person 2's assumed)
    q_scale: float = 1.0                  #multiplies both LQR Q matrices (weight sweep)
    process_noise: float = PROCESS_NOISE_SCALE   #extra EKF process noise, x the base level (0 = none)
    max_time: float = None                #give up after this many seconds. None = 120 s per stop
    deliver_pause_s: float = 1.5          #how long the robot waits at each bedside (not the last stop)


def degrade_image(img, rng):
    #make the perfect simulated image look more like a real camera: darker, blurred, grainy
    out = cv2.GaussianBlur(img, (5, 5), 1.2).astype(np.float32) * 0.75
    out += rng.normal(0.0, 8.0, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def marker_maybe_visible(pose_estimate, marker_id=None, max_range=3.0, half_fov=np.deg2rad(40)):
    """
    only render the camera when the EKF thinks a marker could be in view.
    rendering is the slowest thing in the loop, and a real robot wouldnt waste
    time looking for a marker thats behind it. half_fov is wider than the
    camera's real 31 deg so an estimate thats a bit off still looks.
    marker_id=None means the final stop's marker
    """
    marker_xy = MARKER_XY if marker_id is None else MARKER_POSITIONS[marker_id]
    facing = MARKER_FACING[STOPS[-1]["marker_id"] if marker_id is None else marker_id]
    dx, dy = marker_xy - pose_estimate[:2]
    #the robot must be in front of the marker's face (the classic marker faces south, so the
    #robot must be south of it). the face direction is rounded so south is exactly (0, -1)
    nx, ny = round(np.cos(facing), 12), round(np.sin(facing), 12)
    if np.hypot(dx, dy) > max_range or -(dx * nx + dy * ny) <= 0:
        return False
    return abs(F.wrap(np.arctan2(dy, dx) - pose_estimate[2])) < half_fov


def marker_innovation(ekf, z, marker_xy=MARKER_XY):
    #the same maths as ekf.update_range_bearing, without changing the state, so a
    #reading can be checked (NIS) before its allowed in
    dx, dy = marker_xy - ekf.x[:2]
    q = dx ** 2 + dy ** 2
    z_hat = np.array([np.sqrt(q), F.wrap(np.arctan2(dy, dx) - ekf.x[2])])
    H = np.array([[-dx / np.sqrt(q), -dy / np.sqrt(q), 0], [dy / q, -dx / q, -1]])
    nu = np.asarray(z) - z_hat
    nu[1] = F.wrap(nu[1])
    S = H @ ekf.P @ H.T + ekf.R_rb
    return nu, S


def score_stop(stop, true_pose, clock):
    #how well the robot parked at one stop, from the TRUE pose (scoring only, never fed back)
    park = stop["park"]
    d_xy = true_pose[:2] - park[:2]
    c, s = np.cos(park[2]), np.sin(park[2])
    return {"name": stop["name"], "label": stop["label"], "t": clock, "error_m": float(np.hypot(*d_xy)),
            "along_m": float(c * d_xy[0] + s * d_xy[1]), "sideways_m": float(-s * d_xy[0] + c * d_xy[1]),
            "heading_deg": float(np.degrees(F.wrap(true_pose[2] - park[2])))}


def remaining_path(ghost, t):
    k = int(np.searchsorted(ghost.t, t))
    return np.column_stack([ghost.x[k:], ghost.y[k:]])


# =====================================================================================#
# MAIN stuff
# =====================================================================================#

def run_delivery(cond=None, view=False, verbose=True, watch=None, stops=None):
    """
    one full round. returns (results, run_log, marker_log, paths, replans, lidar_map).
    watch: optional function called after every 20 ms tick with a dict describing the moment
           (make_video.py uses it to film the run). stops: override the ward's round
    """
    cond = cond or TrialConditions()
    stops = STOPS if stops is None else stops
    max_time = cond.max_time if cond.max_time is not None else 120.0 * len(stops)
    say = print if verbose else (lambda *a, **k: None)
    rng = np.random.default_rng(10_000 + cond.seed)        #disturbance randomness
    start_pose = np.array(cond.start_pose if cond.start_pose is not None else PHARMACY_POSE, float)

    driver = TurtleBotDriver(SCENE_OUT, seed=cond.seed, enc_noise_k=EKF_K_ENC * cond.enc_noise_scale,
                             lidar_dropout=cond.lidar_dropout)
    driver.reset(start_pose)
    m, d = driver.model, driver.data

    #the EKF is told where it starts (the robot is placed on a known spot), nothing after that
    r_believed = F.WHEEL_RADIUS * (1 + cond.wheel_radius_error)
    ekf = EKF(start_pose, EKF_P0, EKF_K_ENC, marker_noise(1.0, cond.r_model), r=r_believed)
    lidar_map = LidarMap()
    last_encoders = driver.read_encoders()

    #for scoring only: which geoms are the robot, which are furniture
    robot_root = m.body_rootid[driver.base_body]
    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    trolley_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "trolley")
    marker_ids = sorted(MARKER_POSITIONS)

    viewer = None
    if view:
        from mujoco import viewer as mujoco_viewer
        viewer = mujoco_viewer.launch_passive(m, d)

    run_log = {k: [] for k in ("t", "true", "est", "sigma", "nees", "phase", "stop",
                               "track_err_est", "track_err_true", "effort")}
    marker_log = {"t": [], "z": [], "nis": [], "id": [], "rejected_t": [], "recovery_t": []}
    stats = {"collision_ticks": 0, "first_collision": None, "outliers_injected": 0}
    paths, path_leg, replans, stop_log = [], [], [], []
    state = {"step": 0, "clock": 0.0, "track_clock": None, "rejects_in_a_row": 0, "leg": 0, "path": None}

    def check_collision():
        for c in d.contact[:d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            r1 = m.body_rootid[m.geom_bodyid[g1]] == robot_root
            r2 = m.body_rootid[m.geom_bodyid[g2]] == robot_root
            if r1 != r2:
                other = g2 if r1 else g1
                if other != floor_id:
                    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom {other}"
        return None

    def control_tick(v, w, phase):
        nonlocal last_encoders
        loop_start_time = time.time()

        #disturbance: a sideways push (left of the robot's TRUE heading), applied to the world
        tc = state["track_clock"]
        if (cond.push_force_n and tc is not None and state["leg"] == 0
                and cond.push_at_s <= tc < cond.push_at_s + cond.push_duration_s):
            th = driver.true_pose()[2]
            d.xfrc_applied[driver.base_body, :3] = cond.push_force_n * np.array([-np.sin(th), np.cos(th), 0])
        else:
            d.xfrc_applied[driver.base_body, :3] = 0.0

        driver.set_velocity(v, w)
        driver.step(PHYSICS_STEPS_PER_CONTROL)
        state["step"] += 1
        state["clock"] += DT

        #1. EKF predict from how far the wheels turned
        encoders = driver.read_encoders()
        d_right, d_left = encoders - last_encoders
        last_encoders = encoders
        ekf.predict(d_right, d_left)
        #extra process noise for what the encoder model leaves out (wheel slip, the robot
        #rocking, linearisation). keeps the EKF honest about its uncertainty (see nees)
        ekf.P += (cond.process_noise ** 2) * PROCESS_NOISE_BASE * DT

        #2. markers -> gate -> EKF update, only when a marker could be in view
        seen_now, rejected_now = [], []
        candidates = ([i for i in marker_ids if marker_maybe_visible(ekf.x, i)]
                      if state["step"] % CAMERA_EVERY == 0 else [])
        if candidates:
            #a person standing in front of a marker (marker_hidden test) blocks it within this range
            true_xy = driver.true_pose()[:2]
            blocked = {i for i in marker_ids
                       if np.hypot(*(MARKER_POSITIONS[i] - true_xy)) < cond.marker_hidden_within_m}
            found = {}
            if any(i not in blocked for i in candidates):          #no point rendering a blocked view
                img = driver.read_camera()
                if cond.camera_noise:
                    img = degrade_image(img, rng)
                found = detect_markers(img, marker_ids, calib=(driver.K, driver.R_BC, driver.t_BC))
            for marker_id in sorted(found):
                #only markers the EKF expects to see within 3 m: the camera calibration
                #(Analysis/marker_noise.py) only covers 0.4-3 m, so a marker read from across the
                #room has unknown accuracy and is not used
                if marker_id in blocked or marker_id not in candidates:
                    continue
                marker_xy = MARKER_POSITIONS[marker_id]
                z = correct_marker_reading(found[marker_id], cond.r_model)   #calibration: remove the measured bias
                if cond.outlier_rate and rng.random() < cond.outlier_rate:
                    z += [rng.uniform(-0.5, 0.5), rng.uniform(-0.35, 0.35)]
                    stats["outliers_injected"] += 1
                ekf.R_rb = marker_noise(z[0], cond.r_model)
                nu, S = marker_innovation(ekf, z, marker_xy)
                nis = float(nu @ np.linalg.solve(S, nu))
                if cond.gating and nis > NIS_GATE:
                    marker_log["rejected_t"].append(state["clock"])
                    rejected_now.append(marker_id)
                    state["rejects_in_a_row"] += 1
                    #one bad reading happens (outlier). several in a row means the EKF is the one
                    #thats wrong (it got knocked off and doesnt know), so it widens its uncertainty
                    #and lets the next reading pull it back
                    if cond.gate_recovery and state["rejects_in_a_row"] >= GATE_RECOVERY_AFTER:
                        ekf.P = ekf.P + GATE_RECOVERY_P
                        marker_log["recovery_t"].append(state["clock"])
                        state["rejects_in_a_row"] = 0
                else:
                    state["rejects_in_a_row"] = 0
                    ekf.update_range_bearing(z, marker_xy)
                    marker_log["t"].append(state["clock"])
                    marker_log["z"].append(z)
                    marker_log["nis"].append(nis)
                    marker_log["id"].append(marker_id)
                    seen_now.append(marker_id)

        #3. lidar into the map, at the ESTIMATED pose
        scanned = False
        if state["step"] % LIDAR_EVERY == 0:
            angles, ranges = driver.read_lidar()
            lidar_map.update(angles, ranges, F.lidar_pose(ekf.x))
            scanned = True

        #scoring (truth allowed here, never fed back)
        true_pose = driver.true_pose()
        err = ekf.x - true_pose
        err[2] = F.wrap(err[2])
        run_log["t"].append(state["clock"])
        run_log["true"].append(true_pose)
        run_log["est"].append(ekf.x.copy())
        run_log["sigma"].append(np.sqrt(np.diag(ekf.P)))
        run_log["nees"].append(float(err @ np.linalg.solve(ekf.P, err)))
        run_log["phase"].append(phase)
        run_log["stop"].append(state["leg"])
        hit = check_collision()
        if hit:
            stats["collision_ticks"] += 1
            if stats["first_collision"] is None:
                stats["first_collision"] = (state["clock"], hit, float(np.hypot(*err[:2])))

        if watch is not None:
            leg = min(state["leg"], len(stops) - 1)
            watch({"model": m, "data": d, "clock": state["clock"], "phase": phase, "leg": leg,
                   "stop": stops[leg], "stops": stops, "ekf_x": ekf.x.copy(), "ekf_P": ekf.P.copy(),
                   "path": state["path"], "seen": seen_now, "rejected": rejected_now,
                   "done": [s_["name"] for s_ in stop_log]})

        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, DT - (time.time() - loop_start_time)))
            if not viewer.is_running():
                raise KeyboardInterrupt
        return scanned

    def current_grid():
        return build_known_grid() if cond.map_source == "known" else build_grid_from_lidar(lidar_map)

    def make_tracker(ghost):
        return LQRTracker(ghost, DT, Q_transit=cond.q_scale * Q_TRANSIT, R_transit=R_TRANSIT,
                          Q_dock=cond.q_scale * Q_DOCK, R_dock=R_DOCK)

    error_msg = None
    reason = None
    try:
        #phase 0: stand still for half a second so the lidar sees the room
        for _ in range(25):
            control_tick(0.0, 0.0, "scan")

        for leg, stop in enumerate(stops):
            state["leg"] = leg
            park = stop["park"]
            reason = None
            path = plan_to_bedside(current_grid(), ekf.x, park)
            ghost = build_ghost_robot(path)
            paths.append(path)
            path_leg.append(leg)
            state["path"] = path
            say(f"[{stop['label']}] planned route: {np.sum(np.hypot(*np.diff(path, axis=0).T)):.2f} m, "
                f"{ghost.duration:.1f} s")

            #phase 1: turn on the spot to face along the route, gently so the wheels dont skid
            for _ in range(int(20 / DT)):
                heading_err = F.wrap(ghost.theta[0] - ekf.x[2])
                if abs(heading_err) < np.radians(2):
                    break
                control_tick(0.0, np.clip(1.5 * heading_err, -0.6, 0.6), "align")
            for _ in range(10):
                control_tick(0.0, 0.0, "align")

            #disturbance: a nurse wheels the trolley somewhere new as the robot sets off
            if leg == 0 and cond.move_trolley_to is not None:
                body = m.geom_bodyid[trolley_id]
                if body == 0:                            #classic ward: a loose geom in the world
                    m.geom_pos[trolley_id][:2] = cond.move_trolley_to
                else:                                    #paediatric ward: the whole trolley body
                    m.body_pos[body][:2] = cond.move_trolley_to

            #phase 2: track the route with LQR
            tracker = make_tracker(ghost)
            state["track_clock"] = 0.0
            while reason is None:
                v, w, reason = tracker.step(ekf.x, state["track_clock"])   #only ever the estimate

                xg, yg, thg, ghost_v, ghost_w = ghost.at(state["track_clock"])
                e_est = tracking_error(ekf.x, (xg, yg, thg))
                e_true = tracking_error(driver.true_pose(), (xg, yg, thg))
                run_log["track_err_est"].append(e_est)
                run_log["track_err_true"].append(e_true)
                run_log["effort"].append([ghost_v * np.cos(e_est[2]) - v, ghost_w - w])

                scanned = control_tick(v, w, "track")
                state["track_clock"] += DT

                #replan if a new scan shows the rest of the route now runs into something
                if scanned and cond.map_source == "lidar" and reason is None:
                    ahead = remaining_path(ghost, state["track_clock"])
                    if len(ahead) > 5 and path_blocked(current_grid(), ahead):
                        try:
                            path = plan_to_bedside(current_grid(), ekf.x, park, lead_in=0.15)
                        except (RuntimeError, ValueError) as err:
                            say(f"  replan failed at t={state['clock']:.1f}s ({err}), keeping the old route")
                        else:
                            ghost = build_ghost_robot(path, v_start=max(v, 0.0))
                            tracker = make_tracker(ghost)
                            state["track_clock"] = 0.0
                            paths.append(path)
                            path_leg.append(leg)
                            state["path"] = path
                            replans.append(state["clock"])
                            say(f"  replanned at t={state['clock']:.1f}s: new route {ghost.duration:.1f} s")
                if state["clock"] > max_time:
                    reason = "timeout"
            say(f"[{stop['label']}] stopped ({reason}) at t = {state['clock']:.1f} s")

            last = leg == len(stops) - 1
            if reason != "reached" or last:
                break
            #at a bedside: wait while the medication is handed over, then score the stop
            for _ in range(int(round(cond.deliver_pause_s / DT))):
                control_tick(0.0, 0.0, "deliver")
            stop_log.append(score_stop(stop, driver.true_pose(), state["clock"]))
            state["track_clock"] = None

        for _ in range(50):                                         #settle
            control_tick(0.0, 0.0, "stopped")
        if reason == "reached":
            stop_log.append(score_stop(stops[-1], driver.true_pose(), state["clock"]))
    except (RuntimeError, ValueError) as err:
        reason = "planner failed"
        error_msg = str(err)
        say(f"Planner failed: {err}")
    finally:
        if viewer is not None:
            viewer.close()

    run_log = {k: np.array(val) for k, val in run_log.items()}
    final_true = driver.true_pose()
    driver.close()
    final_park = stops[-1]["park"]
    d_xy = final_true[:2] - final_park[:2]
    c, s = np.cos(final_park[2]), np.sin(final_park[2])
    est_err = run_log["est"] - run_log["true"]
    est_err[:, 2] = F.wrap(est_err[:, 2])
    tracking = run_log["phase"] == "track"
    te_true = run_log["track_err_true"] if len(run_log["track_err_true"]) else np.zeros((1, 3))
    te_est = run_log["track_err_est"] if len(run_log["track_err_est"]) else np.zeros((1, 3))
    effort = run_log["effort"] if len(run_log["effort"]) else np.zeros((1, 2))

    results = {
        **{k: v for k, v in asdict(cond).items() if k != "start_pose"},
        "start_x": start_pose[0], "start_y": start_pose[1], "start_theta": start_pose[2],
        "reason": reason,
        "error": error_msg,
        "time_s": state["clock"],
        "dock_error_m": float(np.hypot(*d_xy)),
        "dock_along_m": float(c * d_xy[0] + s * d_xy[1]),
        "dock_sideways_m": float(-s * d_xy[0] + c * d_xy[1]),
        "dock_heading_deg": float(np.degrees(F.wrap(final_true[2] - final_park[2]))),
        "ekf_final_error_m": float(np.hypot(*est_err[-1, :2])),
        "ekf_rmse_pos_m": float(np.sqrt(np.mean(np.sum(est_err[:, :2] ** 2, axis=1)))),
        "ekf_rmse_heading_deg": float(np.degrees(np.sqrt(np.mean(est_err[:, 2] ** 2)))),
        "mean_nees": float(np.mean(run_log["nees"])),
        "mean_nees_after_marker": float(np.mean(run_log["nees"][run_log["t"] >= marker_log["t"][0]]))
                                  if marker_log["t"] else float("nan"),
        "mean_nis": float(np.mean(marker_log["nis"])) if marker_log["nis"] else float("nan"),
        "marker_updates": len(marker_log["t"]),
        "marker_rejected": len(marker_log["rejected_t"]),
        "gate_recoveries": len(marker_log["recovery_t"]),
        "outliers_injected": stats["outliers_injected"],
        "first_marker_s": marker_log["t"][0] if marker_log["t"] else float("nan"),
        "last_marker_s": marker_log["t"][-1] if marker_log["t"] else float("nan"),
        "replans": len(replans),
        "collision_ticks": stats["collision_ticks"],
        "collision_with": stats["first_collision"][1] if stats["first_collision"] else "",
        "collision_ekf_error_m": stats["first_collision"][2] if stats["first_collision"] else float("nan"),
        "rms_cross_track_true_m": float(np.sqrt(np.mean(te_true[:, 1] ** 2))),
        "rms_cross_track_est_m": float(np.sqrt(np.mean(te_est[:, 1] ** 2))),
        "rms_effort": float(np.sqrt(np.mean(np.sum(effort ** 2, axis=1)))),
    }
    results["max_time"] = max_time
    #every stop of the round, scored the same way as the final dock (5 cm, 5 deg)
    results["stops_total"] = len(stops)
    results["stops_reached"] = len(stop_log)
    results["stop_errors_cm"] = " ".join(f"{100 * r_['error_m']:.2f}" for r_ in stop_log)
    results["worst_stop_error_m"] = max((r_["error_m"] for r_ in stop_log), default=float("nan"))
    results["worst_stop_heading_deg"] = max((abs(r_["heading_deg"]) for r_ in stop_log), default=float("nan"))
    stops_ok = len(stop_log) == len(stops) and all(r_["error_m"] < 0.05 and abs(r_["heading_deg"]) < 5
                                                   for r_ in stop_log[:-1])
    results["success"] = bool(results["reason"] == "reached" and results["dock_error_m"] < 0.05
                              and abs(results["dock_heading_deg"]) < 5 and results["collision_ticks"] == 0
                              and stops_ok)
    run_log["path_leg"] = np.array(path_leg, dtype=int)
    run_log["stops"] = stop_log
    run_log["stop_plan"] = stops
    return results, run_log, marker_log, paths, replans, lidar_map


# =====================================================================================#
# ----------------------------------- PLOTTING ---------------------------------------- #
# =====================================================================================#

def plot_delivery(results, run_log, marker_log, paths, replans, lidar_map, out_file="delivery_result.png",
                  show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Build_scene import OBSTACLES

    fig = plt.figure(figsize=(15, 8.5))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])
    ax = fig.add_subplot(grid[:, 0])
    ax.imshow(lidar_map.probability(), origin="lower", extent=lidar_map.extent(),
              cmap="Greys", vmin=0, vmax=1)
    #routes: the one each leg finally drove in green, routes abandoned by a replan in grey
    legs = run_log.get("path_leg", np.zeros(len(paths), int))
    labelled = set()
    for i, p in enumerate(paths):
        final_for_leg = i == len(paths) - 1 or legs[i + 1] != legs[i]
        label = "planned route" if final_for_leg else "abandoned route (replan)"
        ax.plot(*p.T, "--", color="#2b8a3e" if final_for_leg else "0.45", lw=1.2,
                label=None if label in labelled else label)
        labelled.add(label)
    if len(run_log["est"]):
        ax.plot(*run_log["est"][:, :2].T, color="#2a6fdb", lw=1.5, label="EKF estimate")
        ax.plot(*run_log["true"][:, :2].T, color="#e8590c", lw=1.2, label="true position")
    ax.plot(results["start_x"], results["start_y"], "o", color="0.2", ms=8, label="start")
    plan = run_log.get("stop_plan", STOPS)
    scored = {r_["name"]: r_ for r_ in run_log.get("stops", [])}
    for k, stop in enumerate(plan):
        ax.plot(*stop["marker_xy"], "s", color="k", ms=7, label="marker" if k == 0 else None)
        px, py = stop["park"][:2]
        ok = stop["name"] in scored and scored[stop["name"]]["error_m"] < 0.05
        ax.annotate(f"{k + 1}", (px, py), ha="center", va="center", fontsize=8, color="white", weight="bold",
                    bbox=dict(boxstyle="circle,pad=0.25", fc="#2b8a3e" if ok else "#c92a2a", ec="none"))
    for name, x0, y0, x1, y1, *_ in OBSTACLES:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, name, ha="center", va="center", fontsize=7, color="#c92a2a")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    if len(plan) > 1:
        ax.set_title(f"{results['name']} (seed {results['seed']}): {results['stops_reached']}/{len(plan)} stops, "
                     f"worst {100 * results['worst_stop_error_m']:.1f} cm, "
                     f"{'success' if results['success'] else 'FAILED'}  (numbered circles = stops, green = within 5 cm)",
                     fontsize=9)
    else:
        ax.set_title(f"{results['name']} (seed {results['seed']}): docking error "
                     f"{100 * results['dock_error_m']:.1f} cm, {'success' if results['success'] else 'FAILED'}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=8, frameon=False)

    t = run_log["t"]
    err = run_log["est"] - run_log["true"]
    err[:, 2] = F.wrap(err[:, 2])
    scale = [100, 100, np.degrees(1)]
    labels = ["x error (cm)", "y error (cm)", "heading error (deg)"]
    for i in range(3):
        a = fig.add_subplot(grid[i, 1])
        a.fill_between(t, -3 * run_log["sigma"][:, i] * scale[i], 3 * run_log["sigma"][:, i] * scale[i],
                       color="#2a6fdb", alpha=0.2, label="±3σ")
        a.plot(t, err[:, i] * scale[i], color="#2a6fdb", lw=1, label="EKF error")
        for mt in marker_log["t"]:
            a.axvline(mt, color="#2b8a3e", alpha=0.08, lw=1)
        for mt in marker_log["rejected_t"]:
            a.axvline(mt, color="#e8590c", alpha=0.6, lw=1)
        for mt in marker_log.get("recovery_t", []):
            a.axvline(mt, color="#7048e8", lw=1.5)
        for rt in replans:
            a.axvline(rt, color="#c92a2a", ls=":", lw=1)
        a.axhline(0, color="k", lw=0.5)
        a.set_ylabel(labels[i])
        if i == 0:
            a.set_title("EKF error (green = marker used, orange = rejected, purple = gate recovery, "
                        "red dots = replan)", fontsize=8)
            a.legend(loc="upper left", fontsize=8)
    a.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_file, dpi=130)
    print(f"Saved {out_file}")
    if show:
        plt.show()
    plt.close(fig)


# =====================================================================================#
#main loop
# =====================================================================================#
if __name__ == "__main__":
    from Analysis.conditions import CONDITIONS

    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="watch it in the mujoco viewer")
    parser.add_argument("--map", choices=["lidar", "known"], default="lidar",
                        help="plan on the live lidar map (default) or the known obstacle list")
    parser.add_argument("--condition", choices=sorted(CONDITIONS), default="nominal",
                        help="one of the robustness conditions in Analysis/conditions.py")
    parser.add_argument("--seed", type=int, default=0, help="noise seed")
    args = parser.parse_args()

    print("Controllability rank moving:", controllability_rank(0.15, 0.0),
          "| stopped:", controllability_rank(0.0, 0.0))
    cond = TrialConditions(**{**CONDITIONS[args.condition], "name": args.condition,
                              "seed": args.seed, "map_source": args.map})
    try:
        out = run_delivery(cond, view=args.view)
    except KeyboardInterrupt:
        print("Viewer closed")
        sys.exit(0)
    r = out[0]
    for st in out[1]["stops"]:
        print(f"  {st['label']:<14} parked {100 * st['error_m']:.2f} cm off "
              f"(along {100 * st['along_m']:+.2f}, sideways {100 * st['sideways_m']:+.2f}), "
              f"heading {st['heading_deg']:+.2f} deg, at t = {st['t']:.1f} s")
    print(f"Docking error:    {100 * r['dock_error_m']:.1f} cm "
          f"(along {100 * r['dock_along_m']:+.1f}, sideways {100 * r['dock_sideways_m']:+.1f})")
    print(f"Heading error:    {r['dock_heading_deg']:+.1f} deg")
    print(f"EKF error at end: {100 * r['ekf_final_error_m']:.1f} cm, mean NEES {r['mean_nees']:.2f} (ideal 3), "
          f"mean NIS {r['mean_nis']:.2f} (ideal 2)")
    print(f"Marker: {r['marker_updates']} used, {r['marker_rejected']} rejected | replans {r['replans']} | "
          f"collisions {r['collision_ticks']} | {'SUCCESS' if r['success'] else 'FAILED'}")
    plot_delivery(*out, show=args.view)
