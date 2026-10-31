# Loads/record logs/run_01.npz and checks each sensing output against the simulator truth:
#   (a) the LiDAR occupancy map against the true obstacle outlines from Build_scene
#   (b) encoder dead reckoning against the true path. This sanity-checks the right/left order, the signs, r and b. If R and L were swapped, the
#       90-degree left turn would come out as a right turn.
#   (c) measured marker range against the true range
#   (d) measured marker bearing against the true bearing. A constant bearing offset here means the camera-to-base transform is wrong.

# Run:  python Sensing/test_sensing.py            (uses the existing log)
#       python Sensing/test_sensing.py --record   (records a fresh one first)

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

#this script belongs to the classic ward. it stops (instead of quietly overwriting the
#classic results) if the paediatric ward is selected
from Build_scene import WARD as _WARD
if _WARD != "classic" and __name__ == "__main__":
    raise SystemExit("test_sensing.py checks a log recorded in the classic ward: run it with --ward classic")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import frames as F
from Build_scene import OBSTACLES, ROOM_X, ROOM_Y, WALL_THICKNESS
from Sensing.record_run import record, load_run, DEFAULT_LOG
from Sensing.lidar_map import OccupancyGrid


def obstacle_rects():
    """Ground-truth (name, x0, y0, x1, y1) of the walls and obstacles, for plotting only."""
    t = WALL_THICKNESS
    walls = [("wall_S", -t, -t, ROOM_X + t, 0.0), ("wall_N", -t, ROOM_Y, ROOM_X + t, ROOM_Y + t),
             ("wall_W", -t, 0.0, 0.0, ROOM_Y), ("wall_E", ROOM_X, 0.0, ROOM_X + t, ROOM_Y)]
    return walls + [o[:5] for o in OBSTACLES]


def blocked(p, q, boxes):
    """True if the 2D segment p->q passes through any box (x0, y0, x1, y1) - slab test.
    All ward obstacles (0.5-0.9 m tall) are taller than the camera (0.107 m) and
    the marker centre (0.15 m), so crossing one in plan view means it is hidden."""
    d = q - p
    for x0, y0, x1, y1 in boxes:
        t0, t1 = 0.0, 1.0
        for ax, lo, hi in ((0, x0, x1), (1, y0, y1)):
            if abs(d[ax]) < 1e-12:
                if not lo <= p[ax] <= hi:
                    break
            else:
                ta, tb = (lo - p[ax]) / d[ax], (hi - p[ax]) / d[ax]
                t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
                if t0 > t1:
                    break
        else:
            return True
    return False


def dead_reckon(enc, start):
    """Integrate encoder increments with the midpoint motion model (Person 2's predict, without P)."""
    x = np.array(start, dtype=float)
    out = [x.copy()]
    for dR, dL in np.diff(enc, axis=0):
        ds = F.WHEEL_RADIUS / 2 * (dR + dL)
        dth = F.WHEEL_RADIUS / F.WHEEL_BASE * (dR - dL)
        phi = x[2] + dth / 2
        x = x + [ds * np.cos(phi), ds * np.sin(phi), dth]
        x[2] = F.wrap(x[2])
        out.append(x.copy())
    return np.array(out)


def main(log_path=DEFAULT_LOG, rerecord=False):
    if rerecord or not Path(log_path).exists():
        record(log_path)
    log = load_run(log_path)
    meta = log["meta"]
    t, truth = log["enc_t"], log["true_pose"]

    # --- (b) dead reckoning --------------------------------------------------
    dr = dead_reckon(log["enc"], meta["start_pose"])
    dr_err = np.linalg.norm(dr[-1, :2] - truth[-1, :2])

    # --- (c, d) marker detections -------------------------------------------
    ct = t[log["cam_k"]]
    z, zt = log["cam_z"], log["cam_z_true"]
    det = np.isfinite(z[:, 0])
    K = F.get_camera_calibration()[0]
    half_hfov = np.arctan(F.IMG_W / 2 / K[0, 0])
    # "should be detectable" = inside the field of view AND not hidden behind
    # a bed, the trolley or the cabinet (evaluation only: uses the true pose)
    in_fov = (np.abs(zt[:, 1]) < 0.9 * half_hfov) & (zt[:, 0] < 3.5)
    marker = np.array(meta["marker_xy"])
    boxes = [o[1:5] for o in OBSTACLES]
    cam_pose = truth[log["cam_k"]]
    cam_xy = cam_pose[:, :2] + F.CAMERA_OFFSET[0] * np.column_stack(
        [np.cos(cam_pose[:, 2]), np.sin(cam_pose[:, 2])])
    visible = np.array([not blocked(c, marker, boxes) for c in cam_xy])
    in_view = in_fov & visible
    print(f"In field of view: {in_fov.sum()} frames, of which {(in_fov & ~visible).sum()} "
          f"are hidden behind an obstacle")
    rate = det[in_view].mean() if in_view.any() else np.nan
    e_r = z[det, 0] - zt[det, 0]
    e_b = F.wrap(z[det, 1] - zt[det, 1])

    # --- (a) map ------------------------------------------------------------
    grid = OccupancyGrid.from_arrays(log["map_logodds"], log["map_origin"], log["map_res"])
    plan = grid.to_planner()

    # --- figure -------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0, 0]
    a.imshow(grid.probability(), origin="lower", extent=grid.extent(),
             cmap="gray_r", vmin=0, vmax=1)
    a.contour(plan["blocked"].astype(float), levels=[0.5], origin="lower",
              extent=grid.extent(), colors="orange", linewidths=0.8)
    for name, x0, y0, x1, y1 in obstacle_rects():
        a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="r", ls="--", lw=0.8))
    a.plot(truth[:, 0], truth[:, 1], "b-", lw=1.2, label="true path")
    a.plot(*meta["marker_xy"], "g*", ms=14, label="ArUco marker")
    a.set_title("(a) LiDAR occupancy map: red = true outlines, orange = inflated")
    a.set_aspect("equal")
    a.legend(loc="lower right", fontsize=8)

    a = ax[0, 1]
    a.plot(truth[:, 0], truth[:, 1], "b-", label="true")
    a.plot(dr[:, 0], dr[:, 1], "r--", label="encoder dead reckoning")
    a.set_title(f"(b) Dead reckoning: final error {dr_err * 100:.1f} cm")
    a.set_aspect("equal")
    a.legend(fontsize=8)
    a.set_xlabel("x [m]")
    a.set_ylabel("y [m]")

    a = ax[1, 0]
    a.plot(ct, zt[:, 0], "k-", lw=1, label="true")
    a.plot(ct[det], z[det, 0], "r.", ms=3, label="measured")
    a.set_title(f"(c) Range: bias {e_r.mean() * 100:+.2f} cm, std {e_r.std() * 100:.2f} cm")
    a.set_xlabel("t [s]")
    a.set_ylabel("range [m]")
    a.legend(fontsize=8)

    a = ax[1, 1]
    a.plot(ct, np.rad2deg(zt[:, 1]), "k-", lw=1, label="true")
    a.plot(ct[det], np.rad2deg(z[det, 1]), "r.", ms=3, label="measured")
    a.set_title(f"(d) Bearing: bias {np.rad2deg(e_b.mean()):+.2f} deg, "
                f"std {np.rad2deg(e_b.std()):.2f} deg")
    a.set_xlabel("t [s]")
    a.set_ylabel("bearing [deg]")
    a.legend(fontsize=8)
    fig.tight_layout()

    out_png = Path(log_path).with_name("sensing_test.png")
    fig.savefig(out_png, dpi=150)

    # --- pass/check summary -------------------------------------------------
    checks = [
        ("detection rate while in view > 90%", rate > 0.9, f"{rate * 100:.0f}%"),
        ("|range bias| < 3 cm", abs(e_r.mean()) < 0.03, f"{e_r.mean() * 100:+.2f} cm"),
        ("|bearing bias| < 1 deg", abs(np.rad2deg(e_b.mean())) < 1, f"{np.rad2deg(e_b.mean()):+.2f} deg"),
        ("dead-reckoning final error < 10 cm", dr_err < 0.10, f"{dr_err * 100:.1f} cm"),
        ("map has occupied cells", (grid.classify() == 1).any(), ""),
    ]
    for name, ok, val in checks:
        print(f"[{'PASS' if ok else 'CHECK'}] {name:40s} {val}")
    print(f"Figure saved to {out_png}")
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    main(a.log, a.record)
