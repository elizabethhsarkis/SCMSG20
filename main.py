#run a bedside delivery in MuJoCo
#
#python main.py            # run, print results, save delivery_result.png
#python main.py --view     # also watch it in the MuJoCo viewer
#
#needs: pip install numpy scipy matplotlib mujoco
#all eight files must be in the same folder:
#config.py  scene.py  driver.py  odometry.py
#planner.py  trajectory.py  lqr_tracker.py  main.py
#
#every 20 ms the loop does the same five things:
#1. read the wheel encoders            (driver)
#2. update the pose estimate           (odometry, later the EKF)
#3. choose v and w from the ESTIMATE   (LQR tracker)
#4. send the wheel commands            (driver)
#5. let 20 ms of physics pass          (driver)
#the simulator's true pose is used only for logging and plots

import argparse
import time

import matplotlib
import mujoco
import numpy as np

from config import BEDSIDE_PARK_POSE, DT, MARKER_XY, PHARMACY_POSE, WARD_BOXES, wrap_angle
from driver import MujocoDriver
from lqr_tracker import LQRTracker, controllability_rank
from odometry import WheelOdometry
from planner import plan_to_bedside, build_ward_grid
from trajectory import build_ghost_robot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="watch in the MuJoCo viewer")
    args = parser.parse_args()
    if not args.view:
        matplotlib.use("Agg") #save the plot without opening a window
    import matplotlib.pyplot as plt

    print("Controllability rank moving:", controllability_rank(0.15, 0.0),
          "| stopped:", controllability_rank(0.0, 0.0))
    #---- Plan once, before moving ----
    grid = build_ward_grid()
    path = plan_to_bedside(grid, PHARMACY_POSE[:2], BEDSIDE_PARK_POSE)
    ghost = build_ghost_robot(path)
    tracker = LQRTracker(ghost, DT)
    route_length = np.sum(np.hypot(*np.diff(path, axis=0).T))
    print(f"Planned route: {route_length:.2f} m, {ghost.duration:.1f} s")

    driver = MujocoDriver()
    estimator = WheelOdometry(PHARMACY_POSE)
    viewer = None
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(driver.model, driver.data)
        #hide the LiDAR rays to keep the view plain. To see them, tick
        #"Rangefinder" in the viewer's Rendering panel
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

    def _pace_to_real_time(wall_start):
        #slow the loop to real time for the viewer. False if it was closed
        if viewer is None:
            return True
        viewer.sync()
        time.sleep(max(0.0, DT - (time.time() - wall_start)))
        return viewer.is_running()
    #---- Phase 1: turn on the spot to face along the route ----
    #starting far off the route makes LQR spin the robot hard enough to skid
    #the wheels, which corrupts odometry. A gentle turn first avoids that
    while True:
        loop_start_time = time.time()
        pose_estimate = estimator.update(*driver.read_encoders())
        err = wrap_angle(ghost.theta[0] - pose_estimate[2])
        if abs(err) < np.radians(2):
            break
        driver.set_twist(0.0, np.clip(1.5 * err, -0.8, 0.8))
        driver.advance(DT)
        if not _pace_to_real_time(loop_start_time):
            return

    #---- Phase 2: follow the route with LQR ----
    run_log = {"t": [], "true": [], "est": []}
    t = 0.0
    while True:
        loop_start_time = time.time()
        pose_estimate = estimator.update(*driver.read_encoders())   #1-2. sensors -> estimate
        v, w, reason = tracker.step(pose_estimate, t)   #3.   estimate -> command
        driver.set_twist(v, w) #4.   command -> motors
        driver.advance(DT) #5.   20 ms pass

        run_log["t"].append(t)
        run_log["true"].append(driver.true_pose())   #logging only
        run_log["est"].append(pose_estimate)
        if reason is not None:
            print(f"Stopped ({reason}) after {t:.1f} s")
            break
        if not _pace_to_real_time(loop_start_time):
            return
        t += DT

    #---- Results ----
    driver.set_twist(0.0, 0.0)
    driver.advance(1.0)   #let it settle
    final = driver.true_pose()
    run_log = {k: np.array(val) for k, val in run_log.items()}
    d = final[:2] - BEDSIDE_PARK_POSE[:2]
    c, s = np.cos(BEDSIDE_PARK_POSE[2]), np.sin(BEDSIDE_PARK_POSE[2])
    print(f"Docking error:  {100 * np.hypot(*d):.1f} cm "
          f"(along {100 * (c * d[0] + s * d[1]):+.1f}, "
          f"sideways {100 * (-s * d[0] + c * d[1]):+.1f})")
    print(f"Heading error:  {np.degrees(wrap_angle(final[2] - BEDSIDE_PARK_POSE[2])):+.1f} deg")
    print(f"Odometry drift: {100 * np.hypot(*(run_log['est'][-1][:2] - final[:2])):.1f} cm")

    #---- Plot ----
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                  gridspec_kw={"width_ratios": [1.7, 1]})
    ax.imshow(grid.occ, origin="lower", extent=grid.extent, cmap="Greys", alpha=0.6)
    ax.plot(*path.T, "--", color="0.45", label="planned route")
    ax.plot(*run_log["est"][:, :2].T, color="#2a6fdb", label="estimate (odometry)")
    ax.plot(*run_log["true"][:, :2].T, color="#e8590c", label="true position")
    ax.plot(*PHARMACY_POSE[:2], "o", color="0.2", ms=8, label="pharmacy")
    ax.plot(*MARKER_XY, "s", color="k", ms=8, label="bedside marker")
    for name, x0, x1, y0, y1, _ in WARD_BOXES:
        if not name.startswith("wall"):
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, name, ha="center", va="center",
                    color="w", fontsize=9)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Delivery: docking error {100 * np.hypot(*d):.1f} cm")
    ax.legend(loc="lower right", fontsize=8)
    drift = np.hypot(*(run_log["est"][:, :2] - run_log["true"][:, :2]).T)
    ax2.plot(run_log["t"], 100 * drift, color="#2a6fdb")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("odometry error (cm)")
    ax2.set_title("Estimate drift: what the EKF will fix"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("delivery_result.png", dpi=130)
    print("Saved delivery_result.png")
    if viewer is not None:
        plt.show()
        viewer.close()


if __name__ == "__main__":
    main()