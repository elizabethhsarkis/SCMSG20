
# can the EKF properly work out the pose
#state x = [x, y, theta] (pose of the axle midpoint in the world). the EKF is linearised
#every step, so observability is checked on the linearised model along the real route:
#   x_{k+1} = F_k x_k + ...         F_k = d(motion model)/d(state)   (from ekf.predict)
#   z_k     = H_k x_k + ...         H_k = d(range, bearing)/d(state) (when the marker is seen)
#
#1. one reading:   rank H = 2 < 3. one direction (the null space of H) is invisible
#2. no marker:     H = nothing. odometry alone cant observe the absolute pose at all
#3. along the route: windowed observability gramian
#       W_k = sum_j Phi(j,k)^T H_j^T R_j^-1 H_j Phi(j,k)   over the last WINDOW_S seconds
#   rank W = 3 means the pose is observable over that window; its smallest eigenvalue
#   says how WELL, and its eigenvector says which combination of x, y, theta is weakest
#4. a map of the room: how observable the pose is from each spot, driving towards the marker
#
#run from the repo folder python Analysis/observability.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

#this script belongs to the classic ward. it stops (instead of quietly overwriting the
#classic results) if the paediatric ward is selected
from Build_scene import WARD as _WARD
if _WARD != "classic" and __name__ == "__main__":
    raise SystemExit("observability.py studies the classic ward's single marker: run it with --ward classic")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import frames as F
from Build_scene import OBSTACLES
from Control.nav_settings import PHARMACY_POSE, BEDSIDE_PARK_POSE, MARKER_XY, DT, CAMERA_EVERY, marker_noise
from Control.planner import build_known_grid, plan_to_bedside
from Control.trajectory import build_ghost_robot
from Control.lqr_tracker import controllability_rank

OUT = Path(__file__).resolve().parent
WINDOW_S = 2.0
CAMERA_HALF_FOV = np.arctan(np.tan(np.radians(F.CAMERA_FOVY) / 2) * F.IMG_W / F.IMG_H)   #31 deg
MAX_DETECT_RANGE = 2.0          #from marker_noise.py gives oblique views past ~2 m are blocked by the beds


def motion_jacobian(theta, ds, dth):
    #F from Person 2's ekf.predict da midpoint heading
    phi = theta + dth / 2
    return np.array([[1, 0, -ds * np.sin(phi)], [0, 1, ds * np.cos(phi)], [0, 0, 1]])


def measurement_jacobian(pose):
    #H from ekf.update_range_bearing
    dx, dy = MARKER_XY - pose[:2]
    q = dx ** 2 + dy ** 2
    r = np.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0], [dy / q, -dx / q, -1]])


def marker_visible(pose):
    dx, dy = MARKER_XY - pose[:2]
    r = np.hypot(dx, dy)
    return dy > 0 and r < MAX_DETECT_RANGE and abs(F.wrap(np.arctan2(dy, dx) - pose[2])) < CAMERA_HALF_FOV


def route_poses():
    #the nominal route, sampled every control step (the ghost robot = where the robot should be)
    path = plan_to_bedside(build_known_grid(), PHARMACY_POSE, BEDSIDE_PARK_POSE)
    ghost = build_ghost_robot(path)
    t = np.arange(0, ghost.duration, DT)
    poses = np.array([ghost.at(tt)[:3] for tt in t])
    return t, poses


def windowed_gramian(poses, k, window_steps, camera_every):
    #information about the pose at step k from the marker readings in the last window
    #Phi maps a small error at step k back to step j: dx_j = Phi dx_k.
    #going one step back: dx_{j-1} = F_{j-1}^-1 dx_j, so Phi_{j-1} = F_{j-1}^-1 Phi_j
    W = np.zeros((3, 3))
    Phi = np.eye(3)
    for j in range(k, max(k - window_steps, 0) - 1, -1):
        if j % camera_every == 0 and marker_visible(poses[j]):
            H = measurement_jacobian(poses[j])
            r = np.hypot(*(MARKER_XY - poses[j][:2]))
            Rinv = np.linalg.inv(marker_noise(r, "fitted"))
            W += Phi.T @ H.T @ Rinv @ H @ Phi
        if j > 0:
            ds = np.hypot(*(poses[j][:2] - poses[j - 1][:2]))
            dth = F.wrap(poses[j][2] - poses[j - 1][2])
            F_prev = motion_jacobian(poses[j - 1][2], ds, dth)
            Phi = np.linalg.solve(F_prev, Phi)
    return W


def weak_direction(W):
    vals, vecs = np.linalg.eigh(W)
    return vals, vecs[:, 0]


def rotation_about_marker(pose):
    #the direction that changes nothing the robot can sense: rotate the pose about the marker
    #by a small angle a -> position moves a * (-(y - ym), x - xm), heading changes by a
    v = np.array([-(pose[1] - MARKER_XY[1]), pose[0] - MARKER_XY[0], 1.0])
    return v / np.linalg.norm(v)


def rotate_route(poses, angle):
    c, s_ = np.cos(angle), np.sin(angle)
    rel = poses[:, :2] - MARKER_XY
    out = poses.copy()
    out[:, :2] = MARKER_XY + rel @ np.array([[c, s_], [-s_, c]])
    out[:, 2] = poses[:, 2] + angle
    return out


def sensed(poses):
    #everything the robot measures: wheel increments (ds, dth) and range/bearing to the marker
    ds = np.hypot(*np.diff(poses[:, :2], axis=0).T)
    dth = F.wrap(np.diff(poses[:, 2]))
    rng = np.hypot(*(MARKER_XY - poses[:, :2]).T)
    brg = F.wrap(np.arctan2(*(MARKER_XY - poses[:, :2]).T[::-1]) - poses[:, 2])
    return ds, dth, rng, brg


if __name__ == "__main__":
    # 1 reading
    H = measurement_jacobian(BEDSIDE_PARK_POSE)
    _, _, Vt = np.linalg.svd(H)
    null = Vt[-1] * np.sign(Vt[-1][2])
    print("1. ONE MARKER READING at the dock (0.40 m away, facing it)")
    print(f"   rank H = {np.linalg.matrix_rank(H)} of 3")
    print(f"   invisible direction (null space of H): dx {null[0]:+.3f}, dy {null[1]:+.3f}, dtheta {null[2]:+.3f}")

    #no marker
    print("\n2. NO MARKER (first part of the route): nothing measures the absolute pose, rank 0.")
    print("   odometry only integrates, so the EKF's uncertainty grows until the marker is seen")

    # along the route
    t, poses = route_poses()
    window = int(WINDOW_S / DT)
    visible = np.array([marker_visible(p) for p in poses])
    ranks, eig, weak, rot = [], [], [], []
    for k in range(len(poses)):
        W = windowed_gramian(poses, k, window, CAMERA_EVERY)
        vals, vecs = np.linalg.eigh(W)
        top = max(vals[-1], 1e-300)
        ranks.append(int(np.sum(vals > 1e-6 * top)) if vals[-1] > 0 else 0)
        eig.append(vals / top if vals[-1] > 0 else np.full(3, np.nan))
        v0 = vecs[:, 0] * np.sign(vecs[2, 0] if abs(vecs[2, 0]) > 1e-12 else 1)
        weak.append(v0)
        rot.append(rotation_about_marker(poses[k]))
    ranks, eig, weak, rot = map(np.array, (ranks, eig, weak, rot))
    seen = ranks > 0
    match = np.abs(np.sum(weak[seen] * rot[seen], axis=1))
    print(f"\n3. ALONG THE ROUTE ({t[-1]:.1f} s, {WINDOW_S:.0f} s window of readings)")
    print(f"   marker first in view at t = {t[visible][0]:.1f} s; rank 0 before that")
    print(f"   rank once the marker is in view: {sorted(set(ranks[seen].tolist()))} (never 3)")
    print(f"   smallest eigenvalue/ largest: at most {np.nanmax(eig[seen, 0]):.1e} (numerically zero)")
    print(f"   the weak direction matches rotation about the marker: |cos angle| >= {match.min():.4f}")
    print("   => one range-bearing marker leaves ONE direction unobservable on any route: rotating")
    print("      the robot (position and heading together) about the marker changes no reading.")
    print("      only the known start pose, carried by odometry, pins it down - and that drifts.")

    #demo: two routes that give identical readings
    angle = np.radians(6)
    rotated = rotate_route(poses, angle)
    a_ds, a_dth, a_r, a_b = sensed(poses)
    b_ds, b_dth, b_r, b_b = sensed(rotated)
    print(f"\n   demo: route rotated {np.degrees(angle):.0f} deg about the marker, end point "
          f"{100 * np.hypot(*(rotated[-1, :2] - poses[-1, :2])):.1f} cm away from the real one")
    print(f"   biggest difference in any reading: wheels {max(abs(a_ds - b_ds).max(), abs(F.wrap(a_dth - b_dth)).max()):.1e}, "
          f"range {abs(a_r - b_r).max():.1e} m, bearing {abs(F.wrap(a_b - b_b)).max():.1e} rad")

    # ontrollability
    print("\nCONTROLLABILITY of the tracking error model (Part A, other half)")
    for v, w in [(0.15, 0.0), (0.15, 0.5), (0.0, 0.5), (0.0, 0.0)]:
        print(f"   ghost v {v:4.2f} m/s, w {w:3.1f} rad/s -> rank {controllability_rank(v, w)} of 3")

    #figure
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.1])
    a0 = fig.add_subplot(gs[0, 0])
    a0.step(t, ranks, where="post", color="#2a6fdb")
    a0.fill_between(t, 0, 3.3, where=visible, color="#2b8a3e", alpha=0.12, label="marker in view")
    a0.axhline(3, color="#c92a2a", ls=":", lw=1, label="full rank (3)")
    a0.set_ylabel("rank of W"); a0.set_ylim(-0.2, 3.3); a0.set_yticks([0, 1, 2, 3]); a0.legend(fontsize=8)
    a0.set_title(f"Observability along the nominal route ({WINDOW_S:.0f} s of readings)", fontsize=10)
    a1 = fig.add_subplot(gs[1, 0], sharex=a0)
    for i, (lab, col) in enumerate((("smallest", "#c92a2a"), ("middle", "#e8590c"), ("largest", "#2a6fdb"))):
        a1.semilogy(t, np.clip(eig[:, i], 1e-16, None), color=col, label=lab)
    a1.set_ylabel("eigenvalues of W\n(/ largest)"); a1.legend(fontsize=8, ncol=3); a1.grid(alpha=0.3, which="both")
    a2 = fig.add_subplot(gs[2, 0], sharex=a0)
    for i, (lab, col) in enumerate((("x", "#2a6fdb"), ("y", "#e8590c"), ("θ", "#2b8a3e"))):
        a2.plot(t, np.where(seen, weak[:, i], np.nan), color=col, lw=2.5, alpha=0.5, label=f"{lab}, from W")
        a2.plot(t, np.where(seen, rot[:, i], np.nan), color=col, ls="--", lw=1, label=f"{lab}, rotation about marker")
    a2.set_ylabel("unobservable direction"); a2.set_xlabel("time along route (s)")
    a2.legend(fontsize=7, ncol=2); a2.grid(alpha=0.3)

    a3 = fig.add_subplot(gs[0:2, 1])
    for name, x0, y0, x1, y1, *_ in OBSTACLES:
        a3.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="0.85"))
        a3.text((x0 + x1) / 2, (y0 + y1) / 2, name, ha="center", va="center", fontsize=8)
    a3.plot(poses[:, 0], poses[:, 1], color="#2a6fdb", lw=2, label="real route")
    a3.plot(rotated[:, 0], rotated[:, 1], color="#e8590c", lw=2, ls="--",
            label=f"same route rotated {np.degrees(angle):.0f}° about the marker")
    for p, col in ((poses[-1], "#2a6fdb"), (rotated[-1], "#e8590c")):
        a3.arrow(p[0], p[1], 0.25 * np.cos(p[2]), 0.25 * np.sin(p[2]), color=col, width=0.012)
    a3.plot(*MARKER_XY, "ks", ms=8, label="marker")
    a3.set_aspect("equal"); a3.set_xlim(0, 6); a3.set_ylim(0, 4)
    a3.set_title("Two different routes, identical wheel and marker readings", fontsize=10)
    a3.legend(fontsize=8, loc="lower right")
    a4 = fig.add_subplot(gs[2, 1])
    a4.plot(t[visible], a_r[visible], color="#2a6fdb", lw=3, alpha=0.5, label="range, real")
    a4.plot(t[visible], b_r[visible], color="#e8590c", ls="--", label="range, rotated")
    a4b = a4.twinx()
    a4b.plot(t[visible], np.degrees(a_b[visible]), color="#2b8a3e", lw=3, alpha=0.5, label="bearing, real")
    a4b.plot(t[visible], np.degrees(b_b[visible]), color="k", ls="--", lw=1, label="bearing, rotated")
    a4.set_xlabel("time (s), marker in view"); a4.set_ylabel("range (m)"); a4b.set_ylabel("bearing (deg)")
    a4.legend(fontsize=7, loc="upper center"); a4b.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "observability.png", dpi=130)
    print("\nsaved Analysis/observability.png")
