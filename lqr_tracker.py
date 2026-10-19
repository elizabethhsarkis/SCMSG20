#LQR tracker: your Stage 3 controller, grown up
#
#same error model as Stage 3:
#e1 = ghost ahead, e2 = ghost to the left, e3 = heading difference
#A = [[0, ghost_w, 0], [-ghost_w, 0, ghost_v], [0, 0, 0]],   B = [[1, 0], [0, 0], [0, 1]]
#
#differences from Stage 3:
#- the ghost's speed changes along the route, so the gains are recomputed
#every step for the current speed and turning rate;
#- the gains are computed in discrete time, because the controller acts
#every 20 ms rather than continuously;
#- there are two sets of priorities: relaxed across the ward, strict at
#the bedside

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from config import MAX_V, MAX_W, wrap_angle


def error_model(ghost_v, ghost_w):
    A = np.array([[0, ghost_w, 0], [-ghost_w, 0, ghost_v], [0, 0, 0]], dtype=float)
    B = np.array([[1, 0], [0, 0], [0, 1]], dtype=float)
    return A, B


def controllability_rank(ghost_v, ghost_w):
    #3 = all errors can be corrected. 2 when stopped: a stopped robot
    #can't fix a sideways error
    A, B = error_model(ghost_v, ghost_w)
    return int(np.linalg.matrix_rank(np.hstack([B, A @ B, A @ A @ B])))


def lqr_gain(A, B, Q, R, dt):
    #discrete-time LQR: convert the model to 20 ms steps, then solve
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    Md = expm(M * dt)   #exact conversion to one 20 ms step
    Ad, Bd = Md[:n, :n], Md[:n, n:]
    P = solve_discrete_are(Ad, Bd, Q, R)
    return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)


def bryson(max_errors, max_inputs):
    #bryson's rule: weight = 1 / (largest acceptable value)²
    return np.diag(1 / np.square(max_errors)), np.diag(1 / np.square(max_inputs))


#across the ward: accept 10 cm and 0.3 rad (17°); corrections up to 0.1 m/s, 1 rad/s
Q_TRANSIT, R_TRANSIT = bryson([0.10, 0.10, 0.30], [0.10, 1.0])
#at the bedside: accept only 2 cm and 0.1 rad (6°), with gentler corrections
Q_DOCK, R_DOCK = bryson([0.02, 0.02, 0.10], [0.05, 0.5])


def tracking_error(pose, ref_pose):
    #same as Stage 3: where the ghost is, from the robot's point of view
    x, y, th = pose
    xr, yr, thr = ref_pose
    dx, dy = xr - x, yr - y
    return np.array([np.cos(th) * dx + np.sin(th) * dy,
                     -np.sin(th) * dx + np.cos(th) * dy,
                     wrap_angle(thr - th)])


class LQRTracker:
    def __init__(self, ghost, dt, bedside_switch_distance=0.5, v_floor=0.05, timeout=10.0):
        self.ghost, self.dt = ghost, dt
        self.bedside_switch_distance = bedside_switch_distance  #switch to docking gains this close
        self.v_floor = v_floor  #never linearise about v = 0 (rank 2!)
        self.timeout = timeout
        self.goal = np.array([ghost.x[-1], ghost.y[-1]])
        self.done = False
    def step(self, pose_estimate, t):
        #pose_estimate = the ESTIMATED pose. Returns (v, w, reason): reason is None
        #while driving, or 'reached' / 'timeout' once stopped
        if self.done:
            return 0.0, 0.0, "done"


        xr, yr, thr, ghost_v, ghost_w = self.ghost.at(t)
        e = tracking_error(pose_estimate, (xr, yr, thr))

        #stop once the ghost has finished and we've reached (or passed) its end
        if t >= self.ghost.duration and (e[0] <= 0.01 or t >= self.ghost.duration + self.timeout):
            self.done = True
            return 0.0, 0.0, "reached" if e[0] <= 0.01 else "timeout"
        near_bedside = np.hypot(*(self.goal - pose_estimate[:2])) < self.bedside_switch_distance
        Q, R = (Q_DOCK, R_DOCK) if near_bedside else (Q_TRANSIT, R_TRANSIT)
        A, B = error_model(max(ghost_v, self.v_floor), ghost_w)
        u = -lqr_gain(A, B, Q, R, self.dt) @ e

        #same as Stage 3: do what the ghost does, plus the correction
        v = np.clip(ghost_v * np.cos(e[2]) - u[0], -0.05, MAX_V)
        w = np.clip(ghost_w - u[1], -MAX_W, MAX_W)
        return float(v), float(w), None