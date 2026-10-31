
#the LQR tracker that keeps the robot on the ghost

#error model (same as stage 3):
#e1 = ghost ahead, e2 = ghost to the left, e3 = heading difference
#A = [[0, ghost_w, 0], [-ghost_w, 0, ghost_v], [0, 0, 0]],   B = [[1, 0], [0, 0], [0, 1]]
#the gains get recomputed every step for the ghosts current speed and turning rate,
#in discrete time because the controller acts every 20 ms. relaxed priorities across
#the ward, strict ones at the bedside

import numpy as np
from scipy.linalg import expm, solve_discrete_are

import frames as F
from Control.nav_settings import MAX_V, MAX_W


def error_model(ghost_v, ghost_w):
    A = np.array([[0, ghost_w, 0], [-ghost_w, 0, ghost_v], [0, 0, 0]], dtype=float)
    B = np.array([[1, 0], [0, 0], [0, 1]], dtype=float)
    return A, B


def controllability_rank(ghost_v, ghost_w):
    #3 = every error can be fixed. 2 when stopped: a stopped robot cant fix a sideways error
    A, B = error_model(ghost_v, ghost_w)
    return int(np.linalg.matrix_rank(np.hstack([B, A @ B, A @ A @ B])))


def lqr_gain(A, B, Q, R, dt):
    #discrete-time LQR: convert the model to one 20 ms step (zero-order hold), then solve
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    Md = expm(M * dt)
    Ad, Bd = Md[:n, :n], Md[:n, n:]
    P = solve_discrete_are(Ad, Bd, Q, R)
    return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)


def bryson(max_errors, max_inputs):
    #bryson's rule weight = 1 / (largest acceptable value)^2
    return np.diag(1 / np.square(max_errors)), np.diag(1 / np.square(max_inputs))


#across the ward: accept 10 cm and 0.3 rad (17 deg), corrections up to 0.1 m/s and 1 rad/s
Q_TRANSIT, R_TRANSIT = bryson([0.10, 0.10, 0.30], [0.10, 1.0])
#at the bedside only 2 cm and 0.1 rad (6 deg) with gentler corrections
Q_DOCK, R_DOCK = bryson([0.02, 0.02, 0.10], [0.05, 0.5])


def tracking_error(pose, ghost_pose):
    #where the ghost is, from the robot's point of view
    x, y, th = pose
    xg, yg, thg = ghost_pose
    dx, dy = xg - x, yg - y
    return np.array([np.cos(th) * dx + np.sin(th) * dy,
                     -np.sin(th) * dx + np.cos(th) * dy,
                     F.wrap(thg - th)])


class LQRTracker:
    def __init__(self, ghost, dt, bedside_switch_distance=0.5, v_floor=0.05, timeout=10.0,
                 Q_transit=Q_TRANSIT, R_transit=R_TRANSIT, Q_dock=Q_DOCK, R_dock=R_DOCK):
        self.ghost, self.dt = ghost, dt
        self.bedside_switch_distance = bedside_switch_distance   #switch to docking gains this close
        self.v_floor = v_floor                   #never linearise about v = 0 (rank 2!)
        self.timeout = timeout
        self.Q_transit, self.R_transit = Q_transit, R_transit
        self.Q_dock, self.R_dock = Q_dock, R_dock
        self.goal = np.array([ghost.x[-1], ghost.y[-1]])
        self.done = False

    def step(self, pose_estimate, t):
        """
        pose_estimate = the EKF's ESTIMATED pose, never the true one.
        returns (v, w, reason): reason is None while driving, or 'reached'/'timeout'
        """
        if self.done:
            return 0.0, 0.0, "done"

        xg, yg, thg, ghost_v, ghost_w = self.ghost.at(t)
        e = tracking_error(pose_estimate, (xg, yg, thg))

        #stop once the ghost has finished and we've reached (or passed) its end
        if t >= self.ghost.duration and (e[0] <= 0.01 or t >= self.ghost.duration + self.timeout):
            self.done = True
            return 0.0, 0.0, "reached" if e[0] <= 0.01 else "timeout"

        near_bedside = np.hypot(*(self.goal - pose_estimate[:2])) < self.bedside_switch_distance
        Q, R = (self.Q_dock, self.R_dock) if near_bedside else (self.Q_transit, self.R_transit)
        A, B = error_model(max(ghost_v, self.v_floor), ghost_w)
        u = -lqr_gain(A, B, Q, R, self.dt) @ e

        #feedforward (what the ghost does) plus feedback (the LQR correction)
        v = np.clip(ghost_v * np.cos(e[2]) - u[0], -0.05, MAX_V)
        w = np.clip(ghost_w - u[1], -MAX_W, MAX_W)
        return float(v), float(w), None
