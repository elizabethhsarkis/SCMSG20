#turns the planned path into a timed reference: the "ghost robot"
#
#in Stage 3 the ghost drove a circle. Here it drives the planned path, at a
#speed that respects the robot's limits and slows down at the bedside

import numpy as np


class GhostRobot:
    def __init__(self, t, x, y, theta, v, omega):
        self.t, self.x, self.y, self.theta, self.v, self.omega = t, x, y, theta, v, omega
        self.duration = float(t[-1])
    def at(self, t):
        #where the ghost is and what it's doing at time t:
        #(x, y, theta, v, omega). After the end it waits at the last point
        if t >= self.duration:
            return self.x[-1], self.y[-1], self.theta[-1], 0.0, 0.0

        def f(arr):
            return float(np.interp(t, self.t, arr))

        return f(self.x), f(self.y), f(self.theta), f(self.v), f(self.omega)


def build_ghost_robot(path, v_max=0.18, a_max=0.25, omega_max=1.2,
                    v_dock=0.05, dock_length=0.5, ds=0.01):
    #v_max: top speed (m/s). a_max: acceleration limit (m/s²)
    #omega_max: turning-rate limit on curves (rad/s)
    #v_dock: speed over the last dock_length metres
    path = np.asarray(path, dtype=float)
    #resample the path every 1 cm along its length (s = distance along it)
    s_raw = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    s = np.append(np.arange(0, s_raw[-1], ds), s_raw[-1])
    x = np.interp(s, s_raw, path[:, 0])
    y = np.interp(s, s_raw, path[:, 1])

    #heading = direction of the path. Curvature = how fast heading changes
    theta = np.unwrap(np.arctan2(np.gradient(y, s), np.gradient(x, s)))
    curvature = np.convolve(np.gradient(theta, s), np.ones(7) / 7, mode="same")
    #speed limits: top speed, slower on tight curves, crawl at the bedside
    v = np.minimum(v_max, omega_max / np.maximum(np.abs(curvature), 1e-6))
    near_end = s >= s[-1] - dock_length
    v[near_end] = np.minimum(v[near_end], v_dock)

    #acceleration limit, starting and ending at rest (v² = u² + 2as)
    #the forward pass limits speeding up, the backward pass slowing down
    v[0] = 0.0
    for i in range(1, len(s)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a_max * (s[i] - s[i - 1])))
    v[-1] = 0.0
    for i in range(len(s) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_max * (s[i + 1] - s[i])))


    #time stamps: distance / average speed for each 1 cm step
    t = np.concatenate([[0], np.cumsum(2 * np.diff(s) / np.maximum(v[:-1] + v[1:], 1e-6))])
    return GhostRobot(t, x, y, theta, v, v * curvature)