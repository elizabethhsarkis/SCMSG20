import numpy as np
import matplotlib.pyplot as plt
# scipy solves the LQR maths for us.
from scipy.linalg import solve_continuous_are

maximumVelocity = 0.22 #robot speed limit in m/s
maximumAngularVelocity = 2.84 #robot turning speed limit in rad/s


def step(pose, v, w, dt):
    """move one time step."""
    x, y, theta = pose
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += w * dt
    return np.array([x, y, theta])

def wrapAngle(a):
    """Keep an angle between -pi and +pi."""
    return (a + np.pi) % (2 * np.pi) - np.pi

def trackingErr(pose, ghost):
    x, y, theta = pose
    xr, yr, thetar = ghost
    #put into world directions
    dx = xr - x
    dy = yr-y

    #rotate into robot directions
    #e1 is how much the gap looks like in the robot's forward direction
    e1 = np.cos(theta)*dx + np.sin(theta)*dy
    #e2 is how much the gap looks like in the robot's left direction
    e2 = -np.sin(theta)*dx + np.cos(theta)*dy
    #e3 is heading difference, wrapped
    e3 = wrapAngle(thetar - theta)

    return np.array([e1, e2, e3])

vr = 0.1 #desired forward speed in m/s
wr = 0.2 #desired turning speed in rad/s
#row 1, e1 changes when path curves and when we speed up
#row 2 e2 changes when path surves and when heading is off
#e3 changes only when we turn
A = np.array([[0, wr, 0], [-wr, 0, vr], [0, 0, 0]])

B = np.array([[1, 0], [0, 0], [0, 1]])

C = np.hstack([B, A @ B, A @ A @ B])
print("Controllability rank:", np.linalg.matrix_rank(C), "of 3")

# Bryson's rule: weight = 1 / (largest acceptable value)².
# Q: we accept up to 10 cm of e1 and e2, and 0.3 rad (17°) of e3.
Q = np.diag([1 / 0.1**2, 1 / 0.1**2, 1 / 0.3**2])
# R: we're willing to use up to 0.1 m/s and 1 rad/s of correction.
R = np.diag([1 / 0.1**2, 1 / 1.0**2])

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
print("LQR gains K:\n", np.round(K, 2))

dt = 0.02
ghost = np.array([0.0, 0.0, 0.0])     # ghost starts on the path
pose = np.array([0.0, -0.3, 0.5])     # robot starts 30 cm off, turned 0.5 rad
ghost_path, robot_path, errors = [ghost], [pose], []

for i in range(int(40 / dt)):
    # Measure the errors to the ghost.
    e = trackingErr(pose, ghost)

    # LQR correction: two numbers, one per control.
    u = -K @ e

    # Final commands = what the ghost is doing, plus the correction.
    v = vr * np.cos(e[2]) - u[0]
    w = wr - u[1]
    v = np.clip(v, -maximumVelocity, maximumVelocity)
    w = np.clip(w, -maximumAngularVelocity, maximumAngularVelocity)

    # Move the robot, and move the ghost along its path.
    pose = step(pose, v, w, dt)
    ghost = step(ghost, vr, wr, dt)

    robot_path.append(pose)
    ghost_path.append(ghost)
    errors.append(e)

robot_path = np.array(robot_path)
ghost_path = np.array(ghost_path)
errors = np.array(errors)
time = np.arange(len(errors)) * dt

# ---- 6. Plots ----
# plt.subplots(1, 2) makes two plots side by side.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(ghost_path[:, 0], ghost_path[:, 1], "--", label="ghost (reference)")
ax1.plot(robot_path[:, 0], robot_path[:, 1], label="robot")
ax1.axis("equal")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.legend()

ax2.plot(time, errors[:, 0], label="e1 ahead (m)")
ax2.plot(time, errors[:, 1], label="e2 left (m)")
ax2.plot(time, errors[:, 2], label="e3 heading (rad)")
ax2.set_xlabel("time (s)")
ax2.legend()

plt.tight_layout()
plt.show()
