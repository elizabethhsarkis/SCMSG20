import numpy as np
import matplotlib.pyplot as plt

#robot dimensions
wheelRadius = 0.033
wheelSeparation = 0.160

maximumVelocity = 0.22
maximumAngularVelocity = 2.84

def wrapAngle(k):
    return (k +np.pi) % (2*np.pi) - np.pi

def goalSetting(pose, goal, kv = 0.5, kw = 2.0):
    x, y, theta = pose
    gx, gy = goal
    #how far goal is along x and u
    dx = gx-x
    dy = gy-y

    #straight line distance to goal
    distance = np.sqrt(dx**2 + dy**2)
    #direction to goal as an angle from the x axis
    angleToGoal = np.arctan2(dy, dx)
    #heading error = direction to goal - where robot currently faces
    headingError = wrapAngle(angleToGoal - theta)

    v = kv * distance * max(np.cos(headingError), 0) #slows down when facing away from the goal
    w = kw * headingError

    # The real robot has speed limits, so cap the commands.
    # np.clip(value, low, high) keeps value between low and high.
    v = np.clip(v, 0, maximumVelocity)
    w = np.clip(w, -maximumAngularVelocity, maximumAngularVelocity)

    return v, w, distance

#the below function moes the robot a little at a time
#pose = x, y, theta
#x, y is position in m and theta is heading in rad
#v = forward speed in m/s
#w = turnikng rate in m/s
#dt is how long time frame lasts in s


def step(pose, v, w, dt):
    x, y, theta = pose
    x += v* np.cos(theta)*dt
    y += v* np.sin(theta)*dt
    theta += w*dt
    return np.array([x, y, theta])

pose = np.array([0.0, 0.0, 0.0])   # start at the origin, facing +x
current = 0
#goal = np.array([-1.0, 0.0])        #where we want to go
goals = np.array([
    [1.5, 0.5],    # goal 0
    [1.5, 1.5],    # goal 1
    [0.0, 1.5],    # goal 2
])
dt = 0.02
path = [pose]

for i in range(int(60 / dt)):      # at most 30 seconds
    v, w, distance = goalSetting(pose, goals[current])

    #stop when within 2 cm of the goal.
    #break jumps out of the loop straight away.
    if distance < 0.02:
        print(f"Reached the goal {current} after {i * dt:.1f} seconds")
        current += 1
        if current == len(goals):
            print("All goals visited")
            break
        continue

    pose = step(pose, v, w, dt)
    path.append(pose)

path = np.array(path)

plt.plot(path[:, 0], path[:, 1], label="robot path")
plt.plot(goals[:, 0], goals[:, 1], "rx", markersize=12, label="goals")
plt.axis("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.show()
