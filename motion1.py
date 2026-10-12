import numpy as np
import matplotlib.pyplot as plt

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

#starting pose
pose = np.array([0.0, 0.0, 0.0])
#time step update
dt = 0.02

#list records each pose
path = [pose]

#now for running the sim
#go for 20s, int(20/dt) = 1000 steps
#i is the index of the current step
for i in range(int(20/dt)):
    #move robot one step, 0.1 m/s forward, turning 0.2 rad/s left.
    pose = step(pose, 0.1, 0.2, dt)
    #add to end of list
    path.append(pose)

#urn list into numpy array for easier slicing
path = np.array(path)

#plot the path
#ploy y vs x
plt.plot(path[:, 0], path[:, 1])

plt.axis("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Robot Path")
plt.show()
