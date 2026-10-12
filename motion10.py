import numpy as np
import matplotlib.pyplot as plt

#robot dimensions
wheelRadius = 0.033
wheelSeparation = 0.160

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




def wheelSpinToMotion(leftWheel, rightWheel):
    #ground speed of each wheel
    vLeft = leftWheel * wheelRadius
    vRight = rightWheel * wheelRadius
    v = (vLeft + vRight) / 2
    w = (vRight - vLeft) / wheelSeparation

    return v, w

def twistToWheelSpin(v, w):
    vLeft = v - (w * wheelSeparation / 2)
    vRight = v + (w * wheelSeparation / 2)

    leftWheel = vLeft / wheelRadius
    rightWheel = vRight / wheelRadius

    return leftWheel, rightWheel

#now for running the sim
#go for 20s, int(20/dt) = 1000 steps
#i is the index of the current step
for i in range(int(20/dt)):
    #input wheel speeds
    v, w = wheelSpinToMotion(5, 6) #set a goal
    pose = step(pose, v, w, dt)
    path.append(pose)

#urn list into numpy array for easier slicing
path = np.array(path)

print(wheelSpinToMotion(5, 5))
print(wheelSpinToMotion(-5, 5))
print(wheelSpinToMotion(6, 4))

#plot the path
#ploy y vs x
plt.plot(path[:, 0], path[:, 1])

plt.axis("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Robot Path")
plt.show()
