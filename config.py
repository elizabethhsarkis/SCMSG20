#settings shared by every file: robot sizes, the ward layout, key poses
#
#to change the ward, edit WARD_BOXES. The MuJoCo scene and the planner's
#map are both built from it, so they always match

import numpy as np

#---- TurtleBot3 Burger ----
WHEEL_RADIUS = 0.033  #m
WHEEL_SEPARATION = 0.160 #m, distance between the two wheels
MAX_V = 0.22   #m/s, top forward speed
MAX_W = 2.84 #rad/s, top turning rate
ROBOT_RADIUS = 0.105 #m, a circle that covers the robot's footprint
SAFETY_MARGIN = 0.10 #m, extra clearance the planner keeps from obstacles

#---- LiDAR ----
LIDAR_RAYS = 72  #one ray every 5 degrees (the real one has 360)
LIDAR_HEIGHT = 0.17 #m above the floor
LIDAR_RANGE = 3.5 #m, anything further counts as "nothing there"

#---- The ward ----
#a large, plain room with a single bed against the far wall
#each box: (name, x_min, x_max, y_min, y_max, height), in metres
WARD_WIDTH, WARD_HEIGHT = 10.0, 6.0
WALL = 0.1
WARD_BOXES = [
    ("wall_south", 0.0, WARD_WIDTH, 0.0, WALL, 0.5),
    ("wall_north", 0.0, WARD_WIDTH, WARD_HEIGHT - WALL, WARD_HEIGHT, 0.5),
    ("wall_west",  0.0, WALL, 0.0, WARD_HEIGHT, 0.5),
    ("wall_east",  WARD_WIDTH - WALL, WARD_WIDTH, 0.0, WARD_HEIGHT, 0.5),
    ("bed",        6.5, 7.5, 3.9, WARD_HEIGHT - WALL, 0.6),
]

#---- Key poses: (x, y, heading in radians) ----
PHARMACY_POSE = np.array([1.0, 1.0, 0.0]) #pharmacy, facing +x
MARKER_XY = np.array([7.0, 3.9]) #arUco marker on the foot of the bed
BEDSIDE_PARK_POSE = np.array([7.0, 3.6, np.pi / 2])   #30 cm in front of it, facing it

#---- Control loop ----
DT = 0.02   #s, the loop runs 50 times per second


def wrap_angle(a):
    #keep an angle between -pi and +pi (350 deg becomes -10 deg)
    return (a + np.pi) % (2 * np.pi) - np.pi