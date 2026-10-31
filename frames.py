"""
frames.py is the one shared file for frames, transforms and robot constants
Stage 0 agreement: every subsystem imports geometry from here. The sensor
mounts, marker and start point are NOT redefined. They are imported from
Build_scene.py, which is what actually writes them into the MuJoCo model, so
the numbers the code uses can never drift away from the numbers in the
simulation.

FRAMES (right-handed, z up)
  W   world  : origin at the south-west floor corner, x = east, y = north [m, rad]
  B   base   : body "base" in turtlebot3_waffle_pi.xml. Its origin IS the
               axle midpoint (wheels at (0, +/-0.144, 0.033)), x forward,
               y left. The agreed state is the planar pose of B in W: [x, y, theta].
  L   lidar  : site "lidar" at LIDAR_POS in B. Beam angle 0 = forward, CCW positive.
  C   camera : camera "cam" at CAMERA_POS in B with xyaxes="0 -1 0 0 0 1",
               so x_C = -y_B, y_C = +z_B and the view direction -z_C = +x_B.
  Ccv opencv : x right, y down, z forward. solvePnP returns poses in this frame.
               x_Ccv = -y_B, y_Ccv = -z_B, z_Ccv = +x_B.
  M   marker : ArUco centre, z out of the printed face (towards -y_W).

"""
import warnings

import numpy as np

from Build_scene import (FOLDER, SCENE_OUT, CAMERA_POS, LIDAR_POS, CAMERA_FOVY,
                         START_X, START_Y, START_THETA)

ROOT = FOLDER

# ----------------------------------------------------------------- robot ---
# Checked against turtlebot3_waffle_pi.xml: the wheel bodies sit at
# y = +/-0.144 m (b = 0.288 m) with the axle at z = 0.033 m (r = 0.033 m).
WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.288
MAX_WHEEL_SPEED = 7.88        # = actuator ctrlrange in the XML (0.26 m/s / 0.033 m)
ROBOT_RADIUS = 0.22           #ircumscribed footprint radius about B, used for map inflation
WHEEL_JOINT_R, WHEEL_JOINT_L = "wheel_right", "wheel_left"
WHEEL_ACT_R, WHEEL_ACT_L = "wheel_right", "wheel_left"
CAMERA_NAME = "cam"
LIDAR_SITE = "lidar"

START_POSE = (START_X, START_Y, START_THETA)     #heading 0 = east, matching the home

#tha sensors
LIDAR_OFFSET = np.array(LIDAR_POS, dtype=float)
CAMERA_OFFSET = np.array(CAMERA_POS, dtype=float)
IMG_W, IMG_H = 640, 480 # same as View_scene.py
ARUCO_DICT_NAME = "DICT_4X4_50" #same as Build_scene.make_marker_image()

# OpenCV camera frame is MuJoCo camera frame: x unchanged, y and z flipped.
R_C_CCV = np.diag([1.0, -1.0, -1.0])


def wrap(angle):
    """Wrap an angle (or an array of angles) to [-pi, pi). Same as ekf.wrap."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def unicycle_to_wheels(v, w, r=WHEEL_RADIUS, b=WHEEL_BASE):
    """(v, omega) -> (wR, wL). Inverts v = r(wR+wL)/2, omega = r(wR-wL)/b."""
    return (v + w * b / 2) / r, (v - w * b / 2) / r


def wheels_to_unicycle(wR, wL, r=WHEEL_RADIUS, b=WHEEL_BASE):
    return r * (wR + wL) / 2, r * (wR - wL) / b


def se2_compose(a, b):
    """a (+) b: pose b expressed in frame a, returned in a's parent frame (2D T_WA @ T_AB)."""
    x, y, th = a
    c, s = np.cos(th), np.sin(th)
    return np.array([x + c * b[0] - s * b[1], y + s * b[0] + c * b[1], wrap(th + b[2])])


def lidar_pose(base_pose):
    """World pose of the lidar frame L, given the world pose of B."""
    return se2_compose(base_pose, (LIDAR_OFFSET[0], LIDAR_OFFSET[1], 0.0))
def scan_to_world(angles, ranges, sensor_pose):
    ok = np.isfinite(ranges)
    a = angles[ok] + sensor_pose[2]
    return np.column_stack([sensor_pose[0] + ranges[ok] * np.cos(a),
                            sensor_pose[1] + ranges[ok] * np.sin(a)])




_CALIB_CACHE = {}


def get_camera_calibration(scene_path=SCENE_OUT):
    """
    Return the camera intrinsics K and extrinsics (R_BC, t_BC) such that p_B = R_BC p_C + t_BC.

    K uses the pinhole model from Assessment 2. fx = fy = (H/2)/tan(fovy/2),
    with fovy read from the model (48.8 degrees, the Pi Camera v2). There is
    no distortion in simulation. The extrinsics are read from the compiled
    model once. They are fixed mount constants (the same role as the URDF on
    the real robot), so they do not leak the true pose.
    For your camera they are R_BC = [[0,0,-1],[-1,0,0],[0,1,0]] and t_BC = CAMERA_POS.
    """
    import mujoco
    key = str(scene_path)
    if key in _CALIB_CACHE:
        return _CALIB_CACHE[key]
    model = mujoco.MjModel.from_xml_path(key)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam < 0:
        raise ValueError(f'No camera "{CAMERA_NAME}" - run Build_scene.py first')
    base = model.body_rootid[model.cam_bodyid[cam]]
    R_WB, p_WB = data.xmat[base].reshape(3, 3), data.xpos[base]
    R_WC, p_WC = data.cam_xmat[cam].reshape(3, 3), data.cam_xpos[cam]
    R_BC = R_WB.T @ R_WC
    t_BC = R_WB.T @ (p_WC - p_WB)
    f = 0.5 * IMG_H / np.tan(np.deg2rad(model.cam_fovy[cam]) / 2)   # fovy is always in degrees
    K = np.array([[f, 0, IMG_W / 2], [0, f, IMG_H / 2], [0, 0, 1.0]])
    _CALIB_CACHE[key] = (K, R_BC, t_BC)
    return _CALIB_CACHE[key]

def check_robot_geometry(model, data, base_body, j_right, j_left, tol=0.003):
    """
    Check Person 2's model assumptions against the compiled model:
    the reference point is the axle midpoint, and the wheel separation equals WHEEL_BASE.
    """
    R_WB, p_WB = data.xmat[base_body].reshape(3, 3), data.xpos[base_body]
    aR = R_WB.T @ (data.xanchor[j_right] - p_WB)
    aL = R_WB.T @ (data.xanchor[j_left] - p_WB)
    sep = float(np.linalg.norm((aL - aR)[:2]))
    mid = (aL + aR) / 2
    if abs(sep - WHEEL_BASE) > tol:
        warnings.warn(f"Model wheel separation {sep:.4f} m != WHEEL_BASE {WHEEL_BASE} m")
    if np.linalg.norm(mid[:2]) > tol:
        warnings.warn(f"Axle midpoint {mid[:2]} is not at the base origin")
    return {"wheel_separation": sep, "axle_midpoint_in_body": mid}
