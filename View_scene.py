import time
import numpy as np
import cv2
import mujoco
import mujoco.viewer

from Build_scene import (SCENE_OUT, FOLDER, 
                         MARKER_ID, MARKER_SIDE, MARKER_X, MARKER_Y, MARKER_Z, 
                         DOCK_POSITION, CAMERA_POS, CAMERA_FOVY
                         )

IMG_W, IMG_H = 640, 480                 # camera resolution (same as Pi camera video mode)



# Camera model K from field of view

#Pinhold model: pixel = K * point_in_camera_frame       K = focal length in pixels (f / rho) and principle point (u0, v0)
# **Simulated camera has no lens distortion and principle point = image center 
# K follows vertical FOV: f = (H/2) / tan(fovy/2)
# real robot = checkerboard calibration

f = (IMG_H/2) / np.tan(np.radians(CAMERA_FOVY)/2)
K = np.array([[f, 0, IMG_W/2],
              [0, f, IMG_H/2], 
              [0, 0, 1]])
DIST = np.zeros(5)                  # no lens distortion in simulation



# Find marker and convert to range & bearing
def marker_range_bearing(rgb_image):                # return range & bearning from robot centre to marker or None if no marker
    grey = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)          #detector works on greyscale
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary)
    corners, ids, _ = detector.detectMarkers(grey)              # 4 corner pixels per marker found

    if ids is None or MARKER_ID not in ids.ravel(): 
        return None                                         # marker not visible


    s = MARKER_SIDE/2
    corners_3d = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
    corners_px = corners[list(ids.ravel()).index(MARKER_ID)].reshape(4,2)
    ok, rvec, tvec = cv2.solvePnP(corners_3d, corners_px, K, DIST, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    xc, yc, zc = tvec.ravel()                   #marker centre in camera frame


    # Camera frame to Robot body frame          p_W = T_WB * T_BS * p_S             ##*#*#*#*#
    x_b = zc + CAMERA_POS[0]
    y_b = -xc + CAMERA_POS[1]


    #range and bearing
    # rho = sqrt(x^2 + y^2), alpha = atan2(y, x)

    return np.hypot(x_b, y_b), np.arctan2(y_b, x_b)


# Main code
if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(SCENE_OUT))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    check = mujoco.MjData(model)
    dock_x, dock_y = MARKER_X, MARKER_Y - DOCK_POSITION         # 0.4 m infront of marker

    check.qpos[0:7] = [dock_x, dock_y, 0, np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]       # Robot pose = position then 90 degree turn about z
    mujoco.mj_forward(model, check)                 # update positions without stepping time

    renderer = mujoco.Renderer(model, IMG_H, IMG_W)
    renderer.update_scene(check, camera="cam")                  # look through camera added in Build_scene
    image = renderer.render()
    cv2.imwrite(str(FOLDER / "cam_check.png"), image[:, :, ::-1])           

    result = marker_range_bearing(image)

    expected_range = np.hypot(MARKER_X - dock_x, MARKER_Y - dock_y)
    if result is None:
        print("Camera check, marker NOT detected - open cam_check.png")
    else:
        rng, brg = result
        print(f"Camera check: range {rng:.3f} m (expected {expected_range:.3f}), "
              f"bearing {np.degrees(brg):.2f} deg (expected 0.00)")

    ## drive robot in viewer - wheels use velocity actuators so speed only set once

    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_left")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_right")

    #differential drive: v = r (wR + wL)/2 = 0.033 * 3.0 = ~0.1 m/s
    #w = r (wR - wL)/L = 0.033 * 0.4 / 0.288 = ~0.046 rad/s (slight left curve)

    data.ctrl[left] = 2.8
    data.ctrl[right] = 3.2

    # with mujoco.viewer.launch_passive(model, data) as viewer:
    #     while viewer.is_running():
    #         step_start = time.time()
    #         mujoco.mj_step(model, data)
    #         viewer.sync()

    #         wait = model.opt.timestep - (time.time() - step_start)
    #         if wait > 0:
    #             time.sleep(wait)
    SPEED = 4.0 #1.0 = real time, 4.0 = four times faster; raise it as high as your laptop allows
    FPS = 60 #screen redraws per second the physics still steps every 2 ms

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                frame_start = time.time()

                #advance SPEED/FPS seconds of simulation, then draw once
                sim_target = data.time + SPEED / FPS
                while data.time < sim_target:
                    mujoco.mj_step(model, data)
                viewer.sync()

                #only sleep if we're ahead of the requested speed
                wait = 1 / FPS - (time.time() - frame_start)
                if wait > 0:
                    time.sleep(wait)
    except KeyboardInterrupt:
        print("Viewer stopped")