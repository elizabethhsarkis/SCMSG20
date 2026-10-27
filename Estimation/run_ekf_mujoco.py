import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import mujoco

from Build_scene import SCENE_OUT, MARKER_X, MARKER_Y
from View_scene import marker_range_bearing, IMG_W, IMG_H
from ekf import EKF, wrap

#steps every 0.002s
EKF_EVERY = 10 # prediction timing every 0.02s
CAM_EVERY = 25 # camera timing every 0.05s
T_END = 12
HIDE = (4,7) #window where detections are ignored

K_ENC = 1e-4
R_RB = np.diag([0.02**2,np.deg2rad(2)**2])

def true_pose(data):
    """
    simulator pose [x,y,theta].
    Evaluation only, not passed to the EKF
    """
    qw = data.qpos[3]
    qz = data.qpos[6]
    return np.array([data.qpos[0], data.qpos[1], 2 * np.arctan2(qz,qw)])

def simulate(start = (3.325,1.6,np.pi/2)):
    """
    Drive the robot towards the marker in the scene with the EKF in the loop
    returns:
    dict
    logged time "t" 
    errors "e" (N x 3)
    sigmas "sigma" (N x 3)
    "nis" 
    the final true and estimated poses
    """
    model = mujoco.MjModel.from_xml_path(str(SCENE_OUT))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,0)

    #places the robot below the marker facing north
    data.qpos[0:7] = [start[0], start[1], 0, np.cos(start[2]/2), 0, 0, np.sin(start[2]/2)]
    mujoco.mj_forward(model, data)

    #looks up the joins and actuators
    # the wheel joint angles in qpos are the encoders
    joint = lambda name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    actuator = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    qL = joint("wheel_left")
    qR = joint("wheel_right")
    aL = actuator("wheel_left")
    aR = actuator("wheel_right")
    renderer = mujoco.Renderer(model,IMG_H,IMG_W)

    #start the EKF at known start pose (1cm and 1 deg uncertainty)
    ekf = EKF(start, np.diag([0.01**2, 0.01**2, np.deg2rad(1)**2]), K_ENC, R_RB)
    last = np.array([data.qpos[qR], data.qpos[qL]])
    log = {"t": [], "e": [], "sigma": [], "nis": []}

    rng = np.random.default_rng(0)
    step = 0
    while data.time < T_END:
        t = data.time

        #forward with a slight weave, so heading changes and becomes observable
        data.ctrl[aL] = 3 - 0.6 * np.sin(1.5 * t)
        data.ctrl[aR] = 3 + 0.6 * np.sin(1.5 * t)
        mujoco.mj_step(model,data)
        step += 1

        #predict from the wheel angle change
        if step % EKF_EVERY == 0:
            now = np.array([data.qpos[qR], data.qpos[qL]])
            dR, dL = now - last
            last = now
            ekf.predict(dR, dL)

        #update from the camera
        if step % CAM_EVERY == 0 and not (HIDE[0] <= t < HIDE[1]):
            renderer.update_scene(data, camera="cam")
            z = marker_range_bearing(renderer.render())
            if z is not None:
                #simulated camera noise for now
                z = (z[0] + rng.normal(0,0.02), z[1] + rng.normal(0, np.deg2rad(2)))
                #trust further readings less
                ekf.R_rb = np.diag([0.02**2 + (0.01 * z[0])**2, np.deg2rad(2)**2])
                nu, S = ekf.update_range_bearing(z, (MARKER_X, MARKER_Y))
                log["nis"].append(nu @ np.linalg.solve(S, nu))
        
        #log error for evaluation
        if step % EKF_EVERY == 0:
            e = ekf.x - true_pose(data)
            e[2] = wrap(e[2])
            log["t"].append(t)
            log["e"].append(e)
            log["sigma"].append(np.sqrt(np.diag(ekf.P)))

    lg = {key: np.array(val) for key, val in log.items()}
    lg["x_true"] = true_pose(data)
    lg["x_est"] = ekf.x.copy()
    return lg

if __name__ == "__main__":
    lg = simulate()
    rmse = np.sqrt((lg["e"]**2).mean(axis=0))
    np.save("ekf_log.npy", np.column_stack([lg["t"], lg["e"], lg["sigma"]]))
    print(f"Final pose: {np.round(lg['x_true'], 3)}")
    print(f"Final estmate: {np.round(lg['x_est'], 3)}")
    print(f"RMSE x, y, theta: {np.round(rmse,4)}")
    print(f"Marker updates: {len(lg['nis'])}, mean NIS {np.mean(lg['nis']):.2f}")# should be ~2




