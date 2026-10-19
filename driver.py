#driver layer: the ONLY code that talks to MuJoCo
#
#everything else gets sensor readings from here and sends commands back
#through here. For the real robot, write a second class with these same
#methods that talks to the TurtleBot, and nothing else has to change

import mujoco
import numpy as np

from config import LIDAR_RANGE, LIDAR_RAYS, WHEEL_RADIUS, WHEEL_SEPARATION
from scene import build_ward_xml


class MujocoDriver:
    def __init__(self, camera=False):
        self.model = mujoco.MjModel.from_xml_string(build_ward_xml())
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)


        #look things up by name once, so the control loop is fast
        m = self.model
        self._left_encoder_adr = m.sensor("enc_left").adr[0]
        self._right_encoder_adr = m.sensor("enc_right").adr[0]
        self._lidar_ray_adrs = [m.sensor(f"ray_{i}").adr[0] for i in range(LIDAR_RAYS)]
        self._left_motor_id = m.actuator("motor_left").id
        self._right_motor_id = m.actuator("motor_right").id
        self._base_body_id = m.body("base").id
        self._max_wheel_speed = m.actuator_ctrlrange[self._left_motor_id, 1]
        self.lidar_angles = 2 * np.pi * np.arange(LIDAR_RAYS) / LIDAR_RAYS
        self.renderer = mujoco.Renderer(m, 240, 320) if camera else None


    #---------------- sensors ----------------
    def read_encoders(self):
        #wheel angles (left, right) in radians
        d = self.data.sensordata
        return float(d[self._left_encoder_adr]), float(d[self._right_encoder_adr])
    def read_lidar(self):
        #ray angles (robot frame) and ranges in metres; inf = nothing hit
        r = self.data.sensordata[self._lidar_ray_adrs].copy()
        r[(r < 0) | (r >= LIDAR_RANGE)] = np.inf
        return self.lidar_angles, r

    def read_camera(self):
        #RGB image from the front camera, for Person 1's marker detection
        if self.renderer is None:
            raise RuntimeError("Create the driver with camera=True first")
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False #hide LiDAR rays
        self.renderer.update_scene(self.data, camera="front_camera", scene_option=opt)
        return self.renderer.render()

    #---------------- motors ----------------
    def set_twist(self, v, w):
        #drive at v m/s and w rad/s, converted to wheel speeds
        left_wheel_speed = (v - w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
        right_wheel_speed = (v + w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
        max_speed = self._max_wheel_speed
        self.data.ctrl[self._left_motor_id] = np.clip(left_wheel_speed, -max_speed, max_speed)
        self.data.ctrl[self._right_motor_id] = np.clip(right_wheel_speed, -max_speed, max_speed)

    def advance(self, dt):
        #let dt seconds of physics happen (many small MuJoCo steps)
        n = max(1, int(round(dt / self.model.opt.timestep)))
        mujoco.mj_step(self.model, self.data, nstep=n)
    #---------------- evaluation only ----------------
    def true_pose(self):
        #simulator ground truth. FOR LOGGING AND PLOTS ONLY:
        #the controller must never use this
        x, y = self.data.xpos[self._base_body_id][:2]
        q = self.data.xquat[self._base_body_id]  #orientation as a quaternion (w, x, y, z)
        theta = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                           1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return np.array([x, y, theta])