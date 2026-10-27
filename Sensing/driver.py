import sys
from pathlib import Path
 
sys.path.append(str(Path(__file__).resolve().parent.parent))
 
import numpy as np
import mujoco
 
import frames as F
from Build_scene import SCENE_OUT

class TurtleBotDriver:

    def __init__(self, scene_path = SCENE_OUT, seed=0, enc_noise_k=1e-4, enc_ticks_per_rev=4096,
                 lidar_beams=360, lidar_min=0.12, lidar_max=3.5,
                 lidar_noise=True, lidar_dropout=0.0):

        # enc_noise_k = encoder noise varience per radian turned [rad^2/rad], 0 for perfect encoders
        # enc_ticks_per_rev = quantisation of Dynamixel XM430 encoder (4096 counts per revolution), None = switched off
        # lidar_* = LDS-01 data sheet values: 360 beams at 1 degree, 0.12-3.5 m range. 
        # lidar_dropout = probabiliy beam return nothing (for robustness tests)
        
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.rng = np.random.default_rng(seed)
        m = self.model

        # imported names from 'frames.py' - numbers searched once

        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self._jR, self._jL = jid(F.WHEEL_JOINT_R), jid(F.WHEEL_JOINT_L)
        self._aR, self._aL = aid(F.WHEEL_ACT_R), aid(F.WHEEL_ACT_L)
        if min(self._jR, self._jL, self._aR, self._aL) < 0:
            raise ValueError("Wheel joint/actuator names in frames.py do not match the robot XML")
        self._qR, self._qL = m.jnt_qposadr[self._jR], m.jnt_qposadr[self._jL]
        free = np.where(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0][0]
        self._free_q = m.jnt_qposadr[free]      # qpos[x, y, z, qw, qx, qy, qz]
        self.base_body = m.jnt_bodyid[free]
 
        # encoder model 
        self.enc_noise_k = enc_noise_k
        self._tick = None if not enc_ticks_per_rev else 2 * np.pi / enc_ticks_per_rev
 
        # LiDAR model 
        self.lidar_min, self.lidar_max = lidar_min, lidar_max
        self.lidar_noise, self.lidar_dropout = lidar_noise, lidar_dropout
        self.lidar_angles = np.arange(lidar_beams) * 2 * np.pi / lidar_beams
        self._lidar_dirs_L = np.column_stack([np.cos(self.lidar_angles),
                                              np.sin(self.lidar_angles),
                                              np.zeros(lidar_beams)])
        # Geoms the beam must pass through: the robot's own parts (camera mount, chassis) and the flat marker texture. A MuJoCo plane can behave as
        # infinite for ray casting, so the marker is skipped explicitly.
        robot = m.body_rootid[m.geom_bodyid] == m.body_rootid[self.base_body]
        self._lidar_skip = set(np.where(robot)[0].tolist())
        mk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "aruco_marker")
        if mk >= 0:
            self._lidar_skip.add(mk)
        self._geomid = np.zeros(1, dtype=np.int32)
        # Beam origin: the "lidar" site Build_scene adds at LIDAR_POS. If the site is missing, fall back to the same offset from frames.py.
        self._lidar_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, F.LIDAR_SITE)
 
        # --- camera: fixed calibration constants, renderer created lazily -----
        self.K, self.R_BC, self.t_BC = F.get_camera_calibration(scene_path)
        self._renderer = None
 
        self.last_cmd = np.zeros(2)
        self.reset()
        # Check the state-space-model assumptions against the actual model.
        self.geometry = F.check_robot_geometry(m, self.data, self.base_body,
                                               self._jR, self._jL)

    #simulation control
    @property

    def time(self):
        return self.data.time

    def reset(self, pose=None):                 # reset to keyframe 0 (pharmacy pad), or place robot at (x, y, theta)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)     # "home" keyframe
        else:
            mujoco.mj_resetData(self.model, self.data)
        if pose is not None:
            a = self._free_q
            x, y, th = pose
            self.data.qpos[a:a + 2] = x, y
            self.data.qpos[a + 3:a + 7] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)
        # The encoders count from zero at reset, like a robot that has just restarted.
        self._enc_true_last = self._wheel_angles()
        self._enc_meas = np.zeros(2)
        self.last_cmd = np.zeros(2)

    def step(self, n=1):            #advanced by n steps of dt(2ms each)
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    #command

    def set_wheel_velocities(self, wR, wL):

        # speeds clipped to actuator ctrlrange (+/-7.88)
        #XML = velocity actuators (ctrl = target speed not acheived)
        # kv = 0.1, joint frictionloss = 0.1Nm = wheel speed around 1 rad/s below target
        # commanded speed != actual speed (EKF = encoder increments as inputs not commands)
        
        wR = float(np.clip(wR, -F.MAX_WHEEL_SPEED, F.MAX_WHEEL_SPEED))
        wL = float(np.clip(wL, -F.MAX_WHEEL_SPEED, F.MAX_WHEEL_SPEED))
        self.data.ctrl[self._aR] = wR
        self.data.ctrl[self._aL] = wL
        self.last_cmd = np.array([wR, wL])

    def set_velocity(self, v, w):               #command v[m/s], omega[rad/s] = LQR outputs
        self.set_wheel_velocities(*F.unicycle_to_wheels(v, w))
 
    # sensors
    def _wheel_angles(self):
        return np.array([self.data.qpos[self._qR], self.data.qpos[self._qL]])

    def read_encoders(self):

        # returns cumulative measured wheel angles [phiR, phiL] in rads

        # Error model matchs Estimation assumptions****
            # increments get zero-mean Gaussian noise with variance k*[delta-phi]
            # noise accumulates (random walk) = odometry drifting
            # quantisation to 4096 applied to cumulative angle = does not accumulate
        
        true = self._wheel_angles()
        d = true - self._enc_true_last
        self._enc_true_last = true
        if self.enc_noise_k > 0:
            d = d + self.rng.normal(0.0, np.sqrt(self.enc_noise_k * np.abs(d)))
        self._enc_meas = self._enc_meas + d
        if self._tick:
            return np.round(self._enc_meas / self._tick) * self._tick
        return self._enc_meas.copy()

    def _cast(self, origin, direction):
        
        # return distance from one ray casted to first geom not skipped 
        
        travelled, p = 0.0, origin.copy()
        for _ in range(5):
            d = mujoco.mj_ray(self.model, self.data, p, direction, None, 1,
                              self.base_body, self._geomid)
            if d < 0:
                return np.inf
            if int(self._geomid[0]) in self._lidar_skip:
                hop = d + 1e-4
                travelled += hop
                p = p + hop * direction
                continue
            return travelled + d
        return np.inf

    def read_lidar(self):
        # return a 360 LDS-01 scan as angles [rad] in frame L with ranges [m]
        # ranges = np.inf = no return within lidar_max
        # ranges = np.nan = invalid reading (too close, dropout...)

        #Noise = LDS-01 data sheet = accuracy of +/- 15mm below 0.5m and +/- 5% from 0.5 to 3,5m
        # bounds treated as ~2 sigma

        R_WB = self.data.xmat[self.base_body].reshape(3, 3)
        if self._lidar_site >= 0:
            origin = self.data.site_xpos[self._lidar_site].copy()
        else:
            origin = self.data.xpos[self.base_body] + R_WB @ F.LIDAR_OFFSET
        dirs_W = np.ascontiguousarray(self._lidar_dirs_L @ R_WB.T)
        r = np.array([self._cast(origin, d) for d in dirs_W])
 
        finite = np.isfinite(r)
        if self.lidar_noise:
            sigma = np.where(r[finite] < 0.5, 0.0075, 0.025 * r[finite])
            r[finite] += self.rng.normal(0.0, sigma)
        r[r > self.lidar_max] = np.inf
        r[r < self.lidar_min] = np.nan
        if self.lidar_dropout > 0:
            r[self.rng.random(r.size) < self.lidar_dropout] = np.nan
        return self.lidar_angles.copy(), r

    def read_camera(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, F.IMG_H, F.IMG_W)
        self._renderer.update_scene(self.data, camera=F.CAMERA_NAME)
        return self._renderer.render().copy()

    ## Evaluation only ##
    def true_pose(self):
        # return true [x, y, theta] - for use in error plots, NEES and logs
        
        a, q = self._free_q, self.data.qpos
        return np.array([q[a], q[a+1], F.wrap(2 * np.arctan2(q[a + 6], q[a + 3]))])

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None