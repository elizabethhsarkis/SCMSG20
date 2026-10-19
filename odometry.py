#wheel odometry: estimates the pose from how far each wheel has turned
#
#a stand-in until Person 2's EKF is ready. It never looks at the true pose,
#but small errors build up over the trip (drift). The EKF fixes that using
#the camera. The EKF should offer the same update() method

import numpy as np

from config import WHEEL_RADIUS, WHEEL_SEPARATION


class WheelOdometry:
    def __init__(self, start_pose):
        self.pose = np.array(start_pose, dtype=float)
        self._last_encoder_readings = None

    def update(self, enc_left, enc_right):
        #call once per control step with the encoder angles (rad)
        if self._last_encoder_readings is None:   #first call: just remember
            self._last_encoder_readings = (enc_left, enc_right)
            return self.pose.copy()

        #how far each wheel rolled since last time, in metres
        d_left = WHEEL_RADIUS * (enc_left - self._last_encoder_readings[0])
        d_right = WHEEL_RADIUS * (enc_right - self._last_encoder_readings[1])
        self._last_encoder_readings = (enc_left, enc_right)

        #same maths as wheelSpinToMotion, for distances instead of speeds
        d_centre = (d_left + d_right) / 2
        d_theta = (d_right - d_left) / WHEEL_SEPARATION

        #move along the average heading during the step
        x, y, theta = self.pose
        theta_mid = theta + d_theta / 2
        self.pose = np.array([x + d_centre * np.cos(theta_mid),
                              y + d_centre * np.sin(theta_mid),
                              theta + d_theta])
        return self.pose.copy()