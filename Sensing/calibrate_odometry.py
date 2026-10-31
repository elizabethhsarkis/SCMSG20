"""
Odometry check: does wheel odometry match reality for a straight drive and an
on-the-spot turn? Prints the effective wheel radius and separation.
(This is the idea behind the UMBmark odometry calibration.)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import frames as F
from Sensing.driver import TurtleBotDriver


def run(v, w, seconds):
    drv = TurtleBotDriver(enc_noise_k=0.0, enc_ticks_per_rev=None)   # perfect encoders
    drv.reset(F.START_POSE)
    enc0, p0 = drv.read_encoders(), drv.true_pose()
    yaw_true, last = 0.0, p0[2]
    drv.set_velocity(v, w)
    for _ in range(int(seconds / drv.dt)):
        drv.step()
        th = drv.true_pose()[2]
        yaw_true += F.wrap(th - last)        # unwrapped true yaw
        last = th
    drv.set_velocity(0, 0)
    drv.step(250)                            # let it stop
    dR, dL = drv.read_encoders() - enc0
    p1 = drv.true_pose()
    yaw_true += F.wrap(p1[2] - last)
    odo_dist = F.WHEEL_RADIUS / 2 * (dR + dL)
    odo_yaw = F.WHEEL_RADIUS / F.WHEEL_BASE * (dR - dL)
    true_dist = np.hypot(*(p1[:2] - p0[:2]))
    drv.close()
    return odo_dist, true_dist, odo_yaw, yaw_true


od, td, _, _ = run(0.15, 0.0, 5.0)           # straight, about 0.75 m east from the pad
_, _, oy, ty = run(0.0, 0.8, 6.0)            # on the spot, about 275 degrees
print(f"Straight: odometry {od:.3f} m, true {td:.3f} m  -> ratio {td/od:.3f}  "
      f"(effective r = {F.WHEEL_RADIUS*td/od:.4f} m)")
print(f"Turn:     odometry {np.degrees(oy):.1f} deg, true {np.degrees(ty):.1f} deg  -> ratio {ty/oy:.3f}  "
      f"(effective b = {F.WHEEL_BASE*oy/ty:.3f} m)")