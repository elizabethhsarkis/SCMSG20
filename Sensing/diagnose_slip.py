"""
Sensing/diagnose_slip.py - why does the robot travel less than its wheels say?

Drives straight at 0.15 m/s for 4 s under a few physics variants (changed in
memory only; nothing is written to the XML). For each one it prints the
steady-state slip and every contact force, so we can see whether something is
resisting the motion or the soft contact is just "creeping".

  slip ratio = body speed / (r * mean wheel speed)   (1.00 = perfect rolling)
  Ft         = tangential (friction) force at each contact [N]
  pen        = contact penetration [mm]
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import mujoco

import frames as F
from Sensing.driver import TurtleBotDriver


def geom_name(m, g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or \
        f"geom{g}({mujoco.mjtGeom(m.geom_type[g]).name[7:].lower()})"


def straight(variant, seconds=4.0):
    drv = TurtleBotDriver(enc_noise_k=0.0, enc_ticks_per_rev=None)
    m, d = drv.model, drv.data
    tyres = [g for g in range(m.ngeom) if (geom_name(m, g)).endswith("_tyre")]
    if variant == "noslip":
        m.opt.noslip_iterations = 10            # MuJoCo's post-solver that removes soft-contact slip
    elif variant == "stiff tyre contact":
        for g in tyres:
            m.geom_solref[g] = [0.004, 1.0]     # default is [0.02, 1]: 5x stiffer contact
    drv.reset(F.START_POSE)
    enc0, p0 = drv.read_encoders(), drv.true_pose()
    drv.set_velocity(0.15, 0.0)
    drv.step(int(seconds / drv.dt))

    free = np.where(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0][0]
    v_body = np.linalg.norm(d.qvel[m.jnt_dofadr[free]:m.jnt_dofadr[free] + 2])
    wR = d.qvel[m.jnt_dofadr[drv._jR]]
    wL = d.qvel[m.jnt_dofadr[drv._jL]]
    v_wheel = F.WHEEL_RADIUS * (wR + wL) / 2
    dR, dL = drv.read_encoders() - enc0
    odo = F.WHEEL_RADIUS * (dR + dL) / 2
    true = np.hypot(*(drv.true_pose()[:2] - p0[:2]))

    print(f"\n=== {variant} ===")
    print(f"  steady slip ratio {v_body / v_wheel:.3f}   (body {v_body:.4f} m/s, "
          f"wheels {v_wheel:.4f} m/s)   distance ratio {true / odo:.3f}")
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        mujoco.mj_contactForce(m, d, i, f6)
        a, b = geom_name(m, c.geom1), geom_name(m, c.geom2)
        print(f"  {a:>14s} - {b:<18s} Fn {f6[0]:6.2f} N   Ft {np.hypot(f6[1], f6[2]):6.3f} N   "
              f"pen {-c.dist * 1000:5.2f} mm")
    drv.close()


for v in ("default", "noslip", "stiff tyre contact"):
    straight(v)
