"""
Sensing/diagnose_bounce.py - is the robot bouncing or rocking while it drives?

Drives straight at 0.15 m/s, waits 2 s for steady speed, then watches 2 s
(1000 physics steps). It prints:
  * the fraction of steps each wheel/caster is touching the floor and its mean normal force
  * the body height and pitch wobble (std), and the wheel-speed wobble
  * the slip ratio averaged over the window (not a single snapshot)
Perfect rolling means: every contact present ~100% of the time, tiny wobble, slip ratio ~1.00.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import mujoco

import frames as F
from Sensing.driver import TurtleBotDriver


def run(variant):
    drv = TurtleBotDriver(enc_noise_k=0.0, enc_ticks_per_rev=None)
    m, d = drv.model, drv.data
    if variant == "noslip":
        m.opt.noslip_iterations = 10
    free = np.where(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0][0]
    qa, va = m.jnt_qposadr[free], m.jnt_dofadr[free]
    dR, dL = m.jnt_dofadr[drv._jR], m.jnt_dofadr[drv._jL]

    # the four support geoms: tyres by name, casters = sphere geoms on the base body
    support = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g): g for g in range(m.ngeom)
               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_tyre")}
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == drv.base_body and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            support[f"caster{g}"] = g
    inv = {g: n for n, g in support.items()}

    drv.reset(F.START_POSE)
    drv.set_velocity(0.15, 0.0)
    drv.step(1000)                                        # 2 s to reach steady speed

    N = 1000
    touching = {n: 0 for n in support}
    fn_sum = {n: 0.0 for n in support}
    z, pitch, vb, vw, wspd = [], [], [], [], []
    for _ in range(N):
        drv.step()
        seen = set()
        for i in range(d.ncon):
            c = d.contact[i]
            g = c.geom1 if c.geom1 in inv else c.geom2 if c.geom2 in inv else None
            if g is None:
                continue
            f6 = np.zeros(6)                              # fresh array each contact
            mujoco.mj_contactForce(m, d, i, f6)
            seen.add(inv[g])
            fn_sum[inv[g]] += f6[0]
        for n in seen:
            touching[n] += 1
        qw, qx, qy, qz = d.qpos[qa + 3:qa + 7]
        z.append(d.qpos[qa + 2])
        pitch.append(np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))))
        vb.append(np.linalg.norm(d.qvel[va:va + 2]))
        vw.append(F.WHEEL_RADIUS * (d.qvel[dR] + d.qvel[dL]) / 2)
        wspd.append(d.qvel[dR])

    print(f"\n=== {variant} ===")
    print(f"  slip ratio (2 s mean) {np.mean(vb) / np.mean(vw):.3f}")
    print(f"  body height  mean {np.mean(z)*1000:6.2f} mm  std {np.std(z)*1000:.3f} mm")
    print(f"  body pitch   mean {np.mean(pitch):6.2f} deg std {np.std(pitch):.3f} deg")
    print(f"  right wheel speed mean {np.mean(wspd):.2f} rad/s  std {np.std(wspd):.3f} rad/s")
    for n in support:
        print(f"  {n:18s} in contact {100 * touching[n] / N:5.1f}% of steps   "
              f"mean Fn {fn_sum[n] / N:5.2f} N per step")
    drv.close()


for v in ("default", "noslip"):
    run(v)
