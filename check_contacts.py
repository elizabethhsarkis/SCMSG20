# Lists what is touching what after the robot settles on the pad.
import mujoco
from Build_scene import SCENE_OUT

m = mujoco.MjModel.from_xml_path(str(SCENE_OUT))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
for _ in range(500):                         # 1 s to settle
    mujoco.mj_step(m, d)

def describe(g):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom {g}"
    body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
    gtype = mujoco.mjtGeom(m.geom_type[g]).name
    return f"{name} ({gtype}, body {body})"

print(f"{d.ncon} contacts")
for i in range(d.ncon):
    c = d.contact[i]
    print(f"  {describe(c.geom1):45s} <-> {describe(c.geom2):45s} at {c.pos.round(3)}")