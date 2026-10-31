# Imports

import math
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2

from ward_select import WARD           # "paeds" (default) or "classic", chosen with --ward or the WARD variable

CLASSIC = WARD == "classic"

# File locations
FOLDER = Path(__file__).resolve().parent

ROBOT_IN = FOLDER / "turtlebot3_waffle_pi.xml"        # original ROBOTIS file [ROB], read only
# each ward gets its own generated files, so both can exist side by side
_SUFFIX = "_classic" if CLASSIC else ""
ROBOT_OUT = FOLDER / f"turtlebot3_waffle_pi_cam{_SUFFIX}.xml"   # Sensors added
SCENE_OUT = FOLDER / f"ward_scene{_SUFFIX}.xml"                 # file opened in MuJoCo

# ArUco marker size (both wards)
MARKER_SIDE = 0.12              #black square is 12cm wide
DOCK_POSITION = 0.40                # robot stops 0.40m infront of marker

# Ward layout

## origin = bottom left corner of room (south west: x = east, y = north), values in world frame

if CLASSIC:
    # ---------------- the original ward: 6 x 4 m, one bedside marker ----------------
    ROOM_X, ROOM_Y = 6.0, 4.0               #room is 6m by 4m
    WALL_THICKNESS, WALL_HEIGHT = 0.10, 0.5

    # Obstacles             name, x_min, y_min, x_max, y_max, height
    OBSTACLES = [
        ("bed_A",       1.75, 2.00, 2.65, 4.00, 0.60),
        ("bed_B",       4.00, 2.00, 4.90, 4.00, 0.60),
        ("cabinet",     3.00, 3.40, 3.575, 3.90, 0.50),              #carries marker
        ("trolley",     2.80, 0.80, 3.40, 1.20, 0.90),
    ]

    # robot starting position,      heading 0 means facing x+ (east)
    START_X, START_Y, START_THETA = 0.6, 0.6, 0.0
    START_REGION = ((0.4, 1.4), (0.4, 1.5))       # random trial starts (x range, y range)

    # ArUco marker on the cabinet front face (y=3.40), facing south (-y) into the ward
    MARKERS = [{"name": "bedside", "id": 7, "x": 3.325, "y": 3.399, "z": 0.15,
                "facing": -math.pi / 2, "label": "Bedside"}]
    MISSION = ["bedside"]
    TROLLEY_MOVE_TO = (2.3, 1.5)                   # moved_trolley test
    LIDAR_GEOMGROUPS = None                        # LiDAR sees every geom group (original behaviour)
else:
    # ---------------- the paediatric ward: 4 beds, a marker per bed, a dock ----------
    from ward_paeds import (ROOM_X, ROOM_Y, WALL_THICKNESS, WALL_HEIGHT, OBSTACLES,
                            START_X, START_Y, START_THETA, START_REGION, MARKERS, MISSION,
                            TROLLEY_MOVE_TO)
    LIDAR_GEOMGROUPS = (1, 1, 0, 1, 1, 1)          # LiDAR ignores group 2 (decoration and markers)

MARKER_BY_NAME = {m["name"]: m for m in MARKERS}
# the FINAL stop's marker keeps the original names, so single-marker code still works
_FINAL = MARKER_BY_NAME[MISSION[-1]]
MARKER_ID = _FINAL["id"]
MARKER_X, MARKER_Y, MARKER_Z = _FINAL["x"], _FINAL["y"], _FINAL["z"]
MARKER_PNG = FOLDER / "marker_assets" / f"aruco_{MARKER_ID}.png"

PX, BORDER = 400, 100               # image size = 400 pixles black square + 100 pixels white border outside (for detection)
MARKER_TOTAL = MARKER_SIDE * (PX+2 * BORDER) / PX

# Sensor mounts on robot - fixed transforms     p_W = T_WB * T_BS * p_s

CAMERA_POS = (0.0855, 0.0, 0.107)               # front face of grey camera box in file
LIDAR_POS = (-0.049, 0.0, 0.1255)               # where ROBOTIS placed LDS LIDAR mesh in scene mesh
CAMERA_FOVY = 48.8                              # vertical filed of view of Rasperbby Pi camera v2, in degrees
# focal length from FOV: f = (H/2) / tan(fovy/2)



# Marker picture                Tut 3, part E
def make_marker_image(marker_id=MARKER_ID):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag = cv2.aruco.generateImageMarker(dictionary, marker_id, PX)                      # create marker id & with 400x400 black-and-white image
    tag = cv2.copyMakeBorder(tag, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, value=255)           # add white border
    png = FOLDER / "marker_assets" / f"aruco_{marker_id}.png"
    png.parent.mkdir(exist_ok=True)          # marker assets - if needed
    cv2.imwrite(str(png), tag)               # saved for MuJoCo to load


# Camera & LiDAR to copy of ROBOTIS robot (add elements to allow image production)

# Mesh files: find the ROBOTIS STL meshes and give MuJoCo full paths.
# Relative meshdir paths break when the robot file is included from another folder.
def find_mesh_dir():
    candidates = [FOLDER, FOLDER / "assets"] + sorted(FOLDER.parent.glob("robotis_mujoco_menagerie*"))
    for root in candidates:
        if (root / "waffle_pi_base.stl").exists():
            return root
        if root.is_dir():
            hits = list(root.rglob("waffle_pi_base.stl"))
            if hits:
                return hits[0].parent
    raise FileNotFoundError("waffle_pi_base.stl not found - copy the ROBOTIS 'assets' "
                            "folder into the repo folder")

def add_sensors_to_robot():
    tree = ET.parse(ROBOT_IN)
    root = tree.getroot()

    # Point every mesh at its real file (absolute path), and drop the relative meshdir
    mesh_dir = find_mesh_dir()
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    for mesh in root.iter("mesh"):
        mesh.set("file", (mesh_dir / Path(mesh.get("file")).name).as_posix())

    option = root.find("option")
    option.set("timestep", "0.002")                 # 2ms step
    option.set("integrator", "implicitfast")            # for velocity actuators - wheel motor

    # Wheel motors: ROBOTIS kv=0.1 is too weak to turn on the spot against tyre scrub
    # (torque = kv * speed error, and frictionloss 0.1 N m eats the first 1 rad/s of error).
    # kv=1.0 tracks speed to ~0.1 rad/s; forcerange caps torque near the Dynamixel
    # XM430's real output so the motor isn't unrealistically strong.
    wheel_vel = root.find(".//default[@class='wheel']/velocity")
    wheel_vel.set("kv", "1.0")
    wheel_vel.set("forcerange", "-1.5 1.5")

    # Tyres: replace the ROBOTIS mesh collision with a primitive cylinder (r = 0.033 m).
    # A mesh tyre collides as its convex hull and scrubs during on-the-spot turns, so
    # the wheels slip and odometry over-counts the turn. A cylinder gives clean rolling
    # contact. The wheel body's z axis is the spin axis, which is also the cylinder axis.
    for wheel in ("wheel_left", "wheel_right"):
        body = root.find(f".//body[@name='{wheel}']")
        for g in list(body.findall("geom")):
            if g.get("class") == "collision":
                body.remove(g)
        ET.SubElement(body, "geom", name=f"{wheel}_tyre", type="cylinder",
                      size="0.033 0.009", group="3", condim="3", friction="1.0 0.005 0.0001")
    
    base = root.find(".//body[@name='base']")       # robot body called base
    base.set("pos", f"{START_X} {START_Y} 0")           # spawn at start point (default 0,0 is in walls)

    # Chassis collision hull: the convex hull of waffle_pi_base.stl reaches the floor
    # and drags, so the wheels slip. Bit 2 only: it still hits obstacles (bits 1+2,
    # set in make_scene) but not the floor (bit 1). The casters carry the rear as on
    # the real robot.
    for g in base.findall("geom"):
        if g.get("class") == "collision" and g.get("mesh") == "waffle_pi_base":
            g.set("contype", "2")
            g.set("conaffinity", "2")

    # Caster spheres: ROBOTIS gives them friction 0.0001 and condim 1, but MuJoCo
    # combines contact settings with the floor by taking the max, so they end up
    # with friction 1.0 and condim 3 and drag like rubber feet. priority="1" makes
    # the caster's own (near-frictionless) settings win against the floor.
    for g in base.findall("geom"):
        if g.get("type") == "sphere":
            g.set("priority", "1")

    # MuJoCo camera looks along -z and +y axis, need to look along +x
    ET.SubElement(base, "camera", name="cam", pos=f"{CAMERA_POS[0]} {CAMERA_POS[1]} {CAMERA_POS[2]}", 
                  xyaxes = "0 -1 0 0 0 1", fovy=str(CAMERA_FOVY))

    # LiDAR rays reference point
    ET.SubElement(base, "site", name="lidar", pos=f"{LIDAR_POS[0]} {LIDAR_POS[1]} {LIDAR_POS[2]}", size="0.01" )

    tree.write(ROBOT_OUT)


# Scene
def make_scene():
    if not CLASSIC:
        # the paediatric ward has its own (much bigger) scene writer in ward_paeds.py
        import ward_paeds
        ward_paeds.write_textures(FOLDER)
        SCENE_OUT.write_text(ward_paeds.scene_xml(ROBOT_OUT.name, MARKER_TOTAL))
        return

    walls = [
        ("wall_S", -WALL_THICKNESS, -WALL_THICKNESS, ROOM_X + WALL_THICKNESS, 0.0, WALL_HEIGHT),
        ("wall_N", -WALL_THICKNESS, ROOM_Y, ROOM_X + WALL_THICKNESS, ROOM_Y + WALL_THICKNESS, WALL_HEIGHT),
        ("wall_W", -WALL_THICKNESS, 0.0, 0.0, ROOM_Y, WALL_HEIGHT),
        ("wall_E", ROOM_X, 0.0, ROOM_X + WALL_THICKNESS, ROOM_Y, WALL_HEIGHT),
    ]

    boxes = ""
    for name, x0, y0, x1, y1, h in walls + OBSTACLES:
        cx, cy = (x0+x1)/2, (y0+y1)/2                   # centre of box
        sx, sy, sz = (x1-x0)/2, (y1-y0)/2, h/2          # half-sizes

        boxes += (f'  <geom name="{name}" type="box" pos="{cx} {cy} {sz}" '
                  f'size="{sx} {sy} {sz}" rgba="0.8 0.8 0.85 1" group="0" '
                  f'contype="3" conaffinity="3"/>\n')

    half = MARKER_TOTAL / 2             # MuJoCo plane sizes are half-widths
    dock_y = MARKER_Y - DOCK_POSITION

    scene = f"""<mujoco model="bedside ward">
  <!-- Robot first: the ROBOTIS Waffle Pi with our camera and LiDAR site [ROB] -->
  <include file="{ROBOT_OUT.name}"/>
 
  <!-- Viewer lighting and sky, copied from the ROBOTIS / UR3e scene files [ROB][UR3e] -->
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.62 0.64 0.66" rgb2="0.58 0.60 0.62" width="300" height="300"/>
    <material name="floor" texture="grid" texuniform="true" texrepeat="2 2"/>
    <!-- the ArUco picture made in make_marker_image() [T3-E] -->
    <texture type="2d" name="aruco" file="marker_assets/aruco_7.png"/>
    <material name="aruco" texture="aruco"/>
  </asset>
 
  <worldbody>
    <light pos="3 2 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>
{boxes}
    <!-- ArUco marker: a flat plane on the cabinet face, facing -y into the ward.
         xyaxes="1 0 0 0 0 1": plane x along +x_W, plane y straight up, so its normal is -y_W.
         contype/conaffinity 0 and group 2: visible to the camera, invisible to physics and LiDAR [MJ] -->
    <geom name="aruco_marker" type="plane" pos="{MARKER_X} {MARKER_Y - 0.001} {MARKER_Z}"
          xyaxes="1 0 0 0 0 1" size="{half} {half} 0.01" material="aruco"
          contype="0" conaffinity="0" group="2"/>
 
    <!-- Floor decals for the video: blue = pharmacy start, pink = dock goal (visual only) -->
    <geom type="cylinder" pos="{START_X} {START_Y} 0.001" size="0.15 0.001" rgba="0.2 0.6 0.9 0.5" contype="0" conaffinity="0" group="2"/>
    <geom type="cylinder" pos="{MARKER_X} {dock_y} 0.001" size="0.12 0.001" rgba="0.9 0.3 0.6 0.5" contype="0" conaffinity="0" group="2"/>
  </worldbody>
 
  <!-- "home" keyframe, like the UR3e's [UR3e]. qpos = robot x y z, orientation quaternion
       (1 0 0 0 = facing +x), then the two wheel angles. Reset in the viewer returns here. -->
  <keyframe>
    <key name="home" qpos="{START_X} {START_Y} 0 1 0 0 0 0 0" ctrl="0 0"/>
  </keyframe>
</mujoco>
"""
    SCENE_OUT.write_text(scene)


# Run code directly

#   python Build_scene.py                  builds the paediatric ward (default)
#   python Build_scene.py --ward classic   builds the original ward
#   python Build_scene.py --all            builds both
if __name__ == "__main__":
    if "--all" in sys.argv:
        for ward in ("paeds", "classic"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--ward", ward], check=True)
        sys.exit(0)
    for m in MARKERS:
        make_marker_image(m["id"])
    add_sensors_to_robot()
    make_scene()
    print(f"[{WARD} ward] wrote", ROBOT_OUT.name, SCENE_OUT.name, "and", len(MARKERS), "marker image(s)")

