# Imports

from pathlib import Path              
import xml.etree.ElementTree as ET     
import cv2 


# File locations
FOLDER = Path(__file__).resolve().parent

ROBOT_IN = FOLDER / "turtlebot3_waffle_pi.xml"        # original ROBOTIS file [ROB], read only
ROBOT_OUT = FOLDER / "turtlebot3_waffle_pi_cam.xml"   # Sensors added
SCENE_OUT = FOLDER / "ward_scene.xml"                 # file opened in MuJoCo
MARKER_PNG = FOLDER / "marker_assets" / "aruco_7.png"

# Ward layout

## origin = bottom left corner of room (south west: x = east, y = north), values in world frame


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
START_X, START_Y = 0.6, 0.6


# ArUco marker
MARKER_ID = 7
MARKER_SIDE = 0.12              #black square is 12cm wide
MARKER_X, MARKER_Y, MARKER_Z = 3.325, 3.399, 0.15           #centre on cabinet front face (y=3.40)

DOCK_POSITION = 0.40                # robot stops 0.40m infront of marker

PX, BORDER = 400, 100               # image size = 400 pixles black square + 100 pixels white border outside (for detection)
MARKER_TOTAL = MARKER_SIDE * (PX+2 * BORDER) / PX

# Sensor mounts on robot - fixed transforms     p_W = T_WB * T_BS * p_s

CAMERA_POS = (0.0855, 0.0, 0.107)               # front face of grey camera box in file
LIDAR_POS = (-0.049, 0.0, 0.1255)               # where ROBOTIS placed LDS LIDAR mesh in scene mesh
CAMERA_FOVY = 48.8                              # vertical filed of view of Rasperbby Pi camera v2, in degrees
# focal length from FOV: f = (H/2) / tan(fovy/2)



# Marker picture                Tut 3, part E
def make_marker_image():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, PX)                      # create marker id & with 400x400 black-and-white image
    tag = cv2.copyMakeBorder(tag, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, value=255)           # add white border
    MARKER_PNG.parent.mkdir(exist_ok=True)          # marker assets - if needed
    cv2.imwrite(str(MARKER_PNG), tag)               # saved for MuJoCo to load


# Camera & LiDAR to copy of ROBOTIS robot (add elements to allow image production)


def add_sensors_to_robot():
    tree = ET.parse(ROBOT_IN)
    root = tree.getroot()


    option = root.find("option")
    option.set("timestep", "0.002")                 # 2ms step
    option.set("integrator", "implicitfast")            # for velocity actuators - wheel motor

    base = root.find(".//body[@name='base']")       # robot body called base
    base.set("pos", f"{START_X} {START_Y} 0")           # spawn at start point (default 0,0 is in walls)

    # MuJoCo camera looks along -z and +y axis, need to look along +x
    ET.SubElement(base, "camera", name="cam", pos=f"{CAMERA_POS[0]} {CAMERA_POS[1]} {CAMERA_POS[2]}", 
                  xyaxes = "0 -1 0 0 0 1", fovy=str(CAMERA_FOVY))

    # LiDAR rays reference point
    ET.SubElement(base, "site", name="lidar", pos=f"{LIDAR_POS[0]} {LIDAR_POS[1]} {LIDAR_POS[2]}", size="0.01" )

    tree.write(ROBOT_OUT)


# Scene
def make_scene():
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
                  f'size="{sx} {sy} {sz}" rgba="0.8 0.8 0.85 1" group="0"/>\n')

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

if __name__ == "__main__":
    make_marker_image()
    add_sensors_to_robot()
    make_scene()
    print("Wrote", ROBOT_OUT.name, SCENE_OUT.name, "and", MARKER_PNG.relative_to(FOLDER))

