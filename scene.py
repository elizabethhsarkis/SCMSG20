#builds the ward and the robot as a MuJoCo model (XML text)
#
#colours are plain and neutral: light grey floor, off-white walls, a grey
#bed and a dark grey robot. The marker is black on white, as ArUco needs

import numpy as np

from config import (LIDAR_HEIGHT, LIDAR_RANGE, LIDAR_RAYS, MARKER_XY, PHARMACY_POSE,
                    WARD_BOXES, WARD_HEIGHT, WARD_WIDTH, WHEEL_RADIUS,
                    WHEEL_SEPARATION)

#colours as "red green blue opacity", each between 0 and 1
FLOOR_RGBA = "0.82 0.82 0.80 1"
WALL_RGBA = "0.92 0.92 0.90 1"
FURNITURE_RGBA = "0.70 0.70 0.68 1"
ROBOT_RGBA = "0.25 0.25 0.25 1"


def _build_obstacles_xml():
    #one MuJoCo box per entry in WARD_BOXES
    #muJoCo boxes are given by their CENTRE and HALF sizes
    lines = []
    for name, x0, x1, y0, y1, h in WARD_BOXES:
        colour = WALL_RGBA if name.startswith("wall") else FURNITURE_RGBA
        lines.append(f'<geom name="{name}" type="box" '
                     f'pos="{(x0 + x1) / 2} {(y0 + y1) / 2} {h / 2}" '
                     f'size="{(x1 - x0) / 2} {(y1 - y0) / 2} {h / 2}" rgba="{colour}"/>')
    return "\n    ".join(lines)


def _build_lidar_xml():
    #muJoCo has no LiDAR, but a rangefinder measures the distance along one
    #ray (a site's z-axis). So we place one site per ray, each pointing
    #horizontally in a different direction, and one rangefinder on each
    sites, sensors = [], []
    for i in range(LIDAR_RAYS):
        ang = 2 * np.pi * i / LIDAR_RAYS
        sites.append(f'<site name="lidar_{i}" pos="0 0 {LIDAR_HEIGHT - 0.04}" '
                     f'zaxis="{np.cos(ang):.5f} {np.sin(ang):.5f} 0" size="0.003"/>')
        sensors.append(f'<rangefinder name="ray_{i}" site="lidar_{i}" cutoff="{LIDAR_RANGE}"/>')
    return "\n      ".join(sites), "\n    ".join(sensors)


def build_ward_xml():
    lidar_sites_xml, lidar_sensors_xml = _build_lidar_xml()
    mx, my = MARKER_XY
    start_x, start_y, start_theta = PHARMACY_POSE
    half = WHEEL_SEPARATION / 2
    return f"""
<mujoco model="bedside_ward">
  <!-- RK4 integrator: the default "implicitfast" mishandles the light wheels
       driven by stiff motors, and the robot stalls mid-turn while its wheels
       keep spinning. RK4 is slower but accurate. -->
  <option timestep="0.002" integrator="RK4"/>

  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.25 0.25 0.25"/>
  </visual>

  <worldbody>
    <!-- Directional light: even, like daylight, with no bright spot. -->
    <light directional="true" pos="0 0 6" dir="0.2 0.3 -1" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" size="{WARD_WIDTH} {WARD_HEIGHT} 0.1"
          pos="{WARD_WIDTH / 2} {WARD_HEIGHT / 2} 0" rgba="{FLOOR_RGBA}"/>
    {_build_obstacles_xml()}
    <!-- Bedside marker on the foot of the bed: a black square on a white
         border for now. Person 1 replaces it with a real ArUco texture. -->
    <geom name="marker_border" type="box" pos="{mx} {my - 0.006} 0.25"
          size="0.08 0.004 0.08" rgba="1 1 1 1" contype="0" conaffinity="0"/>
    <geom name="marker" type="box" pos="{mx} {my - 0.011} 0.25"
          size="0.06 0.002 0.06" rgba="0 0 0 1" contype="0" conaffinity="0"/>
    <!-- The robot. A freejoint lets the whole body move and turn freely. -->
    <body name="base" pos="{start_x} {start_y} 0.04" euler="0 0 {start_theta}">
      <freejoint name="base_free"/>
      <geom name="chassis" type="cylinder" size="0.07 0.06" pos="-0.03 0 0.06"
            mass="0.9" rgba="{ROBOT_RGBA}"/>
      <geom name="lidar_housing" type="cylinder" size="0.035 0.02"
            pos="0 0 {LIDAR_HEIGHT - 0.04}" mass="0.1" rgba="{ROBOT_RGBA}"
            contype="0" conaffinity="0"/>


      <!-- Ball casters front and back, 2 mm above the floor. The wheels carry
           the weight (so they grip); the casters only touch down if the robot
           rocks, which stops it tipping onto its nose when it brakes.
           condim="1" makes them frictionless, and priority="1" makes that win
           over the floor's friction. -->
      <geom name="caster_back" type="sphere" size="0.0075" pos="-0.08 0 -0.0305"
            condim="1" priority="1" mass="0.01" rgba="{ROBOT_RGBA}"/>
      <geom name="caster_front" type="sphere" size="0.0075" pos="0.05 0 -0.0305"
            condim="1" priority="1" mass="0.01" rgba="{ROBOT_RGBA}"/>

      <camera name="front_camera" pos="0.04 0 0.12" xyaxes="0 -1 0 0 0 1" fovy="62"/>
      {lidar_sites_xml}
      <!-- Two driven wheels. You see a cylinder; what touches the floor is a
           sphere of the same radius, which gives one clean contact point
           (a cylinder touches at several points and makes the robot judder). -->
      <body name="wheel_left" pos="0 {half} -0.007">
        <joint name="wheel_left_joint" type="hinge" axis="0 1 0" damping="0.001"/>
        <geom type="cylinder" size="{WHEEL_RADIUS} 0.009" euler="90 0 0"
              contype="0" conaffinity="0" mass="0.03" rgba="{ROBOT_RGBA}"/>
        <geom type="sphere" size="{WHEEL_RADIUS}" friction="1.0 0.005 0.0001"
              mass="0.001" rgba="0 0 0 0"/>
      </body>
      <body name="wheel_right" pos="0 {-half} -0.007">
        <joint name="wheel_right_joint" type="hinge" axis="0 1 0" damping="0.001"/>
        <geom type="cylinder" size="{WHEEL_RADIUS} 0.009" euler="90 0 0"
              contype="0" conaffinity="0" mass="0.03" rgba="{ROBOT_RGBA}"/>
        <geom type="sphere" size="{WHEEL_RADIUS}" friction="1.0 0.005 0.0001"
              mass="0.001" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>

  <!-- Velocity motors: ask for a wheel speed (rad/s) and the motor pushes the
       wheel towards it. forcerange caps the torque; without it a sudden stop
       brakes at about 7 g and pitches the robot off its wheels. -->
  <actuator>
    <velocity name="motor_left" joint="wheel_left_joint" kv="0.5"
              ctrlrange="-6.67 6.67" forcelimited="true" forcerange="-0.06 0.06"/>
    <velocity name="motor_right" joint="wheel_right_joint" kv="0.5"
              ctrlrange="-6.67 6.67" forcelimited="true" forcerange="-0.06 0.06"/>
  </actuator>


  <sensor>
    <!-- Wheel encoders: the angle each wheel has turned, in radians. -->
    <jointpos name="enc_left" joint="wheel_left_joint"/>
    <jointpos name="enc_right" joint="wheel_right_joint"/>
    {lidar_sensors_xml}
  </sensor>
</mujoco>
"""