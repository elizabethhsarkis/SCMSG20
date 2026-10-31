
# Control/nav_settings.py  the navigation settings for my part

#everything the planner, ghost robot and LQR tracker need that isnt already in
#frames.py or Build_scene.py. robot sizes and the marker come from those files so
#theres still only one place each number lives (stage 0 agreement)

import numpy as np

import frames as F
from Build_scene import MARKER_X, MARKER_Y, DOCK_POSITION, MARKERS, MISSION, MARKER_BY_NAME

#where the robot starts
PHARMACY_POSE = np.array(F.START_POSE, dtype=float)


def park_pose(marker):
    #DOCK_POSITION (0.40 m) straight out from the marker's face, turned round to face it.
    #e.g. a marker facing south (-pi/2) -> park 0.40 m south of it, facing north (+pi/2)
    f = marker["facing"]
    heading = f + np.pi
    if heading > np.pi:
        heading -= 2 * np.pi
    return np.array([marker["x"] + DOCK_POSITION * np.cos(f),
                     marker["y"] + DOCK_POSITION * np.sin(f), heading])


#every marker the EKF can use ArUco id world position of its face centre, and the way it faces
MARKER_POSITIONS = {m["id"]: np.array([m["x"], m["y"]]) for m in MARKERS}
MARKER_FACING = {m["id"]: m["facing"] for m in MARKERS}

#the round, in order: each stop is a marker to park in front of. the classic ward has one stop
#(the bedside), the paediatric ward has four bedsides then the charging dock
STOPS = [{"name": name, "label": MARKER_BY_NAME[name]["label"], "marker_id": MARKER_BY_NAME[name]["id"],
          "marker_xy": MARKER_POSITIONS[MARKER_BY_NAME[name]["id"]], "park": park_pose(MARKER_BY_NAME[name])}
         for name in MISSION]

#the FINAL stop keeps its original names (single-marker code and the classic analysis use these)
MARKER_XY = np.array([MARKER_X, MARKER_Y])
BEDSIDE_PARK_POSE = STOPS[-1]["park"]

#waffle pi limits, worked out from frames.py
MAX_V = F.MAX_WHEEL_SPEED * F.WHEEL_RADIUS   #7.88 rad/s * 0.033 m = 0.26 m/s
MAX_W = MAX_V / (F.WHEEL_BASE / 2)  #spinning on the spot, about 1.8 rad/s

#extra clearance the planner keeps on top of ROBOT_RADIUS (0.22 m), because the
#robot never follows the plan perfectly
SAFETY_MARGIN = 0.08

#the control loop decide every 20 ms (50 Hz), the same as the EKF prediction rate
DT = 0.02
PHYSICS_STEPS_PER_CONTROL = 10                          #10 x 2 ms mujoco steps

#how often the other sensors are read, as multiples of the control step
LIDAR_EVERY = 10 #5 Hz, the LDS-01 spin rate
CAMERA_EVERY = 5  #10 Hz

#EKF noise settings, same values Person 2 tuned in run_ekf_mujoco.py
EKF_P0 = np.diag([0.01 ** 2, 0.01 ** 2, np.deg2rad(1) ** 2]) #1 cm, 1 deg at the start pad
EKF_K_ENC = 1e-4 #matches the driver's encoder noise

#extra process noise added every step: variance per second for x, y, theta at scale 1
#(2 mm and 0.2 deg per sqrt(second)). the scale is picked by Analysis/consistency_sweep.py
PROCESS_NOISE_BASE = np.diag([0.002 ** 2, 0.002 ** 2, np.deg2rad(0.2) ** 2])
PROCESS_NOISE_SCALE = 2.0                               #NEES after the marker ~2.4 (ideal 3), see consistency_sweep.png


#chi-squared gate for marker updates: reject a detection whose innovation is less
#likely than 1 in 100 for 2 degrees of freedom (NIS > 9.21)
NIS_GATE = 9.21

#gate recovery this many rejections in a row means the EKF is lost, not the camera.
#two outliers in a row at a 10% outlier rate is a 1% event. the covariance is then widened
#by 5 cm and 10 deg so the next reading can pull the estimate back
GATE_RECOVERY_AFTER = 2
GATE_RECOVERY_P = np.diag([0.05 ** 2, 0.05 ** 2, np.deg2rad(10) ** 2])

#fitted marker noise model, written by Analysis/marker_noise.py
#each ward has its own (the camera sees a differently lit room): Analysis/marker_noise.py --ward ...
from Build_scene import WARD
MARKER_NOISE_FILE = F.ROOT / "Analysis" / ("marker_noise_model.json" if WARD == "classic"
                                           else f"marker_noise_model_{WARD}.json")


def marker_noise_fixed(range_m):
    #Assumed noise is 2 cm + 1% of range, 2 deg. further readings trusted less
    return np.diag([0.02 ** 2 + (0.01 * range_m) ** 2, np.deg2rad(2) ** 2])


def marker_noise_fitted(range_m):
    #measured noise: sigma = a + b * range^2 for range (PnP error grows with range^2),
    #and a + b * range for bearing. falls back to the fixed model if not measured yet
    fit = _load_noise_fit()
    if fit is None:
        return marker_noise_fixed(range_m)
    sigma_r = fit["range_a"] + fit["range_b"] * range_m ** 2
    sigma_b = fit["bearing_a"] + fit["bearing_b"] * range_m
    return np.diag([sigma_r ** 2, sigma_b ** 2])


_NOISE_FIT_CACHE = {}


def _load_noise_fit():
    #read the json once and keep it
    import json
    if "fit" not in _NOISE_FIT_CACHE:
        _NOISE_FIT_CACHE["fit"] = (json.loads(MARKER_NOISE_FILE.read_text())
                                   if MARKER_NOISE_FILE.exists() else None)
    return _NOISE_FIT_CACHE["fit"]


def correct_marker_reading(z, model="fitted"):
    #remove the measured detection bias (range grows like range^2), only with the fitted model
    fit = _load_noise_fit() if model == "fitted" else None
    if fit is None:
        return np.asarray(z, dtype=float)
    r, b = float(z[0]), float(z[1])
    return np.array([r - fit["range_bias_c"] * r ** 2, F.wrap(b - fit["bearing_bias"])])


def marker_noise(range_m, model="fitted"):
    return marker_noise_fitted(range_m) if model == "fitted" else marker_noise_fixed(range_m)
