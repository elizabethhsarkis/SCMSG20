
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

import frames as F
from Build_scene import MARKER_ID, MARKER_SIDE      # id 7, 0.12 m black square

_DETECTOR = None


def _get_detector():                        # Build ArUco detector
    global _DETECTOR
    if _DETECTOR is None:
        aruco = cv2.aruco
        dictionary = aruco.getPredefinedDictionary(getattr(aruco, F.ARUCO_DICT_NAME))
        if hasattr(aruco, "ArucoDetector"):
            params = aruco.DetectorParameters()
            params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            det = aruco.ArucoDetector(dictionary, params)
            _DETECTOR = det.detectMarkers
        else:
            params = aruco.DetectorParameters_create()
            params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            _DETECTOR = lambda g: aruco.detectMarkers(g, dictionary, parameters=params)
        # Sub-pixel corner refinement matters: the range error from PnP scales with the corner pixel error times range^2 / (f * size)
        # main mechanism behind the HD noise-versus-distance study
    return _DETECTOR


def _marker_corners_3d(size):
    # returns 3D corners of the black square in marker frame M (z out of face)
    # order: top left, top right, bottom right, bottom left
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def marker_range_bearing(img, marker_id=MARKER_ID, marker_size=MARKER_SIDE,
                         calib=None, noise_std=None, rng=None, return_details=False):
    # detects the bedside marker and returns range/bearing from base frame B

    # Parameters:
    # img = RGB image from driver.read_camera()
    # calib = default readings from frames.py (K, R_BC, t_BC)
    # noise_std = optional (sigma_range, sigma_bearing) of extra Gaussian noise
    # return_details = also returns viewing angle, size in pixels ....

    K, R_BC, t_BC = calib if calib is not None else F.get_camera_calibration()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _get_detector()(gray)
    if ids is None or marker_id not in ids.ravel():
        return (None, None) if return_details else None
    c = corners[list(ids.ravel()).index(marker_id)].reshape(4, 2).astype(np.float64)

    # PnP. There is no distortion in simulation; on the real robot, pass
    #    the calibrated distortion coefficients instead of None.
    ok, rvec, tvec = cv2.solvePnP(_marker_corners_3d(marker_size), c, K, None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return (None, None) if return_details else None

    # Frame chain: marker centre in Ccv -> C -> B.
    #    For a level camera this reduces to View_scene.py's
    #    x_B = z_cv + 0.0855 and y_B = -x_cv (the transforms are identical),
    #    but the general form still holds if the camera is ever tilted.
    p_Ccv = tvec.ravel()
    p_C = F.R_C_CCV @ p_Ccv
    p_B = R_BC @ p_C + t_BC

    # Planar range and bearing from B (counter-clockwise positive, same as the EKF)
    z = np.array([np.hypot(p_B[0], p_B[1]), np.arctan2(p_B[1], p_B[0])])

    if noise_std is not None:
        rng = np.random.default_rng() if rng is None else rng
        z = z + rng.normal(0.0, noise_std)
        z[1] = F.wrap(z[1])

    if not return_details:
        return z

    # Extra outputs (marker noise against distance and viewing angle).
    # Viewing angle = angle between the marker's normal (z axis, out of face) and the line of sight back to the camera; 0 = face-on.
    # IPPE has a two-fold rotation ambiguity at small marker sizes: it affects the angle but not the translation, so z is unaffected.
    R_CcvM, _ = cv2.Rodrigues(rvec)
    normal = R_CcvM[:, 2]
    los = p_Ccv / np.linalg.norm(p_Ccv)
    details = {
        "corners_px": c,
        "p_Ccv": p_Ccv,
        "p_B": p_B,
        "viewing_angle": float(np.arccos(np.clip(-normal @ los, -1.0, 1.0))),
        "side_px": float(np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1))),
    }
    return z, details
