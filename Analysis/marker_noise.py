
# Analysis/marker_noise.py accuracy of marker detection 
#the EKF needs R, the noise of each range/bearing reading. Person 2 assumed 2 cm and
#2 deg. this measures it instead: the robot is placed at hundreds of known poses
#(distances 0.4-3.0 m, viewing angles 0-60 deg, the marker anywhere in the image),
#the camera detects the marker, and the reading is compared with the true range and
#bearing. this is a calibration experiment, so using the true pose here is fine.
#
#outputs to
#   Analysis/marker_noise_model.json   sigma_range = a + b r^2, sigma_bearing = a + b r
#   Analysis/marker_noise.png          error against distance and viewing angle
#   Analysis/marker_noise_samples.csv  every sample
#the paediatric ward (lit differently, no anti-aliasing) gets its own calibration, measured on
#the charging-dock marker: the same three files with "_paeds" added to the name
#
#run from the repo folder:  python Analysis/marker_noise.py

import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import frames as F
from Build_scene import SCENE_OUT, WARD, MARKER_BY_NAME, MISSION
from Sensing.driver import TurtleBotDriver
from Sensing.marker_detection import marker_range_bearing
from Control.nav_settings import MARKER_XY
from Control.planner import build_known_grid
from run_delivery import degrade_image

OUT = Path(__file__).resolve().parent
TAG = "" if WARD == "classic" else f"_{WARD}" #per ward output files
FACING = MARKER_BY_NAME[MISSION[-1]]["facing"] #which way the calibration marker faces
DISTANCES = np.arange(0.4, 3.01, 0.2)
VIEW_ANGLES_DEG = [0, 15, 30, 45, 60]
SAMPLES_EACH = 4


def collect():
    rng = np.random.default_rng(1)
    driver = TurtleBotDriver(SCENE_OUT)
    calib = (driver.K, driver.R_BC, driver.t_BC)
    free = build_known_grid().inflate(F.ROBOT_RADIUS)
    rows = []
    for r in DISTANCES:
        for a_deg in VIEW_ANGLES_DEG:
            for sign in (+1, -1) if a_deg else (+1,):
                for _ in range(SAMPLES_EACH):
                    #viewing angle = angle between the way the marker faces and the line to the robot.
                    #the classic marker faces -y (kept as written so its calibration reproduces exactly)
                    a = np.radians(a_deg) * sign
                    out_dir = (np.array([np.sin(a), -np.cos(a)]) if WARD == "classic"
                               else np.array([np.cos(FACING + a), np.sin(FACING + a)]))
                    p = MARKER_XY + r * out_dir + rng.uniform(-0.02, 0.02, 2)
                    if not free.is_free(*p):
                        continue
                    to_marker = np.arctan2(*(MARKER_XY - p)[::-1])
                    #point the robot so the marker lands anywhere in the image (up to 20 deg off centre)
                    theta = to_marker + rng.uniform(-np.radians(20), np.radians(20))
                    driver.reset((p[0], p[1], theta))
                    img = driver.read_camera()
                    true_r = float(np.hypot(*(MARKER_XY - p)))
                    true_b = float(F.wrap(to_marker - theta))
                    for kind, image in (("clean", img), ("degraded", degrade_image(img, rng))):
                        z = marker_range_bearing(image, calib=calib)
                        rows.append({"kind": kind, "range_true": true_r, "view_angle_deg": abs(a_deg),
                                     "bearing_true_deg": np.degrees(true_b), "detected": z is not None,
                                     "range_err": (z[0] - true_r) if z is not None else np.nan,
                                     "bearing_err_deg": np.degrees(F.wrap(z[1] - true_b)) if z is not None else np.nan})
    driver.close()
    return rows


def fit(rows):
    """
    two steps, like any sensor calibration
      1. bias is da range error is not zero-mean, it grows with range^2 (a constant pixel
         offset at the corners does exactly that through PnP). fit bias = c * r^2
      2. noise is what is left after removing the bias, RMS in distance bins, fitted as
         sigma = a + b r^2 (range) and a + b r (bearing), with small floors (2 mm, 0.1 deg)
    """
    clean = [r for r in rows if r["kind"] == "clean" and r["detected"]]
    rng_true = np.array([r["range_true"] for r in clean])
    e_r = np.array([r["range_err"] for r in clean])
    e_b = np.radians([r["bearing_err_deg"] for r in clean])

    bias_c = float(np.sum(e_r * rng_true ** 2) / np.sum(rng_true ** 4))   #least squares e = c r^2
    bearing_bias = float(np.mean(e_b))
    resid_r = e_r - bias_c * rng_true ** 2
    resid_b = e_b - bearing_bias

    edges = np.arange(0.3, 3.2, 0.2)
    centres, rms_r, rms_b = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (rng_true >= lo) & (rng_true < hi)
        if k.sum() >= 5:
            centres.append(rng_true[k].mean())
            rms_r.append(np.sqrt(np.mean(resid_r[k] ** 2)))
            rms_b.append(np.sqrt(np.mean(resid_b[k] ** 2)))
    centres, rms_r, rms_b = map(np.array, (centres, rms_r, rms_b))
    (ra, rb), *_ = np.linalg.lstsq(np.column_stack([np.ones_like(centres), centres ** 2]), rms_r, rcond=None)
    (ba, bb), *_ = np.linalg.lstsq(np.column_stack([np.ones_like(centres), centres]), rms_b, rcond=None)
    model = {"range_bias_c": bias_c, "bearing_bias": bearing_bias,
             "range_a": max(float(ra), 0.002), "range_b": max(float(rb), 0.0),
             "bearing_a": max(float(ba), np.radians(0.1)), "bearing_b": max(float(bb), 0.0),
             "note": "corrected range = range - range_bias_c * range^2, corrected bearing = bearing - bearing_bias; "
                     "then sigma_range = range_a + range_b * r^2 [m], sigma_bearing = bearing_a + bearing_b * r [rad]. "
                     "Fitted on clean simulated detections binned by true range"}
    return model, centres, rms_r, rms_b


def plot(rows, model, centres, rms_r, rms_b):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rr = np.linspace(0.3, 3.1, 100)
    sig_r = model["range_a"] + model["range_b"] * rr ** 2
    sig_b = model["bearing_a"] + model["bearing_b"] * rr

    #top left is the raw range error, the fitted bias, and the error after correction
    for kind, colour in (("clean", "#2a6fdb"), ("degraded", "#e8590c")):
        k = [r for r in rows if r["kind"] == kind and r["detected"]]
        x = np.array([r["range_true"] for r in k])
        e = np.array([r["range_err"] for r in k])
        axes[0, 0].scatter(x, 100 * e, s=6, alpha=0.35, color=colour, label=f"{kind}, raw")
        axes[0, 1].scatter(x, 100 * (e - model["range_bias_c"] * x ** 2), s=6, alpha=0.5, color=colour,
                           label=f"{kind}, bias removed")
    axes[0, 0].plot(rr, 100 * model["range_bias_c"] * rr ** 2, "k-", lw=1.2, label="fitted bias c·r²")
    axes[0, 0].set_title("Range error before calibration", fontsize=10)
    axes[0, 1].plot(rr, 100 * 3 * sig_r, "k--", lw=1, label="±3σ fitted")
    axes[0, 1].plot(rr, -100 * 3 * sig_r, "k--", lw=1)
    axes[0, 1].plot(rr, 100 * np.sqrt(0.02 ** 2 + (0.01 * rr) ** 2), ":", color="0.5", lw=1.5,
                    label="1σ assumed before")
    axes[0, 1].set_title("Range error after removing the bias", fontsize=10)
    for a in axes[0]:
        a.set_xlabel("true range (m)"); a.set_ylabel("range error (cm)"); a.grid(alpha=0.3); a.legend(fontsize=7)

    #bottom left is the residual range error by viewing angle, split by distance so the two arent mixed up
    clean = [r for r in rows if r["kind"] == "clean" and r["detected"]]
    bands = [(0.3, 1.0, "0.4-1.0 m"), (1.0, 2.0, "1.0-2.0 m"), (2.0, 3.1, "2.0-3.0 m")]
    width = 0.27
    for i, (lo, hi, label) in enumerate(bands):
        vals = []
        for a_deg in VIEW_ANGLES_DEG:
            k = [r for r in clean if r["view_angle_deg"] == a_deg and lo <= r["range_true"] < hi]
            res = [r["range_err"] - model["range_bias_c"] * r["range_true"] ** 2 for r in k]
            vals.append(1000 * np.sqrt(np.mean(np.square(res))) if len(res) >= 3 else np.nan)
        axes[1, 0].bar(np.arange(len(VIEW_ANGLES_DEG)) + (i - 1) * width, vals, width, label=label)
    axes[1, 0].set_xticks(range(len(VIEW_ANGLES_DEG)), [str(a) for a in VIEW_ANGLES_DEG])
    axes[1, 0].set_xlabel("viewing angle (deg, 0 = face-on)")
    axes[1, 0].set_ylabel("RMS range error after bias removal (mm)")
    axes[1, 0].set_title("Viewing angle, by distance band (gaps = never visible)", fontsize=10)
    axes[1, 0].legend(fontsize=8)

    #bottom right is the detection rate as a grid of distance x viewing angle.
    #blank = the robot cant stand there (inside a bed or wall); 0% = the beds block the view
    all_clean = [r for r in rows if r["kind"] == "clean"]
    grid = np.full((len(VIEW_ANGLES_DEG), len(DISTANCES)), np.nan)
    for i, a_deg in enumerate(VIEW_ANGLES_DEG):
        for j, c in enumerate(DISTANCES):
            k = [r["detected"] for r in all_clean if r["view_angle_deg"] == a_deg and abs(r["range_true"] - c) < 0.1]
            if k:
                grid[i, j] = 100 * np.mean(k)
    im = axes[1, 1].imshow(grid, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=100)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                axes[1, 1].text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=7,
                                color="w" if grid[i, j] > 60 else "k")
    axes[1, 1].set_xticks(range(len(DISTANCES)), [f"{c:.1f}" for c in DISTANCES], fontsize=7)
    axes[1, 1].set_yticks(range(len(VIEW_ANGLES_DEG)), [f"{a}°" for a in VIEW_ANGLES_DEG])
    axes[1, 1].set_xlabel("true range (m)"); axes[1, 1].set_ylabel("viewing angle")
    axes[1, 1].set_title("Detection rate (%) - blank = robot cant stand there, 0 = beds block the view",
                         fontsize=9)
    fig.suptitle("Marker detection error: measured and calibrated, against the 2 cm / 2° assumed before")
    fig.tight_layout()
    fig.savefig(OUT / f"marker_noise{TAG}.png", dpi=130)


if __name__ == "__main__":
    rows = collect()
    with open(OUT / f"marker_noise_samples{TAG}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    model, centres, rms_r, rms_b = fit(rows)
    (OUT / f"marker_noise_model{TAG}.json").write_text(json.dumps(model, indent=2))
    plot(rows, model, centres, rms_r, rms_b)
    det = np.mean([r["detected"] for r in rows if r["kind"] == "clean"]) * 100
    print(f"{sum(r['kind'] == 'clean' for r in rows)} poses, {det:.0f}% detected (clean)")
    for c, a, b in zip(centres, rms_r, rms_b):
        print(f"  range {c:4.2f} m: RMS range error after bias removal {1000 * a:5.1f} mm, bearing {np.degrees(b):5.3f} deg")
    print("fitted:", json.dumps({k: round(v, 5) for k, v in model.items() if k != "note"}))
    print(f"saved marker_noise_model{TAG}.json, marker_noise{TAG}.png, marker_noise_samples{TAG}.csv")
