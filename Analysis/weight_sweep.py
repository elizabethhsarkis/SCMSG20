
# Analysis/weight_sweep.py  LQR tracking error against control effort (Part D)
#the LQR weights come from Bryson's rule (Control/lqr_tracker.py). this checks that
#choice on the full system: both Q matrices are scaled from 0.1x to 10x (R fixed) and
#each setting is run from several seeds. bigger Q = care more about error, use more effort.
#
#tracking error is measured two ways:
#   against the ESTIMATE (what the controller sees and can fix)
#   against the TRUTH    (what actually happens, includes the EKF's error)
#
#run from the repo folder:  python Analysis/weight_sweep.py

import csv
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

#this script belongs to the classic ward. it stops if the paediatric ward is selected
from Build_scene import WARD as _WARD
if _WARD != "classic" and __name__ == "__main__":
    raise SystemExit("weight_sweep.py produces the classic-ward figures in RESULTS.md: run it with --ward classic")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_delivery import run_delivery, TrialConditions

OUT = Path(__file__).resolve().parent
SCALES = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
SEEDS = [0, 1, 2]


def one(job):
    q, seed = job
    r = run_delivery(TrialConditions(name=f"q{q}", seed=seed, q_scale=q), verbose=False)[0]
    return {k: r[k] for k in ("q_scale", "seed", "rms_cross_track_est_m", "rms_cross_track_true_m",
                              "rms_effort", "dock_error_m", "dock_heading_deg", "time_s", "success",
                              "collision_ticks")}


if __name__ == "__main__":
    with Pool() as pool:
        rows = pool.map(one, [(q, s) for q in SCALES for s in SEEDS])
    with open(OUT / "weight_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def stat(key, q, scale=1.0):
        v = np.array([r[key] for r in rows if r["q_scale"] == q]) * scale
        return v.mean(), v.std()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for key, colour, label in (("rms_cross_track_est_m", "#2a6fdb", "against the estimate (controller's view)"),
                               ("rms_cross_track_true_m", "#e8590c", "against the truth")):
        e = [stat(key, q, 100) for q in SCALES]
        u = [stat("rms_effort", q) for q in SCALES]
        axes[0].errorbar([a for a, _ in u], [a for a, _ in e], xerr=[b for _, b in u], yerr=[b for _, b in e],
                         fmt="o-", color=colour, capsize=3, label=label)
        if key == "rms_cross_track_est_m":
            for q, (ux, _), (ey, _) in zip(SCALES, u, e):
                axes[0].annotate(f"Q×{q:g}", (ux, ey), textcoords="offset points", xytext=(6, 5), fontsize=8)
    axes[0].set_xlabel("control effort: RMS LQR correction |u|")
    axes[0].set_ylabel("RMS cross-track error (cm)")
    axes[0].set_title(f"Tracking error against effort ({len(SEEDS)} runs per point, ±1 sd)", fontsize=10)
    axes[0].set_yscale("log"); axes[0].grid(alpha=0.3, which="both"); axes[0].legend(fontsize=8)

    d = [stat("dock_error_m", q, 100) for q in SCALES]
    h = [stat("dock_heading_deg", q) for q in SCALES]
    axes[1].errorbar(SCALES, [a for a, _ in d], yerr=[b for _, b in d], fmt="o-", color="#2a6fdb", capsize=3,
                     label="docking error (cm)")
    ax2 = axes[1].twinx()
    ax2.errorbar(SCALES, [abs(a) for a, _ in h], yerr=[b for _, b in h], fmt="s--", color="#2b8a3e", capsize=3,
                 label="|heading error| (deg)")
    axes[1].set_xscale("log"); axes[1].set_xlabel("Q scale (1 = Bryson's rule)")
    axes[1].set_ylabel("docking error (cm)"); ax2.set_ylabel("|heading error| (deg)")
    axes[1].axvline(1, color="0.5", ls=":", lw=1)
    axes[1].set_title("Effect on the final docking", fontsize=10)
    axes[1].legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(OUT / "weight_sweep.png", dpi=130)

    for q in SCALES:
        k = [r for r in rows if r["q_scale"] == q]
        print(f"Q x {q:<5g} cross-track est {100 * stat('rms_cross_track_est_m', q)[0]:5.2f} cm, "
              f"true {100 * stat('rms_cross_track_true_m', q)[0]:5.2f} cm, effort {stat('rms_effort', q)[0]:.3f}, "
              f"dock {stat('dock_error_m', q, 100)[0]:.2f} cm, time {stat('time_s', q)[0]:.1f} s, "
              f"success {sum(r['success'] for r in k)}/{len(k)}, collisions {sum(r['collision_ticks'] > 0 for r in k)}")
    print("saved Analysis/weight_sweep.csv and weight_sweep.png")
