
#EKF consistency marked against responsiveness
#the encoder model leaves things out (wheel slip, rocking, linearisation error), so with no extra process noise the EKF gets overconfident once the marker is seen.
#this sweeps the extra process noise and measures:
#consistency is tje mean NEES after the marker is seen (ideal 3 for 3 states)
#accuracy is the EKF position error after the marker is seen, and docking error
#responsiveness is more process noise = trusts the camera more, jumps more
#run from the repo folder:  python Analysis/consistency_sweep.py

import csv
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

#this script belongs to the classic ward so not the paediatric one
from Build_scene import WARD as _WARD
if _WARD != "classic" and __name__ == "__main__":
    raise SystemExit("consistency_sweep.py produces the classic-ward figures in RESULTS.md: run it with --ward classic")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_delivery import run_delivery, TrialConditions

OUT = Path(__file__).resolve().parent
SCALES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
SEEDS = [0, 1, 2]


def one(args):
    scale, seed, r_model = args
    r = run_delivery(TrialConditions(name=f"pn{scale}", seed=seed, process_noise=scale, r_model=r_model),
                     verbose=False)[0]
    return {k: r[k] for k in ("process_noise", "seed", "r_model", "mean_nees", "mean_nees_after_marker",
                              "mean_nis", "ekf_rmse_pos_m", "ekf_final_error_m", "dock_error_m",
                              "dock_heading_deg", "success")}


if __name__ == "__main__":
    jobs = [(s, seed, rm) for rm in ("fitted", "fixed") for s in SCALES for seed in SEEDS]
    with Pool() as pool:
        rows = pool.map(one, jobs)
    with open(OUT / "consistency_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for rm, colour in (("fitted", "#2a6fdb"), ("fixed", "#e8590c")):
        nees = [np.mean([r["mean_nees_after_marker"] for r in rows if r["r_model"] == rm and r["process_noise"] == s])
                for s in SCALES]
        nis = [np.mean([r["mean_nis"] for r in rows if r["r_model"] == rm and r["process_noise"] == s]) for s in SCALES]
        dock = [np.mean([r["dock_error_m"] for r in rows if r["r_model"] == rm and r["process_noise"] == s]) * 100
                for s in SCALES]
        x = [max(s, 0.25) for s in SCALES]
        axes[0].plot(x, nees, "o-", color=colour, label=f"NEES after marker, R {rm}")
        axes[0].plot(x, nis, "s--", color=colour, alpha=0.6, label=f"NIS, R {rm}")
        axes[1].plot(x, dock, "o-", color=colour, label=f"R {rm}")
    axes[0].axhline(3, color="k", lw=0.8, ls=":"); axes[0].text(0.26, 3.2, "ideal NEES = 3", fontsize=8)
    axes[0].axhline(2, color="0.4", lw=0.8, ls=":"); axes[0].text(0.26, 1.6, "ideal NIS = 2", fontsize=8)
    axes[0].set_yscale("log"); axes[0].set_ylabel("mean statistic (log)")
    axes[1].set_ylabel("docking error (cm), mean of seeds")
    for a in axes:
        a.set_xscale("log"); a.set_xlabel("extra process noise scale (0 plotted at 0.25)")
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
    fig.suptitle(f"EKF consistency against accuracy ({len(SEEDS)} seeds per point)")
    fig.tight_layout(); fig.savefig(OUT / "consistency_sweep.png", dpi=130)
    for rm in ("fitted", "fixed"):
        for s in SCALES:
            k = [r for r in rows if r["r_model"] == rm and r["process_noise"] == s]
            print(f"R {rm:6s} scale {s:4.1f}: NEES(after marker) {np.mean([r['mean_nees_after_marker'] for r in k]):6.2f} "
                  f"NIS {np.mean([r['mean_nis'] for r in k]):5.2f}  EKF rmse {100*np.mean([r['ekf_rmse_pos_m'] for r in k]):4.2f} cm  "
                  f"dock {100*np.mean([r['dock_error_m'] for r in k]):4.2f} cm  success {sum(r['success'] for r in k)}/{len(k)}")
