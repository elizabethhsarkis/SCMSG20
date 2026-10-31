
#repeated trials under every robustness condition (Part E)

#every condition in Analysis/conditions.py is run from the same set of random start
#poses (so conditions are compared like for like). each trial is scored, failures get a
#cause traced to sensing, estimation, planning or control, and a plot is saved for each
#failure so it can be checked by eye.
#
#a trial is successful if it reaches the dock, ends within 5 cm and 5 deg of the dock pose and never touches an obstacle.
#
#run from the repo folder:
#   python Analysis/run_trials.py                       #all conditions, 10 trials each
#   python Analysis/run_trials.py --trials 3            #quick check
#   python Analysis/run_trials.py --conditions nominal push
#
#output (Analysis/trials/): trials.csv (every trial), summary.csv, summary.md,
#trials_summary.png, fail_<condition>_<trial>.png

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_delivery import run_delivery, plot_delivery, TrialConditions
from Analysis.conditions import CONDITIONS
from Build_scene import DOCK_POSITION, START_REGION, WARD

#each ward keeps its own results: the classic ward's are the ones in RESULTS.md
OUT = Path(__file__).resolve().parent / ("trials" if WARD == "classic" else f"trials_{WARD}")


def start_pose(trial):
    #random start near the pharmacy, any heading. same pose for trial i in every condition.
    #the region comes from the ward (Build_scene.START_REGION)
    (x0, x1), (y0, y1) = START_REGION
    rng = np.random.default_rng(500 + trial)
    return (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)), float(rng.uniform(-np.pi, np.pi)))


def diagnose(r):
    """trace a failure to the part of the system that caused it"""
    if r["success"]:
        return ""
    if r["reason"] == "planner failed":
        return f"planning: {r['error']}"
    if r["collision_ticks"] > 0:
        e = 100 * r["collision_ekf_error_m"]
        if e > 10:
            return f"estimation: hit {r['collision_with']} while the EKF was {e:.0f} cm off"
        return (f"planning/control: hit {r['collision_with']} with the EKF only {e:.0f} cm off "
                f"(route too close, or tracking error {100 * r['rms_cross_track_true_m']:.1f} cm RMS)")
    if r["reason"] == "timeout":
        return "control: never reached the dock (timeout)"
    if (r.get("stops_total", 1) > 1 and r["reason"] == "reached" and r["dock_error_m"] < 0.05
            and abs(r["dock_heading_deg"]) < 5):
        return (f"stop accuracy: docked fine but a bedside stop missed 5 cm / 5 deg "
                f"(stop errors {r['stop_errors_cm']} cm, worst heading {r['worst_stop_heading_deg']:.1f} deg)")
    d, e = 100 * r["dock_error_m"], 100 * r["ekf_final_error_m"]
    if r["dock_error_m"] >= 0.05:
        if r["marker_updates"] == 0:
            if r["marker_hidden_within_m"] > 0:
                return f"sensing: marker blocked the whole way, docked on odometry alone ({d:.1f} cm off)"
            return (f"estimation: robot lost before the marker came into view (EKF {e:.0f} cm off), so the "
                    f"camera never faced it; nothing else could reveal the error")
        if e >= 0.5 * d and r["marker_rejected"] > 0 and r["outliers_injected"] == 0:
            return (f"estimation: the gate rejected {r['marker_rejected']} GENUINE marker readings after an "
                    f"unmodelled disturbance (EKF lost and overconfident), ended {e:.1f} cm off")
        if e >= 0.5 * d:
            return (f"estimation: EKF {e:.1f} cm off at the end (last marker reading "
                    f"{r['time_s'] - r['last_marker_s']:.1f} s before the end, {r['marker_rejected']} rejected)")
        if r["outliers_injected"] > 0 and not r["gating"]:
            return (f"estimation: an unrejected bad marker reading jumped the estimate and the robot stopped "
                    f"on it ({d:.1f} cm off; the EKF recovered to {e:.1f} cm afterwards)")
        return f"control: estimate within {e:.1f} cm but stopped {d:.1f} cm from the dock"
    #rotation about the marker is the one direction a single marker cant observe (observability.py).
    #docked rotated by angle a about the marker heading off by a, sideways offset -DOCK_POSITION * a
    a = np.radians(r["dock_heading_deg"])
    if abs(r["dock_sideways_m"] - (-DOCK_POSITION * a)) < 0.015:
        return (f"observability limit: docked rotated {r['dock_heading_deg']:+.1f} deg about the marker "
                f"(sideways {100 * r['dock_sideways_m']:+.1f} cm) - the one direction one marker cannot correct")
    if r["outliers_injected"] > 0 and not r["gating"]:
        return f"estimation: unrejected bad marker readings left the heading {r['dock_heading_deg']:+.1f} deg off"
    return f"control/estimation: heading {r['dock_heading_deg']:+.1f} deg off"


def one_trial(job):
    name, trial = job
    cond = TrialConditions(**{**CONDITIONS[name], "name": name, "seed": trial, "start_pose": start_pose(trial)})
    t0 = time.time()
    out = run_delivery(cond, verbose=False)
    r = out[0]
    r["trial"] = trial
    r["cause"] = diagnose(r)
    r["wall_time_s"] = time.time() - t0
    if not r["success"]:
        plot_delivery(*out, out_file=str(OUT / f"fail_{name}_{trial}.png"))
    return r


def load_rows(path):
    #read trials.csv back, turning numbers and True/False back into python values
    def parse(v):
        if v in ("True", "False"):
            return v == "True"
        try:
            return float(v) if any(ch in v for ch in ".eEn") else int(v)
        except ValueError:
            return v
    with open(path) as f:
        rows = [{k: parse(v) for k, v in r.items()} for r in csv.DictReader(f)]
    for r in rows:
        r["name"], r["cause"], r["reason"] = str(r["name"]), str(r["cause"]), str(r["reason"])
        r["collision_with"] = str(r["collision_with"])
    return rows


def summarise(rows, names):
    summary = []
    for name in names:
        k = [r for r in rows if r["name"] == name]
        dock = np.array([r["dock_error_m"] for r in k if r["reason"] == "reached"]) * 100
        summary.append({
            "condition": name,
            "trials": len(k),
            "success_rate_pct": 100 * np.mean([r["success"] for r in k]),
            "dock_error_mean_cm": float(np.mean(dock)) if len(dock) else np.nan,
            "dock_error_p95_cm": float(np.percentile(dock, 95)) if len(dock) else np.nan,
            "heading_error_mean_abs_deg": float(np.mean([abs(r["dock_heading_deg"]) for r in k])),
            "ekf_rmse_cm": 100 * float(np.mean([r["ekf_rmse_pos_m"] for r in k])),
            "mean_nees_after_marker": float(np.nanmean([r["mean_nees_after_marker"] for r in k])),
            "mean_nis": float(np.nanmean([r["mean_nis"] for r in k])),
            "marker_rejected": float(np.mean([r["marker_rejected"] for r in k])),
            "gate_recoveries": float(np.mean([r["gate_recoveries"] for r in k])),
            "replans": float(np.mean([r["replans"] for r in k])),
            "collisions": int(sum(r["collision_ticks"] > 0 for r in k)),
            "time_s": float(np.mean([r["time_s"] for r in k])),
            "failure_causes": "; ".join(f"#{r['trial']}: {r['cause']}" for r in k if r["cause"]),
        })
    return summary


def plot_summary(rows, summary, names):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1.3, 1.3]})
    y = np.arange(len(names))[::-1]
    rate = [s["success_rate_pct"] for s in summary]
    axes[0].barh(y, rate, color=["#2b8a3e" if v == 100 else "#e8590c" if v >= 80 else "#c92a2a" for v in rate])
    for yy, v, s in zip(y, rate, summary):
        axes[0].text(min(v, 88) + 1, yy, f"{v:.0f}%", va="center", fontsize=8)
    axes[0].set_yticks(y, names); axes[0].set_xlim(0, 100); axes[0].set_xlabel("success rate (%)")
    axes[0].set_title("Success (≤5 cm, ≤5°, no collision)", fontsize=10)

    for i, name in enumerate(names):
        k = [r for r in rows if r["name"] == name and r["reason"] == "reached"]
        d = 100 * np.array([r["dock_error_m"] for r in k])
        e = 100 * np.array([r["ekf_rmse_pos_m"] for r in k])
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(d))
        ok = np.array([r["success"] for r in k], bool)
        axes[1].scatter(d[ok], y[i] + jitter[ok], s=14, color="#2a6fdb", alpha=0.7)
        axes[1].scatter(d[~ok], y[i] + jitter[~ok], s=24, color="#c92a2a", marker="x")
        axes[2].scatter(e, y[i] + jitter, s=14, color="#2a6fdb", alpha=0.7)
    axes[1].axvline(5, color="#c92a2a", ls=":", lw=1); axes[1].text(5.1, y[0] + 0.3, "5 cm limit", fontsize=8,
                                                                     color="#c92a2a")
    axes[1].set_xscale("log"); axes[1].set_xlabel("docking error (cm, log scale)")
    axes[1].set_title("Docking error per trial (× = failed)", fontsize=10)
    axes[2].set_xscale("log"); axes[2].set_xlabel("EKF position RMSE over the run (cm, log scale)")
    axes[2].set_title("Estimation error per trial", fontsize=10)
    for a in axes[1:]:
        a.set_yticks(y, ["" for _ in names]); a.grid(alpha=0.3, which="both", axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "trials_summary.png", dpi=130)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--summary-only", action="store_true",
                        help="redo the diagnosis, summary and plots from trials.csv without running anything")
    parser.add_argument("--append", action="store_true",
                        help="keep earlier results for conditions not being rerun (from trials.csv)")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)

    jobs = [] if args.summary_only else [(name, t) for name in args.conditions for t in range(args.trials)]
    print(f"{len(jobs)} trials ({len(args.conditions)} conditions x {args.trials})", flush=True)
    rows = []
    if args.summary_only:
        args.append, args.conditions = True, []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(one_trial, jobs):
            rows.append(r)
            print(f"  [{len(rows):3d}/{len(jobs)}] {r['name']:18s} #{r['trial']:<2d} "
                  f"{'ok  ' if r['success'] else 'FAIL'} dock {100 * r['dock_error_m']:6.1f} cm  {r['cause']}",
                  flush=True)
    if args.append and (OUT / "trials.csv").exists():
        rows += [r for r in load_rows(OUT / "trials.csv") if r["name"] not in args.conditions]
    for r in rows:
        r["cause"] = diagnose(r)
    order = [c for c in CONDITIONS if c in {r["name"] for r in rows}]
    args.conditions = order
    rows.sort(key=lambda r: (order.index(r["name"]), r["trial"]))
    keys = list(rows[0])
    with open(OUT / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summary = summarise(rows, args.conditions)
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    lines = ["| Condition | Success | Dock error mean (cm) | Dock error 95th (cm) | EKF RMSE (cm) | "
             "NEES after marker | NIS | Rejected | Replans |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in summary:
        lines.append(f"| {s['condition']} | {s['success_rate_pct']:.0f}% | {s['dock_error_mean_cm']:.2f} | "
                     f"{s['dock_error_p95_cm']:.2f} | {s['ekf_rmse_cm']:.2f} | {s['mean_nees_after_marker']:.2f} | "
                     f"{s['mean_nis']:.2f} | {s['marker_rejected']:.1f} | {s['replans']:.1f} |")
    lines += ["", "Failures and their causes:", ""]
    lines += [f"- **{s['condition']}** {s['failure_causes']}" for s in summary if s["failure_causes"]] or ["- none"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    plot_summary(rows, summary, args.conditions)
    print("\n".join(lines))
    print(f"\nsaved {OUT}/trials.csv, summary.csv, summary.md, trials_summary.png")
