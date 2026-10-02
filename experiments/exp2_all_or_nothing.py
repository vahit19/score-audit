"""Experiment 2: all-or-nothing composites hide real differences between systems.

Each problem has k = 3 subtasks; a problem counts only if all three pass. Outputs:
  figures/fig2_composite_floor.{png,pdf}  expected score vs per-subtask skill for k = 1, 2, 3
  figures/fig2b_power.{png,pdf}           power to detect a 5-point subtask gap, all-or-nothing vs partial credit
  results/exp2_power.csv
"""
import numpy as np
import pandas as pd

from score_audit.design_checks import composite_score, power_two_systems
from score_audit.plotting import BLUE, GRAY, GREEN, RED, plt, results_dir, save

P_A, P_B = 0.40, 0.35          # per-subtask success of two systems (a real 5-point gap)
N_GRID = (50, 100, 200, 400, 800, 1600)
N_SIM = 1500


def main():
    out = results_dir()

    p = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for k, col in ((1, GRAY), (2, BLUE), (3, RED)):
        ax.plot(100 * p, 100 * composite_score(p, k), color=col, label=f"{k} subtask{'s' if k > 1 else ''} must all pass")
    ax.axhline(2.0, color=GREEN, ls="--", lw=0.9)
    ax.text(2, 4, "a 2% headline score", color=GREEN, fontsize=8.5)
    ax.set_xlabel("Per-subtask success rate (%)")
    ax.set_ylabel("Benchmark score (%)")
    ax.set_title("A near-zero score can hide ~27% subtask skill")
    ax.legend(loc="upper left")
    save(fig, "fig2_composite_floor")

    rows = []
    for n in N_GRID:
        for aon in (True, False):
            rows.append({"n_problems": n, "scoring": "all-or-nothing" if aon else "partial credit",
                         "power": power_two_systems(P_A, P_B, n, all_or_nothing=aon, n_sim=N_SIM, seed=n)})
    pw = pd.DataFrame(rows)
    pw.to_csv(out / "exp2_power.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for name, col in (("all-or-nothing", RED), ("partial credit", BLUE)):
        d = pw[pw.scoring == name]
        ax.plot(d.n_problems, 100 * d.power, marker="o", color=col, label=name)
    ax.axhline(80, color=GRAY, ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(N_GRID, [str(n) for n in N_GRID])
    ax.set_xlabel("Number of problems (log scale)")
    ax.set_ylabel("Power to detect the better system (%)")
    ax.set_title(f"Same systems ({int(P_A*100)}% vs {int(P_B*100)}% per subtask)")
    ax.legend(loc="lower right")
    save(fig, "fig2b_power")
    print("composite score at p=0.27:", round(100 * composite_score(np.array(0.27)), 2), "%")
    print(pw.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
