"""Experiment 1: adversarial filtering penalises the systems used to build the benchmark.

Known-answer simulation (Rasch model with a system-by-item interaction). Outputs:
  figures/fig1_filtering.{png,pdf}     observed vs true accuracy, filter systems vs equal-ability twins
  figures/fig1b_interaction_dose.{png,pdf}  twin gap vs interaction strength (sd = 0 is the inert control)
  results/exp1_single_run.csv, results/exp1_dose_response.csv
"""
import numpy as np
import pandas as pd

from score_audit.design_checks import simulate_adversarial_filtering
from score_audit.plotting import BLUE, GRAY, RED, plt, results_dir, save

plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "legend.fontsize": 11})  # readable at half page width

N_SEEDS = 200
N_FILTER = 5


def twin_gap(r) -> np.ndarray:
    """Observed-score gap (filter system minus its equal-ability twin), in accuracy points."""
    return 100 * (r.obs_acc_kept[:N_FILTER] - r.obs_acc_kept[N_FILTER:2 * N_FILTER])


def main():
    out = results_dir()

    # single illustrative run
    r = simulate_adversarial_filtering(seed=0)
    role = ["filter"] * N_FILTER + ["twin"] * N_FILTER + ["other"] * (len(r.abilities) - 2 * N_FILTER)
    df = pd.DataFrame({"ability": r.abilities, "role": role, "acc_full_pool": r.true_acc_pool,
                       "acc_after_filtering": r.obs_acc_kept})
    df.to_csv(out / "exp1_single_run.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for i in range(N_FILTER):
        ax.plot([df.acc_full_pool[i], df.acc_full_pool[i + N_FILTER]],
                [df.acc_after_filtering[i], df.acc_after_filtering[i + N_FILTER]], color=GRAY, lw=0.8, zorder=1)
    for name, col, mk in (("filter", RED, "o"), ("twin", BLUE, "s"), ("other", GRAY, "^")):
        d = df[df.role == name]
        lab = {"filter": "used to filter items", "twin": "same ability, not used", "other": "other systems"}[name]
        ax.scatter(d.acc_full_pool, d.acc_after_filtering, c=col, marker=mk, s=38, label=lab, zorder=2)
    ax.set_xlabel("Accuracy on the full item pool (truth)")
    ax.set_ylabel("Accuracy on the filtered benchmark")
    ax.set_title(f"Filtering keeps {r.n_kept} of {r.n_pool} items")
    ax.legend(loc="upper left")
    save(fig, "fig1_filtering")

    # dose-response over the interaction strength, many seeds (sd = 0 is the inert control arm)
    rows = []
    for sd in (0.0, 0.5, 1.0, 1.5):
        gaps = np.concatenate([twin_gap(simulate_adversarial_filtering(interaction_sd=sd, seed=s))
                               for s in range(N_SEEDS)])
        lo, hi = np.quantile(gaps, [0.025, 0.975])
        rows.append({"interaction_sd": sd, "mean_gap_points": gaps.mean(), "se": gaps.std(ddof=1) / np.sqrt(len(gaps)),
                     "q2.5": lo, "q97.5": hi, "share_filter_below_twin": float((gaps < 0).mean())})
    dose = pd.DataFrame(rows)
    dose.to_csv(out / "exp1_dose_response.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.errorbar(dose.interaction_sd, dose.mean_gap_points, yerr=1.96 * dose.se, color=RED, marker="o", capsize=3)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("Interaction sd (0 = inert control)")
    ax.set_ylabel("Filter minus twin (points)")
    ax.set_title("Penalty grows with interaction")
    save(fig, "fig1b_interaction_dose")
    print(df.round(3).to_string(index=False))
    print(dose.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
