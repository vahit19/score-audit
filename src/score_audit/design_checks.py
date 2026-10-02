"""Known-answer simulations of two common benchmark-design choices.

1. Adversarial filtering: keep only the items that a set of "filter" models could not solve reliably.
2. All-or-nothing composites: an item counts as solved only if every one of its subtasks is solved.

Both are simulated under a Rasch (1PL item-response) model, so the true ability of every system is known
and the distortion introduced by the design choice can be measured directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- adversarial filtering
@dataclass
class FilterSim:
    abilities: np.ndarray        # (M,) true ability of each evaluated system
    is_filter: np.ndarray        # (M,) True if the system was used to filter items
    true_acc_pool: np.ndarray    # (M,) expected accuracy on the unfiltered item pool
    obs_acc_kept: np.ndarray     # (M,) observed accuracy on the kept (filtered) items
    n_pool: int
    n_kept: int


def simulate_adversarial_filtering(
    n_pool: int = 3000,
    n_filter: int = 5,
    attempts: int = 3,
    solved_if: int = 2,
    interaction_sd: float = 1.0,
    seed: int = 0,
) -> FilterSim:
    """Filter models and their equal-ability 'twins' (never used for filtering) are evaluated on the kept items.

    Success probability is sigmoid(ability - difficulty + u), where u is a persistent system-by-item
    interaction (some systems are simply better at some items). An item is discarded if ANY filter model
    solves it in at least `solved_if` of `attempts` construction tries. Evaluation is one fresh attempt per
    (system, item). Because filtering removes the items where a filter model's u happened to be high, the
    filter models are evaluated on their own weak spots, while their twins (same ability, independent u) are not.
    """
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(1.5, 1.2, n_pool)                    # hard pool: frontier-style benchmark
    filt = np.linspace(0.6, 1.8, n_filter)                       # strongest systems at construction time
    others = np.array([-0.5, 0.0, 0.4, 2.2, 2.6])                # weaker systems and later, stronger systems
    abilities = np.concatenate([filt, filt, others])             # filter models, their twins, others
    is_filter = np.array([True] * n_filter + [False] * (n_filter + len(others)))

    u = rng.normal(0.0, interaction_sd, (len(abilities), n_pool))
    p_all = sigmoid(abilities[:, None] - difficulty[None, :] + u)

    # construction: repeated attempts by the filter models only (rows 0..n_filter-1)
    wins = rng.binomial(attempts, p_all[:n_filter])
    keep = ~(wins >= solved_if).any(axis=0)

    # evaluation: one fresh attempt per (system, item) on the kept items
    eval_draw = rng.random(p_all.shape) < p_all
    obs = eval_draw[:, keep].mean(axis=1)
    true_pool = p_all.mean(axis=1)
    return FilterSim(abilities, is_filter, true_pool, obs, n_pool, int(keep.sum()))


# --------------------------------------------------------------------------- all-or-nothing composites
def composite_score(p_sub: np.ndarray, k: int = 3) -> np.ndarray:
    """Expected all-or-nothing score when each of k independent subtasks is solved with prob p_sub."""
    return p_sub ** k


def power_two_systems(p_a: float, p_b: float, n_items: int, k: int = 3, all_or_nothing: bool = True,
                      n_sim: int = 4000, seed: int = 0, alpha: float = 0.05) -> float:
    """Probability that a paired test on the same items detects that system A beats system B.

    all_or_nothing=True scores each item 1 only if all k subtasks pass; False gives partial credit
    (mean of the k subtask outcomes). Paired sign-flip permutation-free z-test on item differences.
    """
    rng = np.random.default_rng(seed)
    from scipy import stats as st
    hits = 0
    for _ in range(n_sim):
        a = rng.random((n_items, k)) < p_a
        b = rng.random((n_items, k)) < p_b
        sa = a.all(axis=1).astype(float) if all_or_nothing else a.mean(axis=1)
        sb = b.all(axis=1).astype(float) if all_or_nothing else b.mean(axis=1)
        d = sa - sb
        sd = d.std(ddof=1)
        if sd == 0:
            continue
        z = d.mean() / (sd / np.sqrt(n_items))
        hits += z > st.norm.ppf(1 - alpha / 2)
    return hits / n_sim
