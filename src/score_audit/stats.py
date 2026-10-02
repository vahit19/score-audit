"""Small statistics helpers used across the experiments."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    z = st.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def kendall_tau(a, b) -> float:
    """Rank agreement between two score vectors over the same systems."""
    return float(st.kendalltau(a, b).statistic)


def cohens_kappa(r1, r2) -> float:
    """Chance-corrected agreement between two raters on the same items."""
    r1, r2 = list(r1), list(r2)
    labels = sorted(set(r1) | set(r2), key=str)
    n = len(r1)
    po = sum(x == y for x, y in zip(r1, r2)) / n
    pe = sum((r1.count(l) / n) * (r2.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def mde_paired_binary(n: int, discordant_rate: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable accuracy difference for two systems scored on the same n binary items."""
    z = st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)
    return z * math.sqrt(discordant_rate / n)


def bootstrap_mean_ci(x, n_boot: int = 2000, conf: float = 0.95, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return (float(np.quantile(means, (1 - conf) / 2)), float(np.quantile(means, (1 + conf) / 2)))
