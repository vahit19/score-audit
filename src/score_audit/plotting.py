"""Shared figure style: one look for every figure, saved as PNG (for README) and PDF (for the paper)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
RES = ROOT / "results"

BLUE, RED, GRAY, GREEN = "#2f6db5", "#c8423a", "#8a96a3", "#2e8b57"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "legend.frameon": False,
})


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def results_dir() -> Path:
    RES.mkdir(exist_ok=True)
    return RES
