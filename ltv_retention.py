"""
LTV, retention and payback (Objective 2)
========================================

Acquisition cost only makes sense next to what a customer is worth and how long
it takes to earn the money back. This module builds the retention and lifetime-
value side of the unit economics:

* **Retention cohorts** -- monthly signup cohorts tracked by tenure, as a
  triangular heatmap (cells the data cannot yet observe are left blank rather
  than shown as churn).
* **LTV** by channel, by market and by channel x market.
* **Payback period** -- months of revenue needed to recover CAC.
* **LTV:CAC** -- the headline efficiency ratio, per channel.

Outputs go to ``images/``.

Run:  python ltv_retention.py

Author: Muhammad Nasiruddin
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
IMG_DIR = BASE / "images"

# The observation window closes at the last day of generated data; a cohort can
# only be measured out to its age at this date (the rest is censored, not churn).
ANALYSIS_END = pd.Timestamp("2026-06-30")

PALETTE = ["#2563eb", "#0891b2", "#16a34a", "#ca8a04", "#dc2626", "#7c3aed"]
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def load_customers() -> pd.DataFrame:
    """Paying customers only (the population that has a tenure and an LTV)."""
    u = pd.read_csv(DATA_DIR / "users.csv")
    u = u[u["converted"] == 1].copy()
    u["signup_date"] = pd.to_datetime(u["signup_date"])
    u["cohort"] = u["signup_date"].dt.to_period("M")
    return u


def _cohort_age(cohort: pd.Period) -> int:
    """Number of whole months a cohort has had to age by ANALYSIS_END."""
    start = cohort.to_timestamp()
    return (ANALYSIS_END.year - start.year) * 12 + (ANALYSIS_END.month - start.month)


def retention_matrix(customers: pd.DataFrame) -> pd.DataFrame:
    """Retention (% of cohort still active) by cohort x tenure month.

    Retention at tenure ``t`` is the share of the cohort with
    ``months_active > t``. Cells beyond a cohort's observable age are NaN, so the
    matrix is honestly triangular instead of implying churn we cannot see.
    """
    cohorts = sorted(customers["cohort"].unique())
    max_age = max(_cohort_age(c) for c in cohorts)
    mat = pd.DataFrame(index=[str(c) for c in cohorts], columns=range(max_age + 1),
                       dtype=float)

    for c in cohorts:
        grp = customers[customers["cohort"] == c]
        size = len(grp)
        age = _cohort_age(c)
        for t in range(age + 1):
            mat.loc[str(c), t] = (grp["months_active"] > t).sum() / size * 100
    return mat


def plot_retention_heatmap(mat: pd.DataFrame) -> Path:
    """Triangular cohort-retention heatmap."""
    data = mat.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(mat.shape[1]), mat.columns)
    ax.set_yticks(range(mat.shape[0]), mat.index)
    ax.set_xlabel("Tenure (months since signup)")
    ax.set_ylabel("Signup cohort")
    ax.set_title("Customer retention by cohort (%)", fontweight="bold", loc="left")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > 55 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="retention %", shrink=0.85)
    ax.grid(False)
    fig.tight_layout()
    out = IMG_DIR / "retention_cohorts.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def retention_curves_by_channel(customers: pd.DataFrame, max_t: int = 11) -> pd.DataFrame:
    """Survival curve per channel with a proper at-risk denominator.

    For each tenure ``t`` a customer only counts if their cohort is old enough to
    have been observed at ``t`` (``cohort_age >= t``); retention is then the
    share of those at-risk customers still active. This avoids the censoring bias
    that a naive average over all customers would introduce.
    """
    c = customers.copy()
    c["age"] = c["cohort"].map(_cohort_age)
    curves: dict[str, list[float]] = {}
    for channel, grp in c.groupby("channel"):
        row = []
        for t in range(max_t + 1):
            at_risk = grp[grp["age"] >= t]
            row.append((at_risk["months_active"] > t).mean() * 100 if len(at_risk) else np.nan)
        curves[channel] = row
    return pd.DataFrame(curves, index=range(max_t + 1)).T


def plot_retention_curves(curves: pd.DataFrame) -> Path:
    """Retention curves per channel, ordered by 6-month retention."""
    order = curves[6].sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, channel in enumerate(order):
        ax.plot(curves.columns, curves.loc[channel], marker="o", lw=2,
                color=PALETTE[i % len(PALETTE)], label=channel)
    ax.set_xlabel("Tenure (months since signup)")
    ax.set_ylabel("Retention (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Retention curve by acquisition channel", fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = IMG_DIR / "retention_curves_by_channel.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    IMG_DIR.mkdir(exist_ok=True)
    customers = load_customers()

    mat = retention_matrix(customers)
    mat.to_csv(IMG_DIR / "retention_cohorts.csv")
    print("Retention by cohort (%)\n")
    print(mat.round(0).to_string())

    curves = retention_curves_by_channel(customers)
    curves.to_csv(IMG_DIR / "retention_curves_by_channel.csv")
    print("\nRetention by channel and tenure (%)\n")
    print(curves.round(0).to_string())

    p1 = plot_retention_heatmap(mat)
    p2 = plot_retention_curves(curves)
    print(f"\nSaved charts: {p1.name}, {p2.name}")


if __name__ == "__main__":
    main()
