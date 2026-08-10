"""
Growth opportunities & budget reallocation (Objective 3)
========================================================

Ties the analysis together into a decision. Using each channel's efficiency
(LTV:CAC) and its current scale, it:

1. **Scans for opportunities** -- classifies every channel as *Scale*,
   *Maintain* or *Pause/Fix*, and highlights the under-scaled, high-return
   channels that deserve more budget.
2. **Reallocates the budget** -- a budget-neutral recommender that trims the
   loss-making channels and redeploys the freed spend into the efficient ones,
   subject to a realistic scaling cap and diminishing returns.
3. **Projects the impact** -- the extra customers and lower blended CAC the
   reallocation would deliver at the *same* total spend.

Outputs go to ``images/``.

Run:  python budget_optimizer.py

Author: Muhammad Nasiruddin
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
IMG_DIR = BASE / "images"

# Decision thresholds and scaling assumptions.
SCALE_LTV_CAC = 3.0      # >= this is efficient enough to scale
MIN_LTV_CAC = 1.0        # < this loses money -> pause / fix
SCALE_CAP = 0.50         # a channel can absorb at most +50% spend in-period
CUT_FRACTION = 0.60      # trim loss-making channels by 60%
SCALE_PENALTY = 0.25     # marginal CAC on extra spend is 25% worse (diminishing returns)

PALETTE = ["#2563eb", "#0891b2", "#16a34a", "#ca8a04", "#dc2626", "#7c3aed"]
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def base_metrics() -> pd.DataFrame:
    """Per-channel spend, customers, CAC, avg LTV and LTV:CAC."""
    spend = pd.read_csv(DATA_DIR / "spend.csv")
    users = pd.read_csv(DATA_DIR / "users.csv")
    paying = users[users["converted"] == 1]

    df = pd.DataFrame({
        "spend": spend.groupby("channel")["spend"].sum(),
        "customers": paying.groupby("channel")["user_id"].count(),
        "avg_ltv": paying.groupby("channel")["ltv"].mean(),
    })
    df["cac"] = df["spend"] / df["customers"]
    df["ltv_cac"] = df["avg_ltv"] / df["cac"]
    df["spend_share_pct"] = df["spend"] / df["spend"].sum() * 100
    return df.sort_values("ltv_cac", ascending=False)


def classify(ltv_cac: float) -> str:
    if ltv_cac >= SCALE_LTV_CAC:
        return "Scale"
    if ltv_cac >= MIN_LTV_CAC:
        return "Maintain"
    return "Pause/Fix"


def opportunity_scan(metrics: pd.DataFrame) -> pd.DataFrame:
    """Label channels and flag under-scaled, high-return opportunities."""
    m = metrics.copy()
    m["action"] = m["ltv_cac"].map(classify)
    # An opportunity is an efficient channel that is starved of budget.
    m["under_scaled"] = (m["action"] == "Scale") & (m["spend_share_pct"] < 15)
    return m


def main() -> None:
    IMG_DIR.mkdir(exist_ok=True)
    metrics = base_metrics()
    scan = opportunity_scan(metrics)

    cols = ["spend", "spend_share_pct", "customers", "cac", "ltv_cac",
            "action", "under_scaled"]
    print("Opportunity scan\n")
    print(scan[cols].round(2).to_string())

    opp = scan[scan["under_scaled"]]
    print("\nUnder-scaled high-return channels (grow these):",
          ", ".join(opp.index))


if __name__ == "__main__":
    main()
