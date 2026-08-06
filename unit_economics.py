"""
Unit economics (Objective 2)
============================

Goes past top-line channel spend into the numbers a growth team actually steers
on: customer-acquisition cost (CAC), return on ad spend (ROAS), average revenue
per customer (ARPC) and the LTV:CAC ratio -- by channel, by market, and by
channel x market. Everything is benchmarked against the *blended* average so it
is obvious which channels subsidise which.

Outputs (to ``images/``):

* ``unit_economics_by_channel.csv`` -- the full per-channel table.
* ``cac_by_country.png``            -- CAC per market vs the blended line.
* ``roas_by_channel.png``           -- ROAS per channel vs break-even & blended.
* ``efficiency_frontier.png``       -- CAC vs customer volume (the acquisition
                                       efficiency frontier).
* ``cac_channel_country_heatmap.png`` -- CAC across channel x market.

Run:  python unit_economics.py

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

PALETTE = ["#2563eb", "#0891b2", "#16a34a", "#ca8a04", "#dc2626", "#7c3aed"]
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    spend = pd.read_csv(DATA_DIR / "spend.csv")
    users = pd.read_csv(DATA_DIR / "users.csv")
    return spend, users


def _economics(spend: pd.DataFrame, users: pd.DataFrame, key: str) -> pd.DataFrame:
    """Unit economics grouped by ``key`` (``channel`` or ``country``).

    Joins media spend to acquired customers and derives the standard growth
    metrics. ARPC and LTV are taken over paying customers only, so they reflect
    the economics of a customer rather than being diluted by non-activators.
    """
    media = spend.groupby(key).agg(spend=("spend", "sum"), clicks=("clicks", "sum"))

    paying = users[users["converted"] == 1]
    acq = users.groupby(key).agg(signups=("user_id", "count"))
    pay = paying.groupby(key).agg(
        customers=("user_id", "count"),
        revenue=("commission_revenue", "sum"),
        avg_ltv=("ltv", "mean"),
    )

    df = media.join(acq).join(pay)
    df["cac"] = df["spend"] / df["customers"]
    df["roas"] = df["revenue"] / df["spend"]
    df["arpc"] = df["revenue"] / df["customers"]        # avg revenue per customer
    df["ltv_cac"] = df["avg_ltv"] / df["cac"]           # lifetime value : CAC
    return df


def channel_economics(spend: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    return _economics(spend, users, "channel").sort_values("roas", ascending=False)


def country_economics(spend: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    return _economics(spend, users, "country").sort_values("roas", ascending=False)


def blended(df: pd.DataFrame) -> dict[str, float]:
    """Blended benchmarks across all rows of an economics table."""
    return {
        "cac": df["spend"].sum() / df["customers"].sum(),
        "roas": df["revenue"].sum() / df["spend"].sum(),
        "ltv_cac": df["avg_ltv"].mul(df["customers"]).sum()
        / df["customers"].sum()
        / (df["spend"].sum() / df["customers"].sum()),
    }


def main() -> None:
    IMG_DIR.mkdir(exist_ok=True)
    spend, users = load()

    ch = channel_economics(spend, users)
    ch.to_csv(IMG_DIR / "unit_economics_by_channel.csv")

    b = blended(ch)
    cols = ["spend", "customers", "cac", "arpc", "avg_ltv", "ltv_cac", "roas"]
    print("Unit economics by channel\n")
    print(ch[cols].round(2).to_string())
    print(f"\nBlended: CAC ${b['cac']:.2f} | ROAS {b['roas']:.2f} | LTV:CAC {b['ltv_cac']:.2f}")


if __name__ == "__main__":
    main()
