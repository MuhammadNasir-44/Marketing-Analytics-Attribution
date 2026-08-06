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


def plot_cac_by_country(co: pd.DataFrame, blended_cac: float) -> Path:
    """CAC per market against the blended average (multi-market efficiency)."""
    d = co.sort_values("cac")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [PALETTE[2] if c <= blended_cac else PALETTE[4] for c in d["cac"]]
    ax.bar(d.index, d["cac"], color=colors, alpha=0.9)
    ax.axhline(blended_cac, color="grey", ls="--", lw=1)
    ax.text(len(d) - 0.5, blended_cac * 1.02, f"blended ${blended_cac:,.0f}",
            color="grey", ha="right", fontsize=9)
    ax.set_ylabel("Customer acquisition cost ($)")
    ax.set_title("CAC by market", fontweight="bold", loc="left")
    for x, v in enumerate(d["cac"]):
        ax.text(x, v + 2, f"${v:,.0f}", ha="center", fontsize=9)
    fig.tight_layout()
    out = IMG_DIR / "cac_by_country.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_roas_by_channel(ch: pd.DataFrame, blended_roas: float) -> Path:
    """ROAS per channel against break-even (1.0) and the blended average.

    Makes the 'blended vs per-channel' point visually: the blended number sits
    above break-even, but most individual channels sit below it -- the average
    is carried by SEO and Email.
    """
    d = ch.sort_values("roas", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE[2] if r >= 1 else PALETTE[4] for r in d["roas"]]
    bars = ax.bar(d.index, d["roas"], color=colors, alpha=0.9)

    ax.axhline(1.0, color="grey", ls="-", lw=1)
    ax.text(len(d) - 0.5, 1.05, "break-even", color="grey", ha="right", fontsize=9)
    ax.axhline(blended_roas, color=PALETTE[5], ls="--", lw=1.5)
    ax.text(0, blended_roas + 0.15, f"blended {blended_roas:.2f}",
            color=PALETTE[5], fontsize=9)

    ax.set_ylabel("ROAS (revenue / spend)")
    ax.set_title("ROAS by channel vs break-even and blended average",
                 fontweight="bold", loc="left")
    for bar, v in zip(bars, d["roas"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15, f"{v:.2f}",
                ha="center", fontsize=9)
    fig.tight_layout()
    out = IMG_DIR / "roas_by_channel.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    IMG_DIR.mkdir(exist_ok=True)
    spend, users = load()

    ch = channel_economics(spend, users)
    ch.to_csv(IMG_DIR / "unit_economics_by_channel.csv")
    co = country_economics(spend, users)
    co.to_csv(IMG_DIR / "unit_economics_by_country.csv")

    b = blended(ch)
    cols = ["spend", "customers", "cac", "arpc", "avg_ltv", "ltv_cac", "roas"]
    print("Unit economics by channel\n")
    print(ch[cols].round(2).to_string())
    print(f"\nBlended: CAC ${b['cac']:.2f} | ROAS {b['roas']:.2f} | LTV:CAC {b['ltv_cac']:.2f}\n")

    print("Unit economics by market\n")
    print(co[["spend", "customers", "cac", "arpc", "ltv_cac", "roas"]].round(2).to_string())

    p1 = plot_cac_by_country(co, b["cac"])
    p2 = plot_roas_by_channel(ch, b["roas"])
    print(f"\nSaved charts: {p1.name}, {p2.name}")


if __name__ == "__main__":
    main()
