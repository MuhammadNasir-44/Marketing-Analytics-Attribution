"""
Channel performance (Objective 1)
=================================

Reads the marketing dataset and answers the first question any growth team asks:
*where is the money going, and what is it bringing back?* It builds a channel
summary (spend, customers, CAC, revenue, ROAS) and saves two charts to
``images/``:

* ``channel_spend_vs_roas.png`` -- spend by channel with ROAS overlaid, so
  over-invested low-return channels stand out.
* ``channel_cac.png``          -- customer-acquisition cost by channel against
  the blended average.

The summary table is also written to ``images/channel_summary.csv``.

Run:  python channel_performance.py

Author: Muhammad Nasiruddin
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
IMG_DIR = BASE / "images"

# A restrained, colour-blind-friendly palette used across the project's charts.
PALETTE = ["#2563eb", "#0891b2", "#16a34a", "#ca8a04", "#dc2626", "#7c3aed"]
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def channel_summary() -> pd.DataFrame:
    """Per-channel spend, customers, CAC, revenue and ROAS, sorted by ROAS."""
    spend = pd.read_csv(DATA_DIR / "spend.csv")
    users = pd.read_csv(DATA_DIR / "users.csv")

    media = spend.groupby("channel").agg(
        spend=("spend", "sum"),
        clicks=("clicks", "sum"),
    )
    acq = users.groupby("channel").agg(
        signups=("user_id", "count"),
        customers=("converted", "sum"),
        revenue=("commission_revenue", "sum"),
    )
    df = media.join(acq)
    df["cac"] = df["spend"] / df["customers"]
    df["roas"] = df["revenue"] / df["spend"]
    df["click_to_signup_pct"] = df["signups"] / df["clicks"] * 100
    return df.sort_values("roas", ascending=False)


def plot_spend_vs_roas(df: pd.DataFrame) -> Path:
    """Spend bars with a ROAS line and a break-even (ROAS = 1) reference."""
    d = df.sort_values("spend", ascending=False)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.bar(d.index, d["spend"] / 1000, color=PALETTE[0], alpha=0.85, label="Spend")
    ax1.set_ylabel("Spend ($000s)")
    ax1.set_xlabel("")

    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(d.index, d["roas"], color=PALETTE[4], marker="o", lw=2, label="ROAS")
    ax2.axhline(1.0, color="grey", ls="--", lw=1)
    ax2.text(len(d) - 0.5, 1.03, "break-even", color="grey", ha="right", fontsize=9)
    ax2.set_ylabel("ROAS (revenue / spend)")
    ax2.set_ylim(0, max(df["roas"]) * 1.15)

    ax1.set_title("Spend vs return by channel", fontweight="bold", loc="left")
    fig.tight_layout()
    out = IMG_DIR / "channel_spend_vs_roas.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_cac(df: pd.DataFrame) -> Path:
    """CAC by channel against the blended average."""
    d = df.sort_values("cac")
    blended = d["spend"].sum() / d["customers"].sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE[2] if c <= blended else PALETTE[4] for c in d["cac"]]
    ax.barh(d.index, d["cac"], color=colors, alpha=0.9)
    ax.axvline(blended, color="grey", ls="--", lw=1)
    ax.text(blended, -0.6, f"  blended ${blended:,.0f}", color="grey", fontsize=9)
    ax.set_xlabel("Customer acquisition cost ($)")
    ax.set_title("CAC by channel", fontweight="bold", loc="left")

    for y, v in enumerate(d["cac"]):
        ax.text(v + 3, y, f"${v:,.0f}", va="center", fontsize=9)

    fig.tight_layout()
    out = IMG_DIR / "channel_cac.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    IMG_DIR.mkdir(exist_ok=True)
    df = channel_summary()

    out = df.copy()
    out.to_csv(IMG_DIR / "channel_summary.csv")

    show = out.assign(
        spend=out["spend"].round(0),
        cac=out["cac"].round(2),
        roas=out["roas"].round(2),
        revenue=out["revenue"].round(0),
        click_to_signup_pct=out["click_to_signup_pct"].round(2),
    )[["spend", "customers", "click_to_signup_pct", "cac", "revenue", "roas"]]
    print(show.to_string())

    p1 = plot_spend_vs_roas(df)
    p2 = plot_cac(df)
    print(f"\nSaved charts: {p1.name}, {p2.name}")


if __name__ == "__main__":
    main()
