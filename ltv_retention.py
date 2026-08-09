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


def ltv_by(customers: pd.DataFrame, key: str) -> pd.DataFrame:
    """Average predicted LTV and observed revenue per customer, grouped by key."""
    g = customers.groupby(key).agg(
        customers=("user_id", "count"),
        avg_ltv=("ltv", "mean"),
        avg_observed_rev=("commission_revenue", "mean"),
        avg_tenure=("months_active", "mean"),
    )
    return g.sort_values("avg_ltv", ascending=False)


def value_segments(customers: pd.DataFrame) -> pd.DataFrame:
    """Split customers into High/Mid/Low LTV terciles and show revenue concentration.

    Surfaces the commercial reality that value is concentrated -- a small share of
    customers drives a large share of revenue, which is what a retention/CRM
    programme should be pointed at.
    """
    c = customers.copy()
    c["segment"] = pd.qcut(c["ltv"], 3, labels=["Low", "Mid", "High"])
    seg = c.groupby("segment", observed=True).agg(
        customers=("user_id", "count"),
        avg_ltv=("ltv", "mean"),
        total_revenue=("commission_revenue", "sum"),
    )
    seg["customer_share_pct"] = seg["customers"] / seg["customers"].sum() * 100
    seg["revenue_share_pct"] = seg["total_revenue"] / seg["total_revenue"].sum() * 100
    return seg.loc[["High", "Mid", "Low"]]


def plot_ltv_and_segments(
    ltv_channel: pd.DataFrame, seg: pd.DataFrame,
) -> Path:
    """Avg LTV by channel alongside the value-segment revenue concentration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    d = ltv_channel.sort_values("avg_ltv")
    ax1.barh(d.index, d["avg_ltv"], color=PALETTE[0], alpha=0.9)
    ax1.set_xlabel("Average lifetime value ($)")
    ax1.set_title("LTV by acquisition channel", fontweight="bold", loc="left")
    for y, v in enumerate(d["avg_ltv"]):
        ax1.text(v + 4, y, f"${v:,.0f}", va="center", fontsize=9)

    x = np.arange(len(seg))
    ax2.bar(x - 0.2, seg["customer_share_pct"], 0.4, label="% of customers",
            color=PALETTE[1], alpha=0.9)
    ax2.bar(x + 0.2, seg["revenue_share_pct"], 0.4, label="% of revenue",
            color=PALETTE[2], alpha=0.9)
    ax2.set_xticks(x, [f"{s}\n(${v:,.0f} LTV)" for s, v in zip(seg.index, seg["avg_ltv"])])
    ax2.set_ylabel("Share (%)")
    ax2.set_title("Value concentration by LTV segment", fontweight="bold", loc="left")
    ax2.legend(frameon=False, fontsize=9)
    for i, (cs, rs) in enumerate(zip(seg["customer_share_pct"], seg["revenue_share_pct"])):
        ax2.text(i - 0.2, cs + 1, f"{cs:.0f}%", ha="center", fontsize=8)
        ax2.text(i + 0.2, rs + 1, f"{rs:.0f}%", ha="center", fontsize=8)

    fig.tight_layout()
    out = IMG_DIR / "ltv_and_value_segments.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def payback_scorecard(customers: pd.DataFrame) -> pd.DataFrame:
    """CAC, monthly ARPU, payback period and LTV:CAC per channel.

    Payback is CAC divided by the average revenue a customer generates per active
    month -- i.e. how many months of a customer's life it takes to recoup what we
    paid to acquire them. It is the cash-flow complement to LTV:CAC: a channel can
    have a healthy lifetime ratio yet still tie up cash for a long time.
    """
    spend = pd.read_csv(DATA_DIR / "spend.csv")
    cac = spend.groupby("channel")["spend"].sum() / customers.groupby("channel")["user_id"].count()

    grp = customers.groupby("channel")
    monthly_arpu = grp["commission_revenue"].sum() / grp["months_active"].sum()
    avg_ltv = grp["ltv"].mean()

    df = pd.DataFrame({
        "cac": cac,
        "monthly_arpu": monthly_arpu,
        "avg_ltv": avg_ltv,
    })
    df["payback_months"] = df["cac"] / df["monthly_arpu"]
    df["ltv_cac"] = df["avg_ltv"] / df["cac"]
    return df.sort_values("payback_months")


def plot_payback_scorecard(sc: pd.DataFrame, healthy_ltv_cac: float = 3.0) -> Path:
    """Payback period and LTV:CAC per channel, with health thresholds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    d = sc.sort_values("payback_months")
    colors = [PALETTE[2] if m <= 12 else PALETTE[4] for m in d["payback_months"]]
    ax1.barh(d.index, d["payback_months"], color=colors, alpha=0.9)
    ax1.axvline(12, color="grey", ls="--", lw=1)
    ax1.text(12, -0.7, "12-month target", color="grey", fontsize=9, ha="center")
    ax1.set_xlabel("Payback period (months)")
    ax1.set_title("CAC payback by channel", fontweight="bold", loc="left")
    for y, v in enumerate(d["payback_months"]):
        label = f"{v:,.1f}" if np.isfinite(v) else "n/a"
        ax1.text(v + 0.3, y, label, va="center", fontsize=9)

    d2 = sc.sort_values("ltv_cac")
    colors2 = [PALETTE[2] if r >= healthy_ltv_cac else
               (PALETTE[3] if r >= 1 else PALETTE[4]) for r in d2["ltv_cac"]]
    ax2.barh(d2.index, d2["ltv_cac"], color=colors2, alpha=0.9)
    ax2.axvline(healthy_ltv_cac, color="grey", ls="--", lw=1)
    ax2.text(healthy_ltv_cac, -0.7, "3:1 healthy", color="grey", fontsize=9, ha="center")
    ax2.set_xlabel("LTV : CAC")
    ax2.set_title("LTV:CAC by channel", fontweight="bold", loc="left")
    for y, v in enumerate(d2["ltv_cac"]):
        ax2.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=9)

    fig.tight_layout()
    out = IMG_DIR / "payback_and_ltv_cac.png"
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

    ltv_channel = ltv_by(customers, "channel")
    ltv_country = ltv_by(customers, "country")
    ltv_channel.to_csv(IMG_DIR / "ltv_by_channel.csv")
    ltv_country.to_csv(IMG_DIR / "ltv_by_country.csv")
    seg = value_segments(customers)
    seg.to_csv(IMG_DIR / "ltv_value_segments.csv")

    print("\nLTV by channel\n")
    print(ltv_channel.round(2).to_string())
    print("\nLTV by market\n")
    print(ltv_country.round(2).to_string())
    print("\nValue segments (LTV terciles)\n")
    print(seg.round(2).to_string())

    scorecard = payback_scorecard(customers)
    scorecard.to_csv(IMG_DIR / "payback_ltv_cac_by_channel.csv")
    print("\nPayback & LTV:CAC scorecard by channel\n")
    print(scorecard.round(2).to_string())

    p1 = plot_retention_heatmap(mat)
    p2 = plot_retention_curves(curves)
    p3 = plot_ltv_and_segments(ltv_channel, seg)
    p4 = plot_payback_scorecard(scorecard)
    print(f"\nSaved charts: {p1.name}, {p2.name}, {p3.name}, {p4.name}")


if __name__ == "__main__":
    main()
