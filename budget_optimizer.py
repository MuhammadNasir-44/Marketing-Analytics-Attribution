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


def recommend(scan: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Budget-neutral reallocation: trim losers, fund winners, project impact.

    Loss-making channels are trimmed by ``CUT_FRACTION``; the freed budget is
    redeployed into the efficient channels in LTV:CAC order, each capped at
    ``SCALE_CAP`` extra spend, then into the Maintain channel. Extra spend earns
    customers at a marginal CAC that is ``SCALE_PENALTY`` worse than current, to
    reflect diminishing returns. Any budget the caps can't absorb is held as
    savings, so the plan never spends more than today.
    """
    m = scan.copy()
    new_spend = m["spend"].copy()

    # 1) Trim the loss-making channels.
    losers = m[m["action"] == "Pause/Fix"].index
    cuts = m.loc[losers, "spend"] * CUT_FRACTION
    new_spend[losers] -= cuts
    freed = float(cuts.sum())

    # 2) Redeploy into Scale channels (best first), then Maintain, within caps.
    targets = (list(m[m["action"] == "Scale"].sort_values("ltv_cac", ascending=False).index)
               + list(m[m["action"] == "Maintain"].index))
    for ch in targets:
        headroom = m.loc[ch, "spend"] * SCALE_CAP
        add = min(headroom, freed)
        new_spend[ch] += add
        freed -= add
        if freed <= 0:
            break
    held_savings = max(freed, 0.0)

    # 3) Project customers at the new spend levels.
    proj = {}
    for ch in m.index:
        old, new = m.loc[ch, "spend"], new_spend[ch]
        cac = m.loc[ch, "cac"]
        if new <= old:                       # trimmed: fewer customers at same CAC
            proj[ch] = new / cac
        else:                                # scaled: extra spend at marginal CAC
            proj[ch] = old / cac + (new - old) / (cac * (1 + SCALE_PENALTY))

    rec = m[["action", "spend", "customers", "cac", "ltv_cac"]].copy()
    rec = rec.rename(columns={"spend": "current_spend", "customers": "current_customers"})
    rec["recommended_spend"] = new_spend
    rec["spend_delta"] = rec["recommended_spend"] - rec["current_spend"]
    rec["projected_customers"] = pd.Series(proj).round(0)
    rec["customer_delta"] = (rec["projected_customers"] - rec["current_customers"]).round(0)
    rec = rec.sort_values("spend_delta", ascending=False)

    summary = {
        "spend_before": float(rec["current_spend"].sum()),
        "spend_after": float(rec["recommended_spend"].sum()),
        "held_savings": held_savings,
        "customers_before": int(rec["current_customers"].sum()),
        "customers_after": int(rec["projected_customers"].sum()),
    }
    summary["cac_before"] = summary["spend_before"] / summary["customers_before"]
    summary["cac_after"] = summary["spend_after"] / summary["customers_after"]
    return rec, summary


ACTION_COLOR = {"Scale": PALETTE[2], "Maintain": PALETTE[3], "Pause/Fix": PALETTE[4]}


def plot_recommendation(rec: pd.DataFrame, summary: dict) -> Path:
    """Current vs recommended spend per channel, plus the projected-impact KPIs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                                   gridspec_kw={"width_ratios": [2.1, 1]})

    d = rec.sort_values("recommended_spend", ascending=True)
    y = range(len(d))
    ax1.barh([i + 0.2 for i in y], d["current_spend"] / 1000, height=0.38,
             color="#c9ced6", label="Current")
    ax1.barh([i - 0.2 for i in y], d["recommended_spend"] / 1000, height=0.38,
             color=[ACTION_COLOR[a] for a in d["action"]], label="Recommended")
    ax1.set_yticks(list(y), d.index)
    ax1.set_xlabel("Spend ($000s)")
    ax1.set_title("Current vs recommended budget by channel", fontweight="bold", loc="left")
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    for i, (cur, act) in enumerate(zip(d["recommended_spend"], d["action"])):
        ax1.text(cur / 1000 + 20, i - 0.2, act, va="center", fontsize=8,
                 color=ACTION_COLOR[act])

    # KPI panel.
    ax2.axis("off")
    dc = summary["customers_after"] - summary["customers_before"]
    # Escape '$' so matplotlib doesn't treat it as mathtext.
    d = r"\$"
    lines = [
        ("Customers", f"{summary['customers_before']:,}",
         f"{summary['customers_after']:,}", f"+{dc:,}  (+{dc / summary['customers_before'] * 100:.0f}%)"),
        ("Blended CAC", f"{d}{summary['cac_before']:.0f}",
         f"{d}{summary['cac_after']:.0f}", f"-{d}{summary['cac_before'] - summary['cac_after']:.0f}"),
        ("Total spend", f"{d}{summary['spend_before'] / 1e6:.2f}m",
         f"{d}{summary['spend_after'] / 1e6:.2f}m", f"{d}{summary['held_savings'] / 1000:.0f}k held"),
    ]
    ax2.set_title("Projected impact", fontweight="bold", loc="left")
    ypos = 0.82
    for label, before, after, delta in lines:
        ax2.text(0.0, ypos, label, fontsize=11, fontweight="bold", transform=ax2.transAxes)
        ax2.text(0.0, ypos - 0.06, f"{before}  ->  {after}", fontsize=11,
                 transform=ax2.transAxes)
        ax2.text(0.0, ypos - 0.12, delta, fontsize=11, color=PALETTE[2],
                 transform=ax2.transAxes)
        ypos -= 0.26

    fig.tight_layout()
    out = IMG_DIR / "budget_reallocation.png"
    fig.savefig(out)
    plt.close(fig)
    return out


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

    rec, summary = recommend(scan)
    rec.to_csv(IMG_DIR / "budget_recommendation.csv")
    print("\nScale / Pause / Test recommendation\n")
    show = rec[["action", "current_spend", "recommended_spend", "spend_delta",
                "current_customers", "projected_customers", "customer_delta"]]
    print(show.round(0).to_string())

    print("\nProjected impact (same total budget):")
    print(f"  spend:     ${summary['spend_before']:,.0f} -> ${summary['spend_after']:,.0f} "
          f"(${summary['held_savings']:,.0f} held as savings)")
    print(f"  customers: {summary['customers_before']:,} -> {summary['customers_after']:,} "
          f"(+{summary['customers_after'] - summary['customers_before']:,})")
    print(f"  blended CAC: ${summary['cac_before']:.0f} -> ${summary['cac_after']:.0f}")

    p = plot_recommendation(rec, summary)
    print(f"\nSaved chart: {p.name}")


if __name__ == "__main__":
    main()
