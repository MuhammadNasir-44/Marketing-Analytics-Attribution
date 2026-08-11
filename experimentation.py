"""
CRO & experimentation (Objective 4)
===================================

The measurement discipline behind growth decisions: does a change actually
work, and would we have detected it if it did? Three pieces:

1. **Landing-page A/B test** -- a two-variant conversion test analysed with a
   two-proportion z-test, absolute/relative lift and a 95% confidence interval.
2. **Pre-launch measurement plan** -- the sample size and run-time needed to
   detect a target lift with 80% power, plus a power curve.
3. **Geo-holdout incrementality** -- a cluster-randomised test that measures the
   *incremental* conversions ads actually cause (vs the correlation last-touch
   attribution reports), analysed with a distribution-free permutation test.

Experiment data is synthetic but seeded and realistic. Statistics come from
``experiment_stats.py``; charts go to ``images/``.

Run:  python experimentation.py

Author: Muhammad Nasiruddin
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_stats import (
    geo_lift_permutation,
    power_two_proportions,
    sample_size_two_proportions,
    two_proportion_test,
)

BASE = Path(__file__).parent
IMG_DIR = BASE / "images"
SEED = 20260808

PALETTE = ["#2563eb", "#0891b2", "#16a34a", "#ca8a04", "#dc2626", "#7c3aed"]
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# True (unknown-to-the-analyst) landing-page conversion rates.
CONTROL_CVR = 0.120
TREATMENT_CVR = 0.138
VISITORS_PER_ARM = 8200


def simulate_ab_test(rng: np.random.Generator) -> dict:
    """Simulate a landing-page A/B test and analyse it."""
    conv_a = int(rng.binomial(VISITORS_PER_ARM, CONTROL_CVR))
    conv_b = int(rng.binomial(VISITORS_PER_ARM, TREATMENT_CVR))
    result = two_proportion_test(conv_a, VISITORS_PER_ARM, conv_b, VISITORS_PER_ARM)
    return {"conv_a": conv_a, "conv_b": conv_b, "n": VISITORS_PER_ARM, "result": result}


def plot_ab_test(ab: dict) -> Path:
    """Conversion rate by variant with 95% CIs, plus the lift verdict."""
    r = ab["result"]
    from experiment_stats import _N
    z = _N.inv_cdf(0.975)

    def ci(p, n):
        se = (p * (1 - p) / n) ** 0.5
        return z * se

    rates = [r.rate_a * 100, r.rate_b * 100]
    errs = [ci(r.rate_a, ab["n"]) * 100, ci(r.rate_b, ab["n"]) * 100]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.bar(["Control (A)", "Treatment (B)"], rates,
                  yerr=errs, capsize=8, color=[PALETTE[1], PALETTE[2]], alpha=0.9)
    ax.set_ylabel("Conversion rate (%)")
    ax.set_title("Landing-page A/B test: conversion by variant",
                 fontweight="bold", loc="left")
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15, f"{v:.2f}%",
                ha="center", fontsize=10)

    verdict = ("significant" if r.significant else "not significant")
    ax.text(0.5, 0.94,
            f"+{r.abs_lift * 100:.2f}pp  ({r.rel_lift * 100:+.1f}%)   "
            f"p = {r.p_value:.4f}  ->  {verdict}",
            transform=ax.transAxes, ha="center", fontsize=10,
            color=PALETTE[2] if r.significant else PALETTE[4])
    fig.tight_layout()
    out = IMG_DIR / "ab_test_conversion.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def measurement_plan(baseline: float = CONTROL_CVR,
                     daily_visitors_per_arm: int = 550) -> pd.DataFrame:
    """Pre-launch plan: sample size & run-time to detect a range of lifts.

    For each relative lift a team might target, compute the per-arm sample size
    needed for 80% power at alpha 0.05, and translate it into test days at the
    given traffic. This is the table you agree *before* launching, so nobody
    calls a winner on an underpowered test.
    """
    rows = []
    for rel in [0.05, 0.075, 0.10, 0.15, 0.20]:
        p2 = baseline * (1 + rel)
        n = sample_size_two_proportions(baseline, p2)
        rows.append({
            "target_rel_lift": f"{rel * 100:.1f}%",
            "treatment_cvr": round(p2, 4),
            "n_per_arm": n,
            "total_n": n * 2,
            "test_days": int(np.ceil(n / daily_visitors_per_arm)),
        })
    return pd.DataFrame(rows)


def plot_power_curve(baseline: float = CONTROL_CVR) -> Path:
    """Power vs per-arm sample size for a few target lifts."""
    ns = np.arange(500, 30001, 500)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, rel in enumerate([0.05, 0.10, 0.15, 0.20]):
        p2 = baseline * (1 + rel)
        power = [power_two_proportions(baseline, p2, int(n)) for n in ns]
        ax.plot(ns, power, lw=2, color=PALETTE[i], label=f"+{rel * 100:.0f}% lift")
    ax.axhline(0.80, color="grey", ls="--", lw=1)
    ax.text(ns[-1], 0.81, "80% power target", color="grey", ha="right", fontsize=9)
    ax.set_xlabel("Sample size per arm")
    ax.set_ylabel("Statistical power")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Power curve (baseline CVR {baseline * 100:.0f}%, alpha 0.05)",
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = IMG_DIR / "power_curve.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    IMG_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    ab = simulate_ab_test(rng)
    r = ab["result"]
    print("Landing-page A/B test\n")
    print(f"  Control  (A): {ab['conv_a']:,}/{ab['n']:,}  = {r.rate_a * 100:.2f}%")
    print(f"  Treatment(B): {ab['conv_b']:,}/{ab['n']:,}  = {r.rate_b * 100:.2f}%")
    print(f"  Lift: +{r.abs_lift * 100:.2f}pp  ({r.rel_lift * 100:+.1f}%)")
    print(f"  95% CI on lift: ({r.ci_low * 100:+.2f}pp, {r.ci_high * 100:+.2f}pp)")
    print(f"  z = {r.z:.2f},  p = {r.p_value:.4f}  ->  "
          f"{'SIGNIFICANT' if r.significant else 'not significant'}")

    plan = measurement_plan()
    plan.to_csv(IMG_DIR / "measurement_plan.csv", index=False)
    print("\nPre-launch measurement plan (80% power, alpha 0.05)\n")
    print(plan.to_string(index=False))

    p1 = plot_ab_test(ab)
    p2 = plot_power_curve()
    print(f"\nSaved charts: {p1.name}, {p2.name}")


if __name__ == "__main__":
    main()
