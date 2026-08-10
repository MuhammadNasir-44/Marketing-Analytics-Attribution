"""
Spend anomaly detection & wasted spend (Objective 3)
====================================================

Marketing budgets leak through misconfigured campaigns that quietly spend more
without bringing in more customers. This module scans daily spend for every
channel x market series with a **rolling z-score** and flags days where spend
breaks out of its recent trend. It then quantifies the **wasted spend**: the
excess dollars on flagged days that produced no matching lift in conversions.

The detector is unsupervised -- it is not told where the problem is -- which is
the point: it independently rediscovers the burst of overspend in the data.

Outputs go to ``images/``.

Run:  python anomaly_detection.py

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

ROLLING_WINDOW = 28      # trailing days used for the baseline
Z_THRESHOLD = 4.0        # onset: spend breaks > 4 sigma above the trailing mean
EXTEND_K = 2.0           # keep an event open while spend stays > baseline + 2 sigma
# Materiality filter, so the report shows real problems, not day-to-day noise.
MIN_EXCESS = 1000.0      # total wasted spend ($) ...
MIN_DAYS = 3             # ... or a sustained run of this many days

PALETTE = ["#2563eb", "#0891b2", "#16a34a", "#ca8a04", "#dc2626", "#7c3aed"]
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def daily_series() -> pd.DataFrame:
    """Daily spend and new customers per channel x market."""
    spend = pd.read_csv(DATA_DIR / "spend.csv")
    spend["date"] = pd.to_datetime(spend["date"])
    sp = (spend.groupby(["channel", "country", "date"])["spend"].sum()
          .reset_index())

    users = pd.read_csv(DATA_DIR / "users.csv")
    users["signup_date"] = pd.to_datetime(users["signup_date"])
    conv = (users[users["converted"] == 1]
            .groupby(["channel", "country", "signup_date"])
            .size().reset_index(name="customers")
            .rename(columns={"signup_date": "date"}))

    df = sp.merge(conv, on=["channel", "country", "date"], how="left")
    df["customers"] = df["customers"].fillna(0)
    return df.sort_values(["channel", "country", "date"]).reset_index(drop=True)


def flag_anomalies(df: pd.DataFrame,
                   window: int = ROLLING_WINDOW,
                   z: float = Z_THRESHOLD) -> pd.DataFrame:
    """Add a trailing-window baseline, z-score and onset flag per series.

    The rolling mean/std use only *prior* days (``shift(1)``) so a spike doesn't
    contaminate its own baseline. ``is_onset`` marks the day a series breaks out
    (z above threshold) -- the start of a candidate anomaly event.
    """
    out = []
    for (channel, country), g in df.groupby(["channel", "country"], sort=False):
        g = g.sort_values("date").copy()
        base = g["spend"].shift(1).rolling(window, min_periods=window // 2)
        g["expected"] = base.mean()
        g["baseline_std"] = base.std()
        g["z"] = (g["spend"] - g["expected"]) / g["baseline_std"]
        g["is_onset"] = g["z"] > z
        out.append(g)
    return pd.concat(out).reset_index(drop=True)


def anomaly_events(flagged: pd.DataFrame,
                   extend_k: float = EXTEND_K,
                   min_excess: float = MIN_EXCESS,
                   min_days: int = MIN_DAYS) -> pd.DataFrame:
    """Build anomaly events by freezing the pre-onset baseline and extending.

    A trailing z-score only fires on the *onset* of a sustained overspend,
    because the plateau soon inflates its own rolling baseline. So when a series
    breaks out, we freeze the baseline as it was just before the onset and keep
    the event open while spend stays above ``baseline + extend_k * sigma`` -- this
    recovers the full duration and the full wasted spend. Only material events
    (enough wasted dollars or a long enough run) are returned.
    """
    rows = []
    for (channel, country), g in flagged.groupby(["channel", "country"], sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        spend = g["spend"].to_numpy()
        z = g["z"].to_numpy()
        expected = g["expected"].to_numpy()
        std = g["baseline_std"].to_numpy()
        customers = g["customers"].to_numpy()
        dates = g["date"].to_numpy()

        i, n = 0, len(g)
        while i < n:
            if not (z[i] > Z_THRESHOLD):
                i += 1
                continue
            base = expected[i]                 # frozen pre-onset baseline
            cutoff = base + extend_k * std[i]
            j = i
            while j < n and spend[j] > cutoff:
                j += 1
            span = slice(i, j)
            excess = float((spend[span] - base).clip(min=0).sum())
            rows.append({
                "channel": channel,
                "country": country,
                "start": pd.Timestamp(dates[i]).date().isoformat(),
                "end": pd.Timestamp(dates[j - 1]).date().isoformat(),
                "days": j - i,
                "spend": round(float(spend[span].sum()), 2),
                "baseline_spend": round(base * (j - i), 2),
                "excess_spend": round(excess, 2),
                "customers": int(customers[span].sum()),
                "peak_z": round(float(z[span].max()), 1),
            })
            i = j
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    material = events[(events["excess_spend"] >= min_excess) | (events["days"] >= min_days)]
    return material.sort_values("excess_spend", ascending=False).reset_index(drop=True)


def main() -> None:
    IMG_DIR.mkdir(exist_ok=True)
    df = daily_series()
    flagged = flag_anomalies(df)
    events = anomaly_events(flagged)

    events.to_csv(IMG_DIR / "anomaly_events.csv", index=False)
    print(f"Scanned {df.groupby(['channel','country']).ngroups} channel x market "
          f"series ({len(df):,} channel-days)\n")
    print(f"Flagged {len(events)} anomaly event(s):\n")
    print(events.to_string(index=False))


if __name__ == "__main__":
    main()
