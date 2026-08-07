"""
Multi-touch attribution (Objective 2)
=====================================

Last-touch attribution is the default in most ad platforms, and it is
systematically wrong: it hands all the credit to the closing click and none to
the channels that created demand in the first place. This module credits each
conversion across the *whole* journey in ``touchpoints.csv`` using six models,
so the bias of any single one is visible:

* **First-touch**    - 100% to the first touch (demand creation).
* **Last-touch**     - 100% to the last touch (the platform default).
* **Linear**         - equal credit to every touch.
* **Time-decay**     - exponential weighting toward the most recent touch.
* **Position-based** - U-shaped: 40% first, 40% last, 20% to the middle.
* **Data-driven**    - a Markov-chain *removal effect*, which measures each
                       channel's real contribution by how much conversion
                       probability drops when it is taken out of the journeys.

The point is decision-useful: how much credit Display, Paid Social and the other
upper-funnel channels earn depends entirely on the model, and last-touch
undercounts them badly.

Run:  python attribution.py

Author: Muhammad Nasiruddin
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
IMG_DIR = BASE / "images"

PALETTE = ["#2563eb", "#0891b2", "#16a34a", "#ca8a04", "#dc2626", "#7c3aed"]

TIME_DECAY_HALFLIFE_DAYS = 7.0


def load_journeys() -> tuple[pd.DataFrame, list[str]]:
    """Return touchpoints joined to each user's conversion flag, ordered.

    The channel order is preserved via ``seq``; ``converted`` marks which
    journeys ended in a paying customer (needed for the data-driven model, which
    learns from non-converting paths too).
    """
    tp = pd.read_csv(DATA_DIR / "touchpoints.csv")
    users = pd.read_csv(DATA_DIR / "users.csv")[["user_id", "converted"]]
    df = tp.merge(users, on="user_id", how="left")
    df = df.sort_values(["user_id", "seq"]).reset_index(drop=True)
    channels = sorted(df["channel"].unique())
    return df, channels


def _converting_paths(journeys: pd.DataFrame) -> list[list[str]]:
    """Ordered channel lists for converting users only."""
    conv = journeys[journeys["converted"] == 1]
    return [list(g["channel"]) for _, g in conv.groupby("user_id", sort=False)]


# --------------------------------------------------------------------------- #
# Heuristic (rules-based) models
# --------------------------------------------------------------------------- #

def first_touch(paths: list[list[str]], channels: list[str]) -> pd.Series:
    credit = pd.Series(0.0, index=channels)
    for p in paths:
        credit[p[0]] += 1.0
    return credit


def last_touch(paths: list[list[str]], channels: list[str]) -> pd.Series:
    credit = pd.Series(0.0, index=channels)
    for p in paths:
        credit[p[-1]] += 1.0
    return credit


if __name__ == "__main__":
    journeys, channels = load_journeys()
    paths = _converting_paths(journeys)
    print(f"{len(paths):,} converting journeys, {len(channels)} channels\n")

    comparison = pd.DataFrame({
        "first_touch": first_touch(paths, channels),
        "last_touch": last_touch(paths, channels),
    })
    print(comparison.round(0).to_string())
