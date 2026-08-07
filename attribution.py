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


def _converting_paths_with_recency(
    journeys: pd.DataFrame,
) -> list[tuple[list[str], np.ndarray]]:
    """Per converting user: (channels, days-before-conversion for each touch).

    Conversion is taken to happen at the last (most recent) touch, so the
    closing touch has recency 0. Used by the time-decay model.
    """
    conv = journeys[journeys["converted"] == 1].copy()
    conv["touch_date"] = pd.to_datetime(conv["touch_date"])
    out: list[tuple[list[str], np.ndarray]] = []
    for _, g in conv.groupby("user_id", sort=False):
        conv_date = g["touch_date"].max()
        days_before = (conv_date - g["touch_date"]).dt.days.to_numpy(dtype=float)
        out.append((list(g["channel"]), days_before))
    return out


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


def linear(paths: list[list[str]], channels: list[str]) -> pd.Series:
    """Equal credit to every touch in the journey."""
    credit = pd.Series(0.0, index=channels)
    for p in paths:
        share = 1.0 / len(p)
        for c in p:
            credit[c] += share
    return credit


def time_decay(
    journeys_with_recency: list[tuple[list[str], np.ndarray]],
    channels: list[str],
    half_life: float = TIME_DECAY_HALFLIFE_DAYS,
) -> pd.Series:
    """Exponential decay: touches closer to conversion get more credit.

    Weight for a touch that happened ``d`` days before conversion is
    ``0.5 ** (d / half_life)``; weights are normalised to one conversion.
    """
    credit = pd.Series(0.0, index=channels)
    for path, days_before in journeys_with_recency:
        weights = 0.5 ** (days_before / half_life)
        weights = weights / weights.sum()
        for c, w in zip(path, weights):
            credit[c] += w
    return credit


def position_based(
    paths: list[list[str]], channels: list[str],
    first_weight: float = 0.4, last_weight: float = 0.4,
) -> pd.Series:
    """U-shaped credit: 40% first touch, 40% last, 20% split across the middle.

    Single-touch journeys take the full credit; two-touch journeys split the
    first/last weights only (re-normalised), since there is no middle.
    """
    credit = pd.Series(0.0, index=channels)
    mid_weight = 1.0 - first_weight - last_weight
    for p in paths:
        n = len(p)
        if n == 1:
            credit[p[0]] += 1.0
        elif n == 2:
            total = first_weight + last_weight
            credit[p[0]] += first_weight / total
            credit[p[-1]] += last_weight / total
        else:
            credit[p[0]] += first_weight
            credit[p[-1]] += last_weight
            share = mid_weight / (n - 2)
            for c in p[1:-1]:
                credit[c] += share
    return credit


# --------------------------------------------------------------------------- #
# Data-driven: Markov-chain removal effect
# --------------------------------------------------------------------------- #

def _transition_counts(journeys: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Count state-to-state transitions across *all* journeys.

    Each journey becomes  start -> ch1 -> ... -> chN -> (conv | null),  where the
    terminal is ``conv`` for paying users and ``null`` otherwise. Learning from
    the non-converting paths too is what makes this data-driven rather than a
    fixed rule.
    """
    counts: dict[tuple[str, str], int] = {}
    for _, g in journeys.groupby("user_id", sort=False):
        seq = ["start"] + list(g["channel"])
        seq.append("conv" if g["converted"].iloc[0] == 1 else "null")
        for a, b in zip(seq[:-1], seq[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def _conversion_probability(
    counts: dict[tuple[str, str], int], channels: list[str], removed: str | None = None
) -> float:
    """P(reaching ``conv`` from ``start``) in the absorbing Markov chain.

    If ``removed`` is set, that channel is deleted from the graph and every
    transition into it is redirected to ``null`` (the conversion path is broken)
    -- the basis of the removal effect.
    """
    transient = ["start"] + [c for c in channels if c != removed]
    absorbing = ["conv", "null"]
    all_states = transient + absorbing
    t_idx = {s: i for i, s in enumerate(transient)}
    a_idx = {s: i for i, s in enumerate(all_states)}

    M = np.zeros((len(transient), len(all_states)))
    for (a, b), n in counts.items():
        if a == removed:
            continue                      # removed channel is never reached
        tgt = "null" if b == removed else b
        M[t_idx[a], a_idx[tgt]] += n

    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = M / row_sums

    n_t = len(transient)
    Q = P[:, :n_t]
    R = P[:, n_t:]
    N = np.linalg.inv(np.eye(n_t) - Q)
    B = N @ R                              # absorption probabilities
    return float(B[t_idx["start"], all_states.index("conv") - n_t])


def markov(journeys: pd.DataFrame, channels: list[str], total: float) -> pd.Series:
    """Data-driven credit from Markov removal effects, scaled to ``total``.

    The removal effect of a channel is the relative drop in conversion
    probability when it is taken out of the journeys; normalising the removal
    effects distributes the conversions across channels.
    """
    counts = _transition_counts(journeys)
    baseline = _conversion_probability(counts, channels)
    effects = {
        c: 1.0 - _conversion_probability(counts, channels, removed=c) / baseline
        for c in channels
    }
    eff = pd.Series(effects, index=channels)
    return eff / eff.sum() * total


if __name__ == "__main__":
    journeys, channels = load_journeys()
    paths = _converting_paths(journeys)
    recency = _converting_paths_with_recency(journeys)
    print(f"{len(paths):,} converting journeys, {len(channels)} channels\n")

    comparison = pd.DataFrame({
        "first_touch": first_touch(paths, channels),
        "last_touch": last_touch(paths, channels),
        "linear": linear(paths, channels),
        "time_decay": time_decay(recency, channels),
        "position_based": position_based(paths, channels),
        "data_driven": markov(journeys, channels, total=len(paths)),
    })
    print(comparison.round(0).to_string())
    print("\ncolumn totals:", comparison.sum().round(0).to_dict())
