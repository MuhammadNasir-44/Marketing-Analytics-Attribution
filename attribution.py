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

import matplotlib.pyplot as plt
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


MODEL_ORDER = [
    "first_touch", "last_touch", "linear",
    "time_decay", "position_based", "data_driven",
]


def compare_models(journeys: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """Credited conversions per channel under every model (columns conserve)."""
    paths = _converting_paths(journeys)
    recency = _converting_paths_with_recency(journeys)
    total = len(paths)
    return pd.DataFrame({
        "first_touch": first_touch(paths, channels),
        "last_touch": last_touch(paths, channels),
        "linear": linear(paths, channels),
        "time_decay": time_decay(recency, channels),
        "position_based": position_based(paths, channels),
        "data_driven": markov(journeys, channels, total=total),
    })[MODEL_ORDER]


def plot_model_comparison(comp: pd.DataFrame) -> Path:
    """Grouped bars: credited conversions per channel across all six models."""
    order = comp["data_driven"].sort_values(ascending=False).index
    d = comp.loc[order]
    x = np.arange(len(d))
    width = 0.13

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, model in enumerate(MODEL_ORDER):
        ax.bar(x + (i - 2.5) * width, d[model], width,
               label=model.replace("_", "-"), color=PALETTE[i], alpha=0.9)
    ax.set_xticks(x, d.index)
    ax.set_ylabel("Credited conversions")
    ax.set_title("Attribution by model: credited conversions per channel",
                 fontweight="bold", loc="left")
    ax.legend(ncol=3, fontsize=9, frameon=False)
    fig.tight_layout()
    out = IMG_DIR / "attribution_model_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_lasttouch_bias(comp: pd.DataFrame) -> Path:
    """How much last-touch over- or under-credits each channel vs data-driven."""
    diff = ((comp["last_touch"] - comp["data_driven"])
            / comp["data_driven"] * 100).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE[4] if v < 0 else PALETTE[2] for v in diff]
    ax.barh(diff.index, diff, color=colors, alpha=0.9)
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel("Last-touch credit vs data-driven (%)")
    ax.set_title("Last-touch bias: who the platform default over- and under-credits",
                 fontweight="bold", loc="left")
    for y, v in enumerate(diff):
        ax.text(v + (2 if v >= 0 else -2), y, f"{v:+.0f}%",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.margins(x=0.15)
    fig.tight_layout()
    out = IMG_DIR / "attribution_lasttouch_bias.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    IMG_DIR.mkdir(exist_ok=True)
    journeys, channels = load_journeys()
    comp = compare_models(journeys, channels)

    comp.to_csv(IMG_DIR / "attribution_by_model.csv")
    shares = (comp / comp.sum() * 100).round(1)
    shares.to_csv(IMG_DIR / "attribution_share_by_model.csv")

    n_conv = int(comp["first_touch"].sum())
    print(f"{n_conv:,} converting journeys, {len(channels)} channels\n")
    print("Credited conversions by model\n")
    print(comp.round(0).to_string())
    print("\nShare of credit (%)\n")
    print(shares.to_string())

    p1 = plot_model_comparison(comp)
    p2 = plot_lasttouch_bias(comp)
    print(f"\nSaved charts: {p1.name}, {p2.name}")


if __name__ == "__main__":
    main()
