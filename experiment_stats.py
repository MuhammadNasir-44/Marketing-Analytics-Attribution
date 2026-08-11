"""
Experiment statistics
======================

Small, dependency-light statistics toolkit for the experimentation work: a
two-proportion z-test with a confidence interval, a sample-size / power
calculator for two-proportion tests, and a permutation test for cluster-
randomised (geo-holdout) experiments.

Everything is implemented on top of the standard library's
``statistics.NormalDist`` (no scipy), and the functions are pure so they can be
unit-tested directly (see ``tests/test_experiment_stats.py``).

Author: Muhammad Nasiruddin
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

_N = NormalDist()


# --------------------------------------------------------------------------- #
# Two-proportion test
# --------------------------------------------------------------------------- #

@dataclass
class ProportionTest:
    rate_a: float
    rate_b: float
    abs_lift: float          # rate_b - rate_a (percentage points, as a fraction)
    rel_lift: float          # (rate_b - rate_a) / rate_a
    z: float
    p_value: float           # two-sided
    ci_low: float            # 95% CI on the absolute difference
    ci_high: float
    significant: bool


def two_proportion_test(conv_a: int, n_a: int, conv_b: int, n_b: int,
                        alpha: float = 0.05) -> ProportionTest:
    """Two-sided two-proportion z-test of B (treatment) vs A (control).

    Uses the pooled standard error for the test statistic and the unpooled
    standard error for the confidence interval on the difference (the standard
    convention).
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("group sizes must be positive")
    p_a, p_b = conv_a / n_a, conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)

    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - _N.cdf(abs(z)))

    se_diff = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = _N.inv_cdf(1 - alpha / 2)
    diff = p_b - p_a
    return ProportionTest(
        rate_a=p_a,
        rate_b=p_b,
        abs_lift=diff,
        rel_lift=diff / p_a if p_a > 0 else math.nan,
        z=z,
        p_value=p_value,
        ci_low=diff - z_crit * se_diff,
        ci_high=diff + z_crit * se_diff,
        significant=p_value < alpha,
    )


# --------------------------------------------------------------------------- #
# Sample size & power (two proportions)
# --------------------------------------------------------------------------- #

def sample_size_two_proportions(p1: float, p2: float,
                                alpha: float = 0.05, power: float = 0.80) -> int:
    """Required sample size **per arm** to detect p2 vs p1 (two-sided).

    Standard pooled-variance formula; returns the ceiling.
    """
    if p1 == p2:
        raise ValueError("p1 and p2 must differ")
    z_alpha = _N.inv_cdf(1 - alpha / 2)
    z_beta = _N.inv_cdf(power)
    p_bar = (p1 + p2) / 2
    num = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
           + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p2 - p1) ** 2)


def power_two_proportions(p1: float, p2: float, n_per_arm: int,
                          alpha: float = 0.05) -> float:
    """Statistical power to detect p2 vs p1 at a given per-arm sample size."""
    if n_per_arm <= 0:
        return 0.0
    z_alpha = _N.inv_cdf(1 - alpha / 2)
    p_bar = (p1 + p2) / 2
    se_null = math.sqrt(2 * p_bar * (1 - p_bar) / n_per_arm)
    se_alt = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_arm)
    if se_alt == 0:
        return 1.0
    return float(_N.cdf((abs(p2 - p1) - z_alpha * se_null) / se_alt))


# --------------------------------------------------------------------------- #
# Permutation test for geo-holdout (cluster-randomised) experiments
# --------------------------------------------------------------------------- #

@dataclass
class GeoLiftResult:
    mean_test: float
    mean_holdout: float
    abs_lift: float
    rel_lift: float
    p_value: float


def geo_lift_permutation(test_values, holdout_values,
                         n_perm: int = 10000, seed: int = 0) -> GeoLiftResult:
    """Permutation test of the mean difference between test and holdout geos.

    Distribution-free, which suits a cluster-randomised geo experiment with few
    units: it shuffles the test/holdout labels ``n_perm`` times and compares the
    observed mean difference to the resulting null distribution.
    """
    test = np.asarray(test_values, dtype=float)
    hold = np.asarray(holdout_values, dtype=float)
    observed = test.mean() - hold.mean()

    pooled = np.concatenate([test, hold])
    n_test = len(test)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = pooled[:n_test].mean() - pooled[n_test:].mean()
        if abs(diff) >= abs(observed):
            count += 1
    p_value = (count + 1) / (n_perm + 1)      # add-one smoothing

    return GeoLiftResult(
        mean_test=float(test.mean()),
        mean_holdout=float(hold.mean()),
        abs_lift=float(observed),
        rel_lift=float(observed / hold.mean()) if hold.mean() else math.nan,
        p_value=p_value,
    )
