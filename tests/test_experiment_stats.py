"""
Unit tests for experiment_stats.py
===================================

These test the statistics that the experimentation conclusions rest on -- the
two-proportion test, the sample-size / power calculator (which must be mutually
consistent) and the geo permutation test -- against known values and invariants.

Run:  pytest
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiment_stats import (
    geo_lift_permutation,
    power_two_proportions,
    sample_size_two_proportions,
    two_proportion_test,
)


# --------------------------------------------------------------------------- #
# Two-proportion test
# --------------------------------------------------------------------------- #

def test_two_proportion_lift_and_direction():
    r = two_proportion_test(1000, 10000, 1200, 10000)
    assert r.rate_a == pytest.approx(0.10)
    assert r.rate_b == pytest.approx(0.12)
    assert r.abs_lift == pytest.approx(0.02)
    assert r.rel_lift == pytest.approx(0.20)
    assert r.significant                       # 10% vs 12% at n=10k is significant
    assert r.ci_low < r.abs_lift < r.ci_high


def test_two_proportion_no_difference_not_significant():
    r = two_proportion_test(1000, 10000, 1000, 10000)
    assert r.abs_lift == pytest.approx(0.0)
    assert r.z == pytest.approx(0.0)
    assert r.p_value == pytest.approx(1.0)
    assert not r.significant


def test_two_proportion_ci_brackets_zero_when_tiny_effect():
    # A 0.1pp difference on modest samples should not be significant.
    r = two_proportion_test(1000, 10000, 1010, 10000)
    assert not r.significant
    assert r.ci_low < 0 < r.ci_high


def test_two_proportion_rejects_empty_group():
    with pytest.raises(ValueError):
        two_proportion_test(0, 0, 5, 100)


# --------------------------------------------------------------------------- #
# Sample size & power consistency
# --------------------------------------------------------------------------- #

def test_sample_size_gives_target_power():
    # The n from the sample-size formula should deliver ~80% power.
    p1, p2 = 0.12, 0.135
    n = sample_size_two_proportions(p1, p2, alpha=0.05, power=0.80)
    assert power_two_proportions(p1, p2, n) == pytest.approx(0.80, abs=0.01)


def test_smaller_effect_needs_larger_sample():
    n_big_effect = sample_size_two_proportions(0.12, 0.15)
    n_small_effect = sample_size_two_proportions(0.12, 0.13)
    assert n_small_effect > n_big_effect


def test_power_increases_with_sample_size():
    p1, p2 = 0.12, 0.14
    assert power_two_proportions(p1, p2, 2000) < power_two_proportions(p1, p2, 8000)


def test_sample_size_requires_effect():
    with pytest.raises(ValueError):
        sample_size_two_proportions(0.12, 0.12)


# --------------------------------------------------------------------------- #
# Geo permutation test
# --------------------------------------------------------------------------- #

def test_geo_permutation_detects_real_lift():
    rng = np.random.default_rng(0)
    holdout = rng.normal(1000, 60, 25)
    test = rng.normal(1150, 60, 25)             # a clear +15% effect
    res = geo_lift_permutation(test, holdout, n_perm=2000, seed=1)
    assert res.abs_lift > 0
    assert res.rel_lift == pytest.approx(0.15, abs=0.05)
    assert res.p_value < 0.05


def test_geo_permutation_null_is_not_significant():
    rng = np.random.default_rng(2)
    a = rng.normal(1000, 60, 25)
    b = rng.normal(1000, 60, 25)                # no true difference
    res = geo_lift_permutation(a, b, n_perm=2000, seed=3)
    assert res.p_value > 0.05


def test_geo_permutation_pvalue_in_unit_interval():
    res = geo_lift_permutation([1, 2, 3], [1, 2, 3], n_perm=500, seed=4)
    assert 0.0 < res.p_value <= 1.0
