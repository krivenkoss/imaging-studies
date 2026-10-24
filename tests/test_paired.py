"""Core tests of B14 (imstudies.core.paired) - contract ../REFERENCE.md#b14.

Oracles by another path: the exact two-sided p-value by enumeration of all 2**n sign vectors (n = 6, 10, 16; to
1e-12); k by the same enumeration of T+; the 95 % interval by counting T+(d - theta) on a grid of theta between
the Walsh averages (theta is inside the open interval exactly when k <= T+ <= N - k); W+ and the Hodges-Lehmann
estimate by explicit loops; Holm on a worked example and the cap at 1. Every refusal by its exact reason.
"""

import itertools
import math
import statistics

import numpy as np
import pytest

from imstudies.core import paired
from imstudies.core.metrics import Refusal

pytestmark = pytest.mark.core


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    out = [0] * len(values)
    for rank, i in enumerate(order, start=1):
        out[i] = rank
    return out


def t_plus(values):
    return sum(r for r, v in zip(ranks(values), values) if v > 0)


def sign_sums(n):
    """T+ of every one of the 2**n sign vectors on the ranks 1..n."""
    return [sum(r for r, plus in zip(range(1, n + 1), signs) if plus) for signs in itertools.product((0, 1), repeat=n)]


def enumerated_p(d):
    n, big_n = len(d), len(d) * (len(d) + 1) // 2
    t, sums = t_plus(d), sign_sums(len(d))
    return sum(1 for s in sums if abs(2 * s - big_n) >= abs(2 * t - big_n)) / 2 ** n


def enumerated_k(n):
    sums = sign_sums(n)
    return max(k for k in range(0, n * (n + 1) // 2) if sum(1 for s in sums if s <= k - 1) / 2 ** n <= 0.025)


def walsh(d):
    return sorted((d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d)))


def sample(n, seed):
    return np.random.default_rng(seed).normal(0.15, 0.3, n)


@pytest.mark.parametrize("n", [6, 10, 16])
def test_p_w_plus_hl_and_k_against_enumeration(n):
    d = sample(n, n)
    got = paired.wilcoxon_hl(d)
    assert got.n == n and got.w_plus == t_plus(list(d))
    assert abs(got.p_value - enumerated_p(list(d))) <= 1e-12
    assert got.hl == statistics.median(walsh(list(d)))
    assert got.k == enumerated_k(n) == paired.ci_k(n)
    assert got.ratio == math.exp(got.hl) and got.ratio_ci95 == (math.exp(got.ci95[0]), math.exp(got.ci95[1]))


@pytest.mark.parametrize("n", [6, 10, 16])
def test_interval_membership_by_counting_t_plus(n):
    d = list(sample(n, 100 + n))
    got = paired.wilcoxon_hl(np.array(d))
    k, big_n, averages = enumerated_k(n), n * (n + 1) // 2, walsh(d)
    thetas = [averages[0] - 1] + [(a + b) / 2 for a, b in zip(averages, averages[1:]) if a != b] + [averages[-1] + 1]
    for theta in thetas:
        inside = k <= t_plus([x - theta for x in d]) <= big_n - k
        assert inside == (got.ci95[0] < theta < got.ci95[1]), theta


def test_coverage_of_k_is_the_largest_at_95_percent():
    for n in (6, 10, 16):
        sums, k = sign_sums(n), paired.ci_k(n)
        tail = lambda kk: sum(1 for s in sums if s <= kk - 1) / 2 ** n  # noqa: E731
        assert tail(k) <= 0.025 < tail(k + 1)


def test_all_positive_six_gives_the_smallest_p():
    got = paired.wilcoxon_hl(np.array([0.11, 0.23, 0.35, 0.47, 0.59, 0.71]))
    assert got.w_plus == 21 and got.p_value == pytest.approx(2 / 64, abs=1e-12) and got.k == 1
    assert got.ci95 == (0.11, 0.71)  # A(1) and A(21): the smallest and the largest Walsh averages


def test_holm_worked_example_and_cap():
    # by hand: sorted 0.005 * 4 = 0.02; 0.01 * 3 = 0.03; 0.03 * 2 = 0.06; 0.04 * 1 = 0.04 -> 0.06 (monotone)
    assert paired.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx((0.03, 0.06, 0.06, 0.02), abs=1e-15)
    assert paired.holm([0.5, 0.6]) == (1.0, 1.0)
    assert paired.holm([0.2]) == (0.2,)


@pytest.mark.parametrize("value, reason", [
    ([0.1] * 8, "wrong_input"),
    (np.arange(8), "wrong_input"),
    (np.ones((3, 3)), "wrong_input"),
    (np.array([0.1, 0.2, np.nan, 0.4, 0.5, 0.6]), "wrong_input"),
    (np.array([0.1, 0.2, np.inf, 0.4, 0.5, 0.6]), "wrong_input"),
    (np.array([0.1, 0.2, 0.3, 0.4, 0.5]), "too_small"),
    (np.linspace(0.1, 2.6, 26), "input_too_large"),
    (np.array([0.1, 0.0, 0.3, 0.4, 0.5, 0.6]), "ties_or_zeros"),
    (np.array([0.1, -0.1, 0.3, 0.4, 0.5, 0.6]), "ties_or_zeros"),
])
def test_refusals_by_their_exact_reason(value, reason):
    with pytest.raises(Refusal) as caught:
        paired.wilcoxon_hl(value)
    assert caught.value.reason == reason


@pytest.mark.parametrize("value", [[], [1.2], [-0.1], [0.5, float("nan")], "0.5"])
def test_holm_refuses_wrong_p(value):
    with pytest.raises(Refusal) as caught:
        paired.holm(value)
    assert caught.value.reason == "wrong_input"
