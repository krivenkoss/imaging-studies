"""Paired comparison of E5 (P001): the exact two-sided Wilcoxon signed-rank test, the Holm correction, the
Hodges-Lehmann estimate and its 95 % confidence interval.

Contract: ../REFERENCE.md#b14 (one level above this repository). Refusals: #spec-result. The exact p-value is
scipy.stats.wilcoxon; its version goes into the provenance of the caller. The contract is the norm; this module
does not restate it.
"""

import math
from typing import NamedTuple

import numpy as np
from scipy import stats

from imstudies.core.metrics import Refusal

N_MIN, N_MAX = 6, 25  # B14: below 6 the exact interval does not exist at 95 %; above 25 - input_too_large


class PairedResult(NamedTuple):
    """Outputs of B14 for one comparison: the logarithmic scale and the same back in times (`ratio`)."""

    n: int
    w_plus: float
    p_value: float
    hl: float
    ci95: tuple
    k: int
    ratio: float
    ratio_ci95: tuple


def signed_rank_counts(n):
    """The exact null distribution of T+ for n: counts[t] = number of the 2**n sign vectors with T+ = t."""
    counts = [1]
    for rank in range(1, n + 1):
        grown = counts + [0] * rank
        for t, c in enumerate(counts):
            grown[t + rank] += c
        counts = grown
    return counts


def ci_k(n):
    """The largest k with P(T+ <= k - 1) <= 0.025 under the exact distribution (integer counts: 40 * c <= 2**n)."""
    k, cumulative = 0, 0
    for t, c in enumerate(signed_rank_counts(n)):
        cumulative += c
        if 40 * cumulative > 2 ** n:
            break
        k = t + 1
    return k


def _check(d):
    if not isinstance(d, np.ndarray) or d.dtype != np.float64 or d.ndim != 1 or not np.all(np.isfinite(d)):
        raise Refusal("wrong_input", "expected a 1-D finite float64 numpy array")
    if d.size < N_MIN:
        raise Refusal("too_small", f"n = {d.size} < {N_MIN}")
    if d.size > N_MAX:
        raise Refusal("input_too_large", f"n = {d.size} > {N_MAX}")
    a = np.abs(d)
    if np.any(a == 0.0) or np.unique(a).size != a.size:
        raise Refusal("ties_or_zeros", "zero or tied |d|: the exact test does not admit them")


def wilcoxon_hl(d):
    """W+, the exact two-sided p, the Hodges-Lehmann estimate and its 95 % interval of d = log r - contract #b14."""
    _check(d)
    n = d.size
    ranks = np.empty(n, dtype=np.int64)
    ranks[np.argsort(np.abs(d), kind="stable")] = np.arange(1, n + 1)
    w_plus = float(ranks[d > 0].sum())
    p_value = float(stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue)
    i, j = np.triu_indices(n)
    walsh = np.sort((d[i] + d[j]) / 2.0)
    hl = float(np.median(walsh))
    k = ci_k(n)
    ci = (float(walsh[k - 1]), float(walsh[walsh.size - k]))  # A(k), A(N + 1 - k), 1-based
    return PairedResult(n, w_plus, p_value, hl, ci, k, math.exp(hl), (math.exp(ci[0]), math.exp(ci[1])))


def holm(p):
    """Holm-adjusted p-values in the given order: p(i) -> max over j <= i of min(1, (m - j + 1) * p(j))."""
    values = [float(x) for x in p] if isinstance(p, (list, tuple)) else None
    if not values or any(not (0.0 <= x <= 1.0) for x in values):
        raise Refusal("wrong_input", "expected a non-empty list of p-values in [0, 1]")
    m = len(values)
    order = sorted(range(m), key=lambda i: values[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * values[i]))
        adjusted[i] = running
    return tuple(adjusted)
