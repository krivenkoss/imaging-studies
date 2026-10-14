"""Core tests of blocks B02-B04 (../REFERENCE.md#b02, #b03, #b04): imstudies.core.metrics.

Truth comes from outside the implementation: analytic SSE and the decimal oracle
(imstudies.oracle.psnr) for PSNR; values of the authors' code stored as bytes in
tests/vectors/metrics_reference.json for PSNR-HVS-M and HaarPSI.

Tolerances, declared before the results:
- V1-a (B02): SSE exact; PSNR bit-exact where 65025*N/SSE = 10**k, else <= 2 ulp of the oracle.
- vectors: |d PSNR-HVS-M| <= 1e-9 dB against psnrhvsm.m; |d HaarPSI| <= 1e-12 against
  haarPsi.py and HaarPSI.m. Tighter than V1-b/V1-c (0.01 dB, 1e-4): the expected
  difference is rounding (~1e-12 dB, ~1e-14), so anything above these bounds is a defect.
"""

import base64
import json
import math
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from imstudies.core.metrics import (
    CSF,
    MASK_COF,
    REFUSAL_REASONS,
    Refusal,
    _sse,
    haarpsi,
    psnr,
    psnr_hvs_m,
)
from imstudies.oracle import psnr as oracle

pytestmark = pytest.mark.core

VECTORS = json.loads((Path(__file__).parent / "vectors" / "metrics_reference.json").read_text(encoding="utf-8"))
METRICS = {"psnr": (psnr, 1), "psnr_hvs_m": (psnr_hvs_m, 8), "haarpsi": (haarpsi, 1)}


def _ulps(value, exact):
    return abs(Decimal(value) - exact) / Decimal(math.ulp(float(exact)))


def _pair_with_differences(shape, diffs):
    """ref = 0 (or 255 where the difference is negative), dist differs by diffs[(i, j)] at given pixels."""
    ref = np.zeros(shape, np.uint8)
    dist = np.zeros(shape, np.uint8)
    for (i, j), d in diffs.items():
        if d >= 0:
            dist[i, j] = d
        else:
            ref[i, j] = -d
    return ref, dist


# --- V1-a: PSNR on analytic cases --------------------------------------------------------------

# (shape, {pixel: difference}, analytic SSE, exact PSNR in dB) where 65025*N/SSE = 10**k
POWER_OF_TEN = [
    ((1, 1), {(0, 0): 255}, 65025, 0),
    ((2, 5), {(1, 3): 255}, 65025, 10),
    ((10, 10), {(4, 7): -255}, 65025, 20),
    ((4, 25), {(0, j): 255 for j in range(10)}, 650250, 10),
    ((25, 40), {(24, 39): 255}, 65025, 30),
    ((100, 100), {(0, 0): -255}, 65025, 40),
    ((1000, 1000), {(500, 1): 255}, 65025, 60),
    ((10, 10), {(i, j): 255 for i in range(10) for j in range(10)}, 6502500, 0),
]


@pytest.mark.parametrize("shape, diffs, sse, db", POWER_OF_TEN)
def test_psnr_bit_exact_at_powers_of_ten(shape, diffs, sse, db):
    ref, dist = _pair_with_differences(shape, diffs)
    assert _sse(ref, dist) == sse
    assert oracle.sse(ref.tolist(), dist.tolist()) == sse
    assert oracle.psnr(ref.tolist(), dist.tolist()) == db
    assert psnr(ref, dist, max_pixels=ref.size).value == float(db)


def _constant(shape, d):
    return np.full(shape, 100, np.uint8), np.full(shape, 100 + d, np.uint8)


def _checkerboard(shape, d):
    ref = np.full(shape, 128, np.uint8)
    sign = np.where((np.indices(shape).sum(axis=0) % 2) == 0, 1, -1)
    return ref, (128 + sign * d).astype(np.uint8)


ANALYTIC = (
    [("constant", (64, 48), d, d * d * 64 * 48) for d in (1, 2, 3, 7, 16, 100, 127)]
    + [("checkerboard", (33, 17), d, d * d * 33 * 17) for d in (1, 5, 64, 127)]
)


@pytest.mark.parametrize("kind, shape, d, sse", ANALYTIC)
def test_psnr_within_two_ulp_on_analytic_sse(kind, shape, d, sse):
    ref, dist = (_constant if kind == "constant" else _checkerboard)(shape, d)
    assert _sse(ref, dist) == sse
    exact = oracle.psnr(ref.tolist(), dist.tolist())
    assert oracle.sse(ref.tolist(), dist.tolist()) == sse
    assert _ulps(psnr(ref, dist, max_pixels=ref.size).value, exact) <= 2


def test_psnr_within_two_ulp_on_random_pairs():
    rng = np.random.default_rng(11)
    for _ in range(200):
        h, w = (int(v) for v in rng.integers(1, 70, 2))
        ref = rng.integers(0, 256, (h, w), dtype=np.uint8)
        step = int(rng.integers(1, 80))
        dist = np.clip(ref.astype(np.int64) + rng.integers(-step, step + 1, (h, w)), 0, 255).astype(np.uint8)
        exact = oracle.psnr(ref.tolist(), dist.tolist())
        assert _sse(ref, dist) == oracle.sse(ref.tolist(), dist.tolist())
        if exact.is_infinite():
            assert psnr(ref, dist, max_pixels=ref.size).value == math.inf
        else:
            assert _ulps(psnr(ref, dist, max_pixels=ref.size).value, exact) <= 2


@pytest.mark.xfail(strict=True, reason="B02 formula exceeds 2 ulp near 0 dB: log10 of a quotient near 1 "
                                       "(reported to the owner, cycle 1; contract not edited)")
def test_psnr_two_ulp_near_zero_db():
    ref = np.zeros((512, 512), np.uint8)
    dist = np.full((512, 512), 255, np.uint8)
    dist.flat[262100:] = 0
    dist.flat[262100] = 1
    exact = oracle.psnr(ref.tolist(), dist.tolist())
    assert _ulps(psnr(ref, dist, max_pixels=ref.size).value, exact) <= 2


# --- identical images and outputs ----------------------------------------------------------------

@pytest.mark.parametrize("shape", [(8, 8), (21, 13), (512, 512)])
def test_identical_images(shape):
    img = np.random.default_rng(3).integers(0, 256, shape, dtype=np.uint8)
    assert psnr(img, img.copy(), max_pixels=img.size).value == math.inf
    assert psnr_hvs_m(img, img.copy(), max_pixels=img.size).value == math.inf
    assert abs(haarpsi(img, img.copy(), max_pixels=img.size).value - 1.0) <= 1e-12


def test_output_fields():
    ref, dist = _constant((21, 13), 3)
    p, h, s = (f(ref, dist, max_pixels=ref.size) for f in (psnr, psnr_hvs_m, haarpsi))
    assert p.coverage is None and s.coverage is None
    assert h.coverage == (16 * 8) / (21 * 13)
    assert psnr_hvs_m(*_constant((16, 24), 3), max_pixels=384).coverage == 1.0
    assert p.uncertainty is None and h.uncertainty is None and s.uncertainty is None
    assert all(isinstance(r.value, float) for r in (p, h, s))


def test_psnr_hvs_m_ignores_partial_strips_only():
    rng = np.random.default_rng(5)
    ref = rng.integers(0, 256, (21, 13), dtype=np.uint8)
    dist = np.clip(ref.astype(int) + rng.integers(-9, 10, ref.shape), 0, 255).astype(np.uint8)
    base = psnr_hvs_m(ref, dist, max_pixels=ref.size).value
    outside = dist.copy()
    outside[16:, :] = 0
    outside[:, 8:] = 255
    assert psnr_hvs_m(ref, outside, max_pixels=ref.size).value == base
    inside = dist.copy()
    inside[15, 7] = 255 - inside[15, 7]
    assert psnr_hvs_m(ref, inside, max_pixels=ref.size).value != base


# --- authors' code: stored reference vectors -------------------------------------------------------

def _vector(case, key):
    return np.frombuffer(base64.b64decode(case[key]), np.uint8).reshape(case["shape"])


@pytest.mark.parametrize("case", [c for c in VECTORS["cases"] if c["psnrhvsm_m"] is not None], ids=lambda c: c["id"])
def test_psnr_hvs_m_matches_psnrhvsm_m(case):
    ref, dist = _vector(case, "ref"), _vector(case, "dist")
    assert abs(psnr_hvs_m(ref, dist, max_pixels=ref.size).value - case["psnrhvsm_m"]) <= 1e-9


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda c: c["id"])
def test_haarpsi_matches_haarpsi_py_and_m(case):
    ref, dist = _vector(case, "ref"), _vector(case, "dist")
    value = haarpsi(ref, dist, max_pixels=ref.size).value
    assert abs(value - case["haarpsi_py"]) <= 1e-12
    assert abs(value - case["haarpsi_m"]) <= 1e-12


def test_tables_shape():
    for table in (CSF, MASK_COF):
        assert table.shape == (8, 8) and table.dtype == np.float64 and (table > 0).all()


# --- refusals: both sides of every decision --------------------------------------------------------

def _good(min_side):
    rng = np.random.default_rng(9)
    ref = rng.integers(0, 256, (min_side + 3, min_side + 5), dtype=np.uint8)
    return ref, np.clip(ref.astype(int) + 4, 0, 255).astype(np.uint8)


def _refusal(fn, *args, **kwargs):
    with pytest.raises(Refusal) as info:
        fn(*args, **kwargs)
    return info.value.reason


@pytest.mark.parametrize("name", METRICS)
def test_not_2d(name):
    fn, m = METRICS[name]
    ref, dist = _good(m)
    assert _refusal(fn, ref[None], dist[None], max_pixels=ref.size) == "not_2d"
    assert _refusal(fn, ref, dist.ravel(), max_pixels=ref.size) == "not_2d"
    assert _refusal(fn, ref.tolist(), dist, max_pixels=ref.size) == "not_2d"
    assert math.isfinite(fn(ref, dist, max_pixels=ref.size).value)


@pytest.mark.parametrize("name", METRICS)
@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.float64, np.bool_])
def test_unsupported_dtype(name, dtype):
    fn, m = METRICS[name]
    ref, dist = _good(m)
    assert _refusal(fn, ref.astype(dtype), dist, max_pixels=ref.size) == "unsupported_dtype"
    assert _refusal(fn, ref, dist.astype(dtype), max_pixels=ref.size) == "unsupported_dtype"
    assert math.isfinite(fn(ref, dist, max_pixels=ref.size).value)


@pytest.mark.parametrize("name", METRICS)
def test_shape_mismatch(name):
    fn, m = METRICS[name]
    ref, dist = _good(m)
    for other in (dist[:-1], dist[:, :-1], dist[:-1, :-1], dist.T):
        assert _refusal(fn, ref, other, max_pixels=ref.size) == "shape_mismatch"
    assert math.isfinite(fn(ref, dist, max_pixels=ref.size).value)


@pytest.mark.parametrize("name", METRICS)
def test_image_too_small(name):
    fn, m = METRICS[name]
    rng = np.random.default_rng(2)
    for shape in [(m - 1, m + 2), (m + 2, m - 1), (0, 0)]:
        z = np.zeros(shape, np.uint8)
        assert _refusal(fn, z, z.copy(), max_pixels=10**6) == "image_too_small"
    ref = rng.integers(1, 255, (m, m), dtype=np.uint8)
    assert math.isfinite(fn(ref, ref + 1, max_pixels=ref.size).value)


@pytest.mark.parametrize("name", METRICS)
def test_parameter_not_assigned(name):
    fn, m = METRICS[name]
    ref, dist = _good(m)
    assert _refusal(fn, ref, dist, max_pixels=None) == "parameter_not_assigned"
    assert math.isfinite(fn(ref, dist, max_pixels=ref.size).value)


@pytest.mark.parametrize("name", METRICS)
def test_image_too_large_at_the_boundary(name):
    fn, m = METRICS[name]
    ref, dist = _good(m)
    assert _refusal(fn, ref, dist, max_pixels=ref.size - 1) == "image_too_large"
    assert math.isfinite(fn(ref, dist, max_pixels=ref.size).value)


@pytest.mark.parametrize("name", METRICS)
def test_image_too_large_before_allocation(name):
    fn, _ = METRICS[name]
    # 1e10-pixel views with no memory behind them: a check after allocation would exhaust memory.
    huge = np.broadcast_to(np.uint8(7), (100_000, 100_000))
    assert _refusal(fn, huge, huge, max_pixels=10**6) == "image_too_large"


def test_zero_weight_sum():
    zeros = np.zeros((16, 16), np.uint8)
    assert _refusal(haarpsi, zeros, zeros.copy(), max_pixels=256) == "zero_weight_sum"
    one = zeros.copy()
    one[15, 15] = 1
    assert math.isfinite(haarpsi(zeros, one, max_pixels=256).value)
    assert math.isfinite(haarpsi(one, zeros, max_pixels=256).value)


@pytest.mark.parametrize("name", METRICS)
def test_validation_order(name):
    fn, m = METRICS[name]
    ref, dist = _good(m)
    f64 = ref.astype(np.float64)
    assert _refusal(fn, f64[None], dist[:-1], max_pixels=None) == "not_2d"
    assert _refusal(fn, f64, dist[:-1], max_pixels=None) == "unsupported_dtype"
    small = np.zeros((m - 1 if m > 1 else 0, 3), np.uint8)
    assert _refusal(fn, small, dist, max_pixels=None) == "shape_mismatch"
    assert _refusal(fn, small, small.copy(), max_pixels=None) == "image_too_small"
    huge = np.broadcast_to(np.uint8(7), (100_000, 100_000))
    assert _refusal(fn, huge, huge, max_pixels=None) == "parameter_not_assigned"


def test_refusal_reasons_are_the_flat_list():
    spec_result = {"not_2d", "unsupported_dtype", "shape_mismatch", "image_too_small",
                   "parameter_not_assigned", "image_too_large", "zero_weight_sum", "internal_error"}
    assert set(REFUSAL_REASONS) == spec_result - {"internal_error"}
    assert Refusal("not_2d").reason == "not_2d"
    with pytest.raises(ValueError):
        Refusal("no_such_reason")
