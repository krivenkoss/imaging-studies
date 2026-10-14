"""Full-reference quality metrics of 8-bit grayscale images: PSNR, PSNR-HVS-M, HaarPSI.

Contracts: PSNR - ../REFERENCE.md#b02, PSNR-HVS-M - #b03, HaarPSI - #b04 (one level
above this repository). Refusals and their statuses: #spec-result. The contracts are
the norm; this module does not restate them.
"""

import math
from typing import NamedTuple

import numpy as np
from scipy.fft import dctn
from scipy.signal import convolve2d

# Refusal reasons reachable in this module (flat list of ../REFERENCE.md#spec-result).
REFUSAL_REASONS = (
    "not_2d",
    "unsupported_dtype",
    "shape_mismatch",
    "image_too_small",
    "parameter_not_assigned",
    "image_too_large",
    "zero_weight_sum",
)


class Refusal(Exception):
    """Refusal of a core block; `reason` is a name from REFUSAL_REASONS (status `no_result`)."""

    def __init__(self, reason, detail=""):
        if reason not in REFUSAL_REASONS:
            raise ValueError(f"unknown refusal reason {reason!r}")
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class MetricResult(NamedTuple):
    """Output fields of blocks B02-B04: value, coverage (None = no selection), uncertainty."""

    value: float
    coverage: float | None
    uncertainty: None = None


# PSNR-HVS-M tables, copied from psnrhvsm.m by N. Ponomarenko (Copyright 2006;
# https://www.ponomarenko.info/psnrhvsm.m, sha256 4a373c54...41b7, see
# studies/P001_vvc-dental/STUDY.md section 6): CSFCof and MaskCof, row k, column l.
CSF = np.array([
    [1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887],
    [2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911],
    [1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555],
    [1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082],
    [1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222],
    [1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729],
    [0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803],
    [0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950],
])
MASK_COF = np.array([
    [0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874],
    [0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058],
    [0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888],
    [0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015],
    [0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866],
    [0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815],
    [0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803],
    [0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203],
])
_MASK_COF_AC = MASK_COF.copy()
_MASK_COF_AC[0, 0] = 0.0  # E(X) skips the DC coefficient

# HaarPSI constants of the definition (B04): not parameters, not in config.json.
_HAARPSI_C = 30.0
_HAARPSI_ALPHA = 4.2


def _validate(ref, dist, max_pixels, min_side):
    """Validation steps (1)-(6) of B02 (B03, B04: min_side); reads shape and dtype only."""
    for x in (ref, dist):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            raise Refusal("not_2d", f"expected a 2-D numpy array, got {type(x).__name__} "
                                    f"with ndim={getattr(x, 'ndim', None)}")
    for x in (ref, dist):
        if x.dtype != np.uint8:
            raise Refusal("unsupported_dtype", f"expected uint8, got {x.dtype}")
    if ref.shape != dist.shape:
        raise Refusal("shape_mismatch", f"{ref.shape} vs {dist.shape}")
    h, w = ref.shape
    if h < min_side or w < min_side:
        raise Refusal("image_too_small", f"{h}x{w}, each side must be >= {min_side}")
    if max_pixels is None:
        raise Refusal("parameter_not_assigned", "max_pixels (core_params.io.max_image_pixels)")
    if h * w > max_pixels:
        raise Refusal("image_too_large", f"{h}x{w} = {h * w} pixels > max_pixels = {max_pixels}")


def _sse(ref, dist):
    """Sum of squared differences as an exact Python int (int64 accumulation, B02)."""
    d = ref.astype(np.int64) - dist.astype(np.int64)
    return int(np.sum(d * d))


def psnr(ref, dist, *, max_pixels):
    """PSNR, dB, peak 255 - contract ../REFERENCE.md#b02."""
    _validate(ref, dist, max_pixels, min_side=1)
    sse = _sse(ref, dist)
    if sse == 0:
        return MetricResult(math.inf, None)
    return MetricResult(10.0 * math.log10((65025 * ref.size) / sse), None)


def _block_energy(x):
    """V(X) = var(X, ddof=1) * n over the last two axes (B03)."""
    n = x.shape[-2] * x.shape[-1]
    return np.var(x, axis=(-2, -1), ddof=1) * n


def _masking(blocks, coefs):
    """M(X) of B03 for every block: blocks and their DCT coefficients, shape (nb, 8, 8)."""
    e = np.sum(coefs * coefs * _MASK_COF_AC, axis=(1, 2))
    v = _block_energy(blocks)
    quadrants = (_block_energy(blocks[:, :4, :4]) + _block_energy(blocks[:, :4, 4:])
                 + _block_energy(blocks[:, 4:, 4:]) + _block_energy(blocks[:, 4:, :4]))
    p = np.divide(quadrants, v, out=np.zeros_like(v), where=v != 0)
    return np.sqrt(e * p) / 32.0


def _blocks8(x):
    """Non-overlapping 8x8 blocks from the top-left corner, row by row: (nb, 8, 8) float64."""
    h, w = x.shape[0] // 8, x.shape[1] // 8
    x = x[:8 * h, :8 * w].astype(np.float64)
    return x.reshape(h, 8, w, 8).transpose(0, 2, 1, 3).reshape(h * w, 8, 8)


def psnr_hvs_m(ref, dist, *, max_pixels):
    """PSNR-HVS-M, dB, window step 8 - contract ../REFERENCE.md#b03."""
    _validate(ref, dist, max_pixels, min_side=8)
    h, w = ref.shape
    coverage = (8 * (h // 8) * 8 * (w // 8)) / (h * w)
    a_blocks, b_blocks = _blocks8(ref), _blocks8(dist)
    a = dctn(a_blocks, type=2, norm="ortho", axes=(1, 2))
    b = dctn(b_blocks, type=2, norm="ortho", axes=(1, 2))
    m = np.maximum(_masking(a_blocks, a), _masking(b_blocks, b))
    u = np.abs(a - b)
    u_masked = np.maximum(u - m[:, None, None] / MASK_COF, 0.0)
    u_masked[:, 0, 0] = u[:, 0, 0]
    s = float(np.sum((u_masked * CSF) ** 2)) / (64 * a.shape[0])
    if s == 0.0:
        return MetricResult(math.inf, coverage)
    return MetricResult(10.0 * math.log10(65025.0 / s), coverage)


def _conv_same(x, h):
    """conv2(x, h, 'same') of MATLAB with zero padding (B04): scipy with 180-degree turns."""
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(h, 2), mode="same"), 2)


def _haar_coefficients(x):
    """Haar responses of scales 1..3 and orientations 0 (h_s), 1 (h_s^T), keyed (scale, ori)."""
    out = {}
    for s in (1, 2, 3):
        k = 2 ** s
        h = 2.0 ** (-s) * np.ones((k, k))
        h[: k // 2, :] = -h[: k // 2, :]
        out[s, 0] = _conv_same(x, h)
        out[s, 1] = _conv_same(x, h.T)
    return out


def haarpsi(ref, dist, *, max_pixels):
    """HaarPSI, dimensionless, grayscale branch - contract ../REFERENCE.md#b04."""
    _validate(ref, dist, max_pixels, min_side=1)
    mean2 = np.ones((2, 2)) / 4.0
    x = _conv_same(ref.astype(np.float64), mean2)[::2, ::2]
    y = _conv_same(dist.astype(np.float64), mean2)[::2, ::2]
    ca, cb = _haar_coefficients(x), _haar_coefficients(y)
    c = _HAARPSI_C
    local, weights = [], []
    for o in (0, 1):
        weights.append(np.maximum(np.abs(ca[3, o]), np.abs(cb[3, o])))
        ls = 0.0
        for s in (1, 2):
            ma, mb = np.abs(ca[s, o]), np.abs(cb[s, o])
            ls = ls + (2 * ma * mb + c) / (ma ** 2 + mb ** 2 + c)
        local.append(ls / 2)
    local = np.stack(local, axis=2)
    weights = np.stack(weights, axis=2)
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        raise Refusal("zero_weight_sum", "both images are entirely zero")
    alpha = _HAARPSI_ALPHA
    sigmoid = 1.0 / (1.0 + np.exp(-alpha * local))
    p = np.sum(sigmoid * weights) / weight_sum
    return MetricResult(float((np.log(p / (1 - p)) / alpha) ** 2), None)
