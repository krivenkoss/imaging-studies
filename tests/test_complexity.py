"""Core tests of B13 (imstudies.core.complexity.mgm) - contract ../REFERENCE.md#b13.

The oracle is written again with explicit loops over the inner pixels and the Sobel kernel and its transpose as
tables (relative error <= 1e-12); closed forms: a constant gives 0, a step 0/255 across columns gives
8 / ((W - 2) * 4.472), a slope of one grey level per column gives 8 / (255 * 4.472). Every refusal by its exact
reason, in the declared order; the input limit is refused on a broadcast view, before any array of that size.
"""

import math

import numpy as np
import pytest

from imstudies.core.complexity import mgm
from imstudies.core.metrics import Refusal

pytestmark = pytest.mark.core

KERNEL_X = ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))
G_MAX = 4.472  # the contract's value


def oracle(image):
    h, w = image.shape
    total = 0.0
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            gx = gy = 0.0
            for a in range(3):
                for b in range(3):
                    v = int(image[i - 1 + a][j - 1 + b]) / 255.0
                    gx += KERNEL_X[a][b] * v
                    gy += KERNEL_X[b][a] * v
            total += math.hypot(gx, gy)
    return total / ((h - 2) * (w - 2) * G_MAX)


def test_constant_step_and_slope_in_closed_form():
    assert mgm(np.full((5, 7), 137, np.uint8)) == 0.0
    step = np.zeros((7, 9), np.uint8)
    step[:, 4:] = 255
    assert mgm(step) == pytest.approx(8 / ((9 - 2) * G_MAX), rel=1e-12) == oracle(step)
    slope = np.tile(np.arange(20, dtype=np.uint8), (6, 1))
    assert mgm(slope) == pytest.approx(8 / (255 * G_MAX), rel=1e-12)
    assert mgm(slope) == pytest.approx(oracle(slope), rel=1e-12)


@pytest.mark.parametrize("shape", [(3, 3), (3, 17), (31, 23), (64, 64)])
def test_random_images_against_the_oracle(shape):
    image = np.random.default_rng(sum(shape)).integers(0, 256, shape).astype(np.uint8)
    assert mgm(image) == pytest.approx(oracle(image), rel=1e-12)


def test_edges_are_not_padded():
    """Only the inner pixels count: a frame of other values around an image changes nothing inside, so the sum
    over the inner pixels of the framed image differs from that of the image by the ring only."""
    image = np.random.default_rng(3).integers(0, 256, (12, 15)).astype(np.uint8)
    inner = oracle(image) * (10 * 13 * G_MAX)
    assert mgm(image) * (10 * 13 * G_MAX) == pytest.approx(inner, rel=1e-12)
    framed = np.pad(image, 1, constant_values=0)
    ring = sum(math.hypot(*g) for g in ring_gradients(framed))
    assert mgm(framed) * (12 * 15 * G_MAX) == pytest.approx(inner + ring, rel=1e-12)


def ring_gradients(image):
    """Gradients at the inner pixels of `image` that lie on its second ring (next to the border)."""
    h, w = image.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if i in (1, h - 2) or j in (1, w - 2):
                gx = sum(KERNEL_X[a][b] * int(image[i - 1 + a][j - 1 + b]) / 255.0 for a in range(3) for b in range(3))
                gy = sum(KERNEL_X[b][a] * int(image[i - 1 + a][j - 1 + b]) / 255.0 for a in range(3) for b in range(3))
                yield gx, gy


@pytest.mark.parametrize("value, reason", [
    (np.zeros((5, 5), np.float64), "wrong_dtype"),
    ([[0] * 5] * 5, "wrong_dtype"),
    (np.zeros(9, np.float32), "wrong_dtype"),            # the dtype is looked at before the shape
    (np.zeros((4, 4, 3), np.uint8), "wrong_shape"),
    (np.zeros(2, np.uint8), "wrong_shape"),              # the shape before the size
    (np.zeros((2, 9), np.uint8), "too_small"),
    (np.zeros((9, 2), np.uint8), "too_small"),
    (np.broadcast_to(np.uint8(0), (2 ** 14 + 1, 2 ** 14)), "input_too_large"),
])
def test_refusals_by_their_exact_reason(value, reason):
    with pytest.raises(Refusal) as caught:
        mgm(value)
    assert caught.value.reason == reason


def test_the_limit_itself_is_accepted():
    assert mgm(np.broadcast_to(np.uint8(9), (2 ** 14, 2 ** 14))[:3, :]) == 0.0  # a small view of a huge one
