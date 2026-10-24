"""Mean gradient magnitude MGM of an 8-bit grayscale image - a measure of its complexity (P001, E5).

Contract: ../REFERENCE.md#b13 (one level above this repository). Refusals: #spec-result. The contract is the
norm; this module does not restate it.
"""

import numpy as np

from imstudies.core.metrics import Refusal

G_MAX = 4.472  # the normalising value of the source (Bondzulic et al., 2021), B13
MAX_PIXELS = 2 ** 28  # the input limit of B13


def mgm(image):
    """MGM, dimensionless - contract ../REFERENCE.md#b13: Sobel gradients of image / 255 on inner pixels only."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise Refusal("wrong_dtype", f"expected a uint8 numpy array, got {type(image).__name__} "
                                     f"{getattr(image, 'dtype', None)}")
    if image.ndim != 2:
        raise Refusal("wrong_shape", f"expected a 2-D array, got ndim={image.ndim}")
    h, w = image.shape
    if h < 3 or w < 3:
        raise Refusal("too_small", f"{h}x{w}, each side must be >= 3")
    if h * w > MAX_PIXELS:
        raise Refusal("input_too_large", f"{h}x{w} = {h * w} pixels > {MAX_PIXELS}")
    x = image.astype(np.float64) / 255.0
    # correlation with [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] (gx) and its transpose (gy), inner pixels: no padding
    gx = (x[:-2, 2:] - x[:-2, :-2]) + 2.0 * (x[1:-1, 2:] - x[1:-1, :-2]) + (x[2:, 2:] - x[2:, :-2])
    gy = (x[2:, :-2] - x[:-2, :-2]) + 2.0 * (x[2:, 1:-1] - x[:-2, 1:-1]) + (x[2:, 2:] - x[:-2, 2:])
    return float(np.sum(np.sqrt(gx * gx + gy * gy)) / ((h - 2) * (w - 2) * G_MAX))
