"""Shared pieces of the in-memory still-image codec wrappers B06 (JPEG), B07 (HEIF), B08 (AVIF).

Contracts: ../REFERENCE.md#b06, #b07, #b08. Validation order is the same in all three; the
quality range differs. Nothing is written to disk.
"""

import io
from typing import NamedTuple

import numpy as np
from PIL import Image

from imstudies.core.metrics import Refusal


class CodedImage(NamedTuple):
    """Outputs of B06-B08: the stored file (for CR), its decoding, provenance."""

    file: bytes
    recon: np.ndarray
    provenance: dict


def validate(image, quality, max_pixels, quality_range):
    """Validation steps (1)-(6) of B06-B08."""
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise Refusal("not_2d", f"expected a 2-D numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise Refusal("unsupported_dtype", f"expected uint8, got {image.dtype}")
    h, w = image.shape
    if h < 8 or w < 8:
        raise Refusal("image_too_small", f"{h}x{w}, each side must be >= 8")
    for name, value in (("quality", quality), ("max_pixels", max_pixels)):
        if value is None:
            raise Refusal("parameter_not_assigned", name)
    low, high = quality_range
    if isinstance(quality, (bool, np.bool_)) or not isinstance(quality, (int, np.integer)) or not low <= quality <= high:
        raise Refusal("parameter_out_of_range", f"quality={quality!r}, allowed integers {low}..{high}")
    if h * w > max_pixels:
        raise Refusal("image_too_large", f"{h}x{w} = {h * w} pixels > max_pixels = {max_pixels}")


def check_versions(codec, expected, actual):
    """Refusal in words before the call (run level, like a missing executable in B05) when a version differs."""
    wrong = {k: (expected[k], actual.get(k)) for k in expected if actual.get(k) != expected[k]}
    if wrong:
        raise RuntimeError(f"{codec}: library versions differ from the declared ones (declared, found): {wrong}")


def decode_gray(file, shape, codec):
    """Decode `file` with Pillow; a mode other than L or another shape is a failure (status `error`)."""
    with Image.open(io.BytesIO(file)) as im:
        mode = im.mode
        recon = np.asarray(im).astype(np.uint8, copy=True) if mode == "L" else None
    if recon is None:
        raise RuntimeError(f"{codec}: decoded mode {mode}, expected L")
    if recon.shape != tuple(shape):
        raise RuntimeError(f"{codec}: decoded shape {recon.shape}, expected {tuple(shape)}")
    return recon
