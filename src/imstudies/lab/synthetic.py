"""Synthetic frames of the study runs: the recipe of frame E0 and its crops.

The recipe of PROMPT cycles 1 and 3: y, x = np.mgrid[0:H, 0:W]; rng = np.random.default_rng(1);
(x/2 + y/4 + rng.normal(0, 8, (H, W))).clip(0, 255).astype(np.uint8). E0 is 512 x 512; the crops
of cycle 3 (8 x 8, 21 x 13, 517 x 389) are top-left crops of the recipe on 517 x 512. run_v2a.py
keeps its own copy as the record of cycle 3; parity is tested against its run file.
"""

import numpy as np

CROP_BASE = (517, 512)


def e0_recipe(h, w):
    """The E0 recipe on (h, w), seed 1."""
    y, x = np.mgrid[0:h, 0:w]
    return (x / 2 + y / 4 + np.random.default_rng(1).normal(0, 8, (h, w))).clip(0, 255).astype(np.uint8)


def e0_frame():
    """Frame E0, 512 x 512."""
    return e0_recipe(512, 512)


def e0_crop(h, w):
    """Top-left (h, w) crop of the recipe on 517 x 512 (the synthetic sizes of cycle 3)."""
    if h > CROP_BASE[0] or w > CROP_BASE[1]:
        raise ValueError(f"crop {h}x{w} exceeds the base {CROP_BASE[0]}x{CROP_BASE[1]}")
    return np.ascontiguousarray(e0_recipe(*CROP_BASE)[:h, :w])
