"""Reading 8-bit grey images for study runs: palette or grey files to brightness, standard test images.

Moved from studies/P001_vvc-dental/run_v1.py (cycle 4, read_gray and test_set); run_v1.py keeps
its own copies as the record of cycle 1. Pixels are returned as read: no scaling, no crop.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from imstudies.lab.runs import RunRefusal, sha256_file


def read_gray(path):
    """8-bit brightness of a palette or grey image: (uint8 array, how it was read)."""
    path = Path(path)
    with Image.open(path) as im:
        mode = im.mode
        idx = np.asarray(im)
        palette = im.getpalette() if mode == "P" else None
    if mode == "L":
        if path.suffix.lower() == ".bmp":
            head = path.read_bytes()[:30]
            if head[:2] != b"BM" or int.from_bytes(head[28:30], "little") != 8:
                raise RunRefusal(f"{path.name}: not an 8-bit BMP")
        return idx.astype(np.uint8), "Pillow mode L (grey ramp palette or grey file): value = brightness"
    if mode == "P":
        pal = np.array(palette, dtype=np.int64).reshape(-1, 3)
        used = np.unique(idx)
        if used.max() >= len(pal) or not (pal[used, 0] == pal[used, 1]).all() or not (pal[used, 1] == pal[used, 2]).all():
            raise RunRefusal("a palette image is not grey on the indices it uses")
        return pal[idx, 0].astype(np.uint8), "Pillow mode P: brightness = grey palette entry of the index"
    raise RunRefusal(f"unsupported image mode {mode}")


def load_test_images(images_root, names, shape):
    """Test images `names` of `images_root`/BMP, each of `shape` (H, W), as run records with the pixels."""
    images = []
    for name in names:
        path = Path(images_root) / "BMP" / name
        if not path.is_file():
            raise RunRefusal(f"test image missing: {path}")
        arr, how = read_gray(path)
        if arr.shape != tuple(shape):
            raise RunRefusal(f"{name}: expected {shape[0]}x{shape[1]}, got {arr.shape}")
        images.append({"code": path.stem.lower(), "class": "test", "source": f"BMP/{name}",
                       "source_sha256": sha256_file(path), "full_shape": list(arr.shape), "crop_top_left": [0, 0],
                       "read": how, "array": arr})
    return images
