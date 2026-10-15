"""JPEG baseline (Pillow -> libjpeg-turbo): one 8-bit grey image -> file at a quality -> decoded image.

Contract: ../REFERENCE.md#b06 (one level above this repository). In memory; no files.
"""

import io

import numpy as np
import PIL
from PIL import Image, features

from imstudies.lab.coders.still import CodedImage, check_versions, decode_gray, validate

EXPECTED_VERSIONS = {"pillow": "12.3.0", "libjpeg_turbo": "3.1.4.1"}
QUALITY_RANGE = (1, 100)  # Pillow itself takes 0 and 101; B06 does not
ENCODE_ARGS = {"optimize": False, "progressive": False}


def versions():
    return {"pillow": PIL.__version__, "libjpeg_turbo": features.version_feature("libjpeg_turbo")}


def _encode(image, quality):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image), "L").save(buf, "JPEG", quality=quality, **ENCODE_ARGS)
    return buf.getvalue()


def encode_decode(image, quality, *, max_pixels):
    """Encode `image` to JPEG baseline at `quality` and decode it - contract ../REFERENCE.md#b06."""
    validate(image, quality, max_pixels, QUALITY_RANGE)
    found = versions()
    check_versions("JPEG", EXPECTED_VERSIONS, found)
    file = _encode(image, int(quality))
    recon = decode_gray(file, image.shape, "JPEG")
    provenance = {"versions": found, "encode": {"call": "PIL.Image.fromarray(image, 'L').save(buf, 'JPEG', ...)",
                                                "quality": int(quality), **ENCODE_ARGS},
                  "decode": "PIL.Image.open(buf), mode L"}
    return CodedImage(file, recon, provenance)
