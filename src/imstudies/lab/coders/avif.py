"""AVIF, AV1 4:0:0 (Pillow -> libavif, aom; dav1d): one grey image -> file at a quality -> decoded image.

Contract: ../REFERENCE.md#b08 (one level above this repository). In memory; no files.
max_threads is 1: the file depends on the number of encoder threads, and Pillow's default is
the number of cores of the machine. A file whose av1C is not monochrome is a failure.
"""

import io

import numpy as np
import PIL
from PIL import Image, _avif, features

from imstudies.lab.coders import isobmff
from imstudies.lab.coders.still import CodedImage, check_versions, decode_gray, validate

EXPECTED_VERSIONS = {"pillow": "12.3.0", "libavif": "1.4.2", "aom": "3.14.1", "dav1d": "1.5.3"}
QUALITY_RANGE = (0, 100)
ENCODE_ARGS = {"subsampling": "4:0:0", "speed": 6, "codec": "aom", "max_threads": 1}


def codec_versions():
    """'dav1d [dec]:1.5.3-0-gb546257, aom [enc]:3.14.1' -> {'dav1d': '1.5.3-0-gb546257', 'aom': '3.14.1'}."""
    found = {}
    for part in _avif.codec_versions().split(","):
        name, _, version = part.strip().partition(":")
        found[name.split()[0]] = version.strip()
    return found


def versions():
    codecs = codec_versions()
    return {"pillow": PIL.__version__, "libavif": features.version("avif"), "aom": codecs.get("aom"),
            "dav1d": (codecs.get("dav1d") or "").split("-")[0] or None}


def _encode(image, quality):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image), "L").save(buf, "AVIF", quality=quality, **ENCODE_ARGS)
    return buf.getvalue()


def encode_decode(image, quality, *, max_pixels):
    """Encode `image` to AVIF (AV1 4:0:0) at `quality` and decode it - contract ../REFERENCE.md#b08."""
    validate(image, quality, max_pixels, QUALITY_RANGE)
    found = versions()
    check_versions("AVIF", EXPECTED_VERSIONS, found)
    file = _encode(image, int(quality))
    monochrome = isobmff.av1_monochrome(file)
    if monochrome != 1:
        raise RuntimeError(f"AVIF: av1C monochrome = {monochrome}, not 1 - a colour substitute is a failure")
    recon = decode_gray(file, image.shape, "AVIF")
    provenance = {"versions": found, "codec_versions": codec_versions(),
                  "encode": {"call": "PIL.Image.fromarray(image, 'L').save(buf, 'AVIF', ...)", "quality": int(quality),
                             **ENCODE_ARGS},
                  "decode": "PIL.Image.open(buf), mode L", "av1C_monochrome": monochrome}
    return CodedImage(file, recon, provenance)
