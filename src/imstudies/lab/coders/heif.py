"""HEIF, HEVC monochrome (pillow-heif -> libheif, x265; libde265): grey image -> file at a quality -> decoded image.

Contract: ../REFERENCE.md#b07 (one level above this repository). In memory; no files.
A file whose hvcC is not 4:0:0 is a failure, never a silent 4:2:0 substitute.
"""

import io

import numpy as np
import PIL
import pillow_heif
from PIL import Image

from imstudies.lab.coders import isobmff
from imstudies.lab.coders.still import CodedImage, check_versions, decode_gray, validate

EXPECTED_VERSIONS = {"pillow": "12.3.0", "pillow_heif": "1.8.1", "libheif": "1.23.6",
                     "x265": "x265 HEVC encoder (4.3+1-e9b8812)", "libde265": "libde265 HEVC decoder, version 1.1.3"}
QUALITY_RANGE = (0, 100)  # -1 (lossless in pillow-heif) is outside B07
ENC_PARAMS = {"preset": "slow", "tune": "ssim", "tu-intra-depth": "2"}  # libheif defaults, passed explicitly
OPTIONS = {"SAVE_NCLX_PROFILE": True, "THUMBNAILS": True, "GRID_TILE_SIZE": 0}  # pillow_heif.options, fixed


def versions():
    info = pillow_heif.libheif_info()
    return {"pillow": PIL.__version__, "pillow_heif": pillow_heif.__version__, "libheif": info["libheif"],
            "x265": info["encoders"].get("x265"), "libde265": info["decoders"].get("libde265")}


def _set_options():
    pillow_heif.register_heif_opener()
    for key, value in OPTIONS.items():
        setattr(pillow_heif.options, key, value)


def _encode(image, quality):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image), "L").save(buf, "HEIF", quality=quality, enc_params=dict(ENC_PARAMS))
    return buf.getvalue()


def encode_decode(image, quality, *, max_pixels):
    """Encode `image` to HEIF (HEVC 4:0:0) at `quality` and decode it - contract ../REFERENCE.md#b07."""
    validate(image, quality, max_pixels, QUALITY_RANGE)
    found = versions()
    check_versions("HEIF", EXPECTED_VERSIONS, found)
    _set_options()
    file = _encode(image, int(quality))
    chroma = isobmff.hevc_chroma_format_idc(file)
    if chroma != 0:
        raise RuntimeError(f"HEIF: hvcC chroma_format_idc = {chroma}, not 0 (monochrome) - a colour substitute is a failure")
    recon = decode_gray(file, image.shape, "HEIF")
    provenance = {"versions": found, "options": {k: getattr(pillow_heif.options, k) for k in OPTIONS},
                  "encode": {"call": "PIL.Image.fromarray(image, 'L').save(buf, 'HEIF', ...)", "quality": int(quality),
                             "enc_params": dict(ENC_PARAMS)},
                  "decode": "PIL.Image.open(buf) (pillow-heif opener), mode L", "hvcC_chroma_format_idc": chroma}
    return CodedImage(file, recon, provenance)
