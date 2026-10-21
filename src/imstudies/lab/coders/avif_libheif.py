"""AVIF through libheif (pillow-heif 0.22.0, separate environment): grey image -> file at a quality -> decoded image.

Contract: ../REFERENCE.md#b12 (one level above this repository). For P001 it replaces B08 (AVIF of Pillow,
libavif), which stays for the repetition of A05. The coding runs in the separate environment of the key
avif_libheif_python of lab_paths.json: `python -I avif_libheif_exec.py <work dir>` through run_tool, files
exchanged in a temporary folder under work_root and removed after the call; folders of work_root that the
call itself created are removed after it when empty, so nothing is left under cache_root. One call codes one
image at one quality or at a list of qualities (one start of the environment). The aom parameters are libheif's own
defaults passed explicitly, the number of threads fixed: the file depends on it, and libheif's default is the
number of cores of the machine. A file whose av1C is not monochrome is a failure.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from imstudies.core.metrics import Refusal
from imstudies.lab.coders import isobmff
from imstudies.lab.coders.process import run_tool
from imstudies.lab.coders.still import CodedImage, check_versions
from imstudies.lab.coders.vtm import check_timeout_s

EXEC = Path(__file__).with_name("avif_libheif_exec.py")
EXPECTED_VERSIONS = {"python": "3.13.15", "pillow_heif": "0.22.0", "pillow": "13.0.0", "numpy": "2.5.4",
                     "libheif": "1.19.7", "aom_encoder": "AOMedia Project AV1 Encoder v3.12.0",
                     "aom_decoder": "AOMedia Project AV1 Decoder v3.12.0"}
QUALITY_RANGE = (0, 100)
ENC_PARAMS = {"speed": "6", "tune": "ssim", "threads": "4"}  # libheif 1.19.7 encoder_aom defaults; threads fixed
OPTIONS = {"SAVE_NCLX_PROFILE": True, "THUMBNAILS": True, "QUALITY": None, "SAVE_HDR_TO_12_BIT": False,
           "DECODE_THREADS": 4, "PREFERRED_ENCODER": {"AVIF": "", "HEIF": ""},
           "PREFERRED_DECODER": {"AVIF": "", "HEIF": ""}}  # pillow_heif.options, set on every call
INTERNAL = ("libheif 1.19.7 plugins/encoder_aom.cc, not parameters: usage all-intra; "
            "cq_level = ((100 - quality) * 63 + 50) / 100 (integer division), end usage AOM_Q")


def _validate(image, quality, max_pixels, timeout_s):
    """Validation steps (1)-(6) of B12; returns the list of qualities."""
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise Refusal("not_2d", f"expected a 2-D numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise Refusal("unsupported_dtype", f"expected uint8, got {image.dtype}")
    h, w = image.shape
    if h < 8 or w < 8:
        raise Refusal("image_too_small", f"{h}x{w}, each side must be >= 8")
    for name, value in (("quality", quality), ("max_pixels", max_pixels), ("timeout_s", timeout_s)):
        if value is None:
            raise Refusal("parameter_not_assigned", name)
    qualities = list(quality) if isinstance(quality, (list, tuple)) else [quality]
    low, high = QUALITY_RANGE
    bad = [q for q in qualities if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer))
           or not low <= q <= high]
    if not qualities or bad:
        raise Refusal("parameter_out_of_range", f"quality={quality!r}, allowed: integers {low}..{high} or a non-empty "
                                                f"list of them")
    check_timeout_s(timeout_s)
    if h * w > max_pixels:
        raise Refusal("image_too_large", f"{h}x{w} = {h * w} pixels > max_pixels = {max_pixels}")
    return [int(q) for q in qualities]


def encode_decode(image, quality, *, max_pixels, timeout_s, python, work_root):
    """Encode `image` to AVIF through libheif at `quality` (an int or a list) and decode it - contract B12.

    Returns a CodedImage for an int, a list of CodedImage (in the order of the list) for a list."""
    qualities = _validate(image, quality, max_pixels, timeout_s)
    if python is None:
        raise FileNotFoundError("avif_libheif_python is not set in lab_paths.json: the environment of B12 is unknown")
    python = Path(python)
    if not python.is_file():
        raise FileNotFoundError(f"the python of the B12 environment is not found: {python}")
    h, w = image.shape
    created, folder = [], Path(work_root)  # folders this call creates: removed after it if empty, deepest first
    while not folder.exists():
        created.append(folder)
        folder = folder.parent
    Path(work_root).mkdir(parents=True, exist_ok=True)
    try:
        return _code(image, quality, qualities, python, work_root, timeout_s, h, w)
    finally:
        for folder in created:
            try:
                folder.rmdir()
            except OSError:
                break


def _code(image, quality, qualities, python, work_root, timeout_s, h, w):
    with tempfile.TemporaryDirectory(dir=work_root, prefix="avif_libheif_") as work:
        work = Path(work)
        np.save(work / "image.npy", np.ascontiguousarray(image))
        (work / "request.json").write_text(json.dumps({"qualities": qualities, "enc_params": ENC_PARAMS,
                                                       "options": OPTIONS}), encoding="utf-8")
        cmd = [str(python), "-I", str(EXEC), str(work)]
        stdout = run_tool(cmd, timeout_s)
        result_path = work / "result.json"
        if not result_path.is_file():
            raise RuntimeError(f"the B12 executor wrote no result.json; output: {stdout[-600:]}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        check_versions("AVIF (libheif)", EXPECTED_VERSIONS, result["versions"])
        coded = []
        for item, q in zip(result["items"], qualities, strict=True):
            file = (work / f"q{q:03d}.avif").read_bytes()
            recon = np.load(work / f"q{q:03d}_recon.npy")
            monochrome, ispe = isobmff.av1_monochrome(file), isobmff.ispe(file)
            if monochrome != 1:
                raise RuntimeError(f"AVIF (libheif) q {q}: av1C monochrome = {monochrome}, not 1 - a colour substitute "
                                   f"is a failure")
            if item["mode"] != "L" or recon.dtype != np.uint8 or recon.shape != (h, w):
                raise RuntimeError(f"AVIF (libheif) q {q}: decoded mode {item['mode']} {recon.dtype} {recon.shape}, "
                                   f"expected L uint8 {(h, w)}")
            if ispe != (w, h):
                raise RuntimeError(f"AVIF (libheif) q {q}: ispe {ispe}, expected {(w, h)}")
            provenance = {"versions": result["versions"], "libheif_info": result["libheif_info"],
                          "options": result["options"], "enc_params": dict(ENC_PARAMS), "internal": INTERNAL,
                          "encode": {"call": "pillow_heif.from_pillow(Image.fromarray(image, 'L')).save(buf, "
                                             "format='AVIF', quality=q, enc_params=...)", "quality": q},
                          "decode": "pillow_heif.open_heif(buf, convert_hdr_to_8bit=False)[0], mode L",
                          "av1C_monochrome": monochrome, "executor": EXEC.name,
                          "executor_sha256": hashlib.sha256(EXEC.read_bytes()).hexdigest(),
                          "command": [Path(cmd[0]).name, *cmd[1:2], EXEC.name, "<work dir>"]}
            coded.append(CodedImage(file, np.ascontiguousarray(recon), provenance))
    return coded if isinstance(quality, (list, tuple)) else coded[0]
