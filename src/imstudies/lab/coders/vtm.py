"""VVC codec VTM: one 8-bit grey image -> bitstream at a QP -> decoded image.

Contract: ../REFERENCE.md#b05 (one level above this repository); refusals: #spec-result.
Executables are our builds of tools/build_coders.ps1; their paths come from lab_paths.json.
"""

import hashlib
import math
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from imstudies.core.metrics import Refusal

QP_MIN, QP_MAX = -12, 63  # B05: -6 * (InternalBitDepth - 8) .. 63 at InternalBitDepth 10
BIT_DEPTHS = {"InputBitDepth": 8, "InternalBitDepth": 10, "OutputBitDepth": 8}


class VtmResult(NamedTuple):
    """Outputs of B05: stored bitstream, DecoderApp and EncoderApp reconstructions, log PSNR-Y, provenance."""

    bitstream: bytes
    recon: np.ndarray
    encoder_recon: np.ndarray
    log_psnr_y: float | None
    provenance: dict


def check_timeout_s(timeout_s):
    """Step (5) of B05 for timeout_s, shared with B09: a positive finite number, not bool."""
    is_number = (isinstance(timeout_s, (int, float, np.integer, np.floating))
                 and not isinstance(timeout_s, (bool, np.bool_)))
    if not is_number or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise Refusal("parameter_out_of_range", f"timeout_s={timeout_s!r}, allowed: a positive finite number of seconds")


def _validate(image, qp, max_pixels, level, frame_rate, timeout_s):
    """Validation steps (1)-(6) of B05; nothing is written before they pass."""
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise Refusal("not_2d", f"expected a 2-D numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise Refusal("unsupported_dtype", f"expected uint8, got {image.dtype}")
    h, w = image.shape
    if h < 8 or w < 8:
        raise Refusal("image_too_small", f"{h}x{w}, each side must be >= 8")
    explicit = (("qp", qp), ("max_pixels", max_pixels), ("level", level), ("frame_rate", frame_rate),
                ("timeout_s", timeout_s))
    for name, value in explicit:  # None for timeout_s would silently mean "no limit"
        if value is None:
            raise Refusal("parameter_not_assigned", name)
    if isinstance(qp, bool) or not isinstance(qp, (int, np.integer)) or not QP_MIN <= qp <= QP_MAX:
        raise Refusal("parameter_out_of_range", f"qp={qp!r}, allowed integers {QP_MIN}..{QP_MAX}")
    check_timeout_s(timeout_s)
    if h * w > max_pixels:
        raise Refusal("image_too_large", f"{h}x{w} = {h * w} pixels > max_pixels = {max_pixels}")


def _tail(text, n=1500):
    return text[-n:] if text else "(no output)"


def _run(cmd, timeout_s):
    """Run an executable; a non-zero exit or the time limit is a failure (status `error`), never a refusal."""
    started = time.perf_counter()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"{Path(cmd[0]).name} exceeded timeout_s={timeout_s}: "
                           f"{time.perf_counter() - started:.1f} s elapsed; command: {cmd}") from exc
    if out.returncode != 0:
        raise subprocess.CalledProcessError(out.returncode, cmd, output=out.stdout[-2000:], stderr=out.stderr[-2000:])
    return out.stdout


def _plane(path, h, w, what):
    size = path.stat().st_size if path.is_file() else None
    if size != h * w:
        raise RuntimeError(f"{what}: expected {h * w} bytes (8-bit {h}x{w} plane), found {size}")
    return np.fromfile(path, dtype=np.uint8).reshape(h, w)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_decode(image, qp, *, max_pixels, level, frame_rate, timeout_s, encoder, decoder, cfg, work_root):
    """Encode `image` with EncoderApp at `qp`, decode with DecoderApp - contract ../REFERENCE.md#b05."""
    _validate(image, qp, max_pixels, level, frame_rate, timeout_s)
    encoder, decoder, cfg = (Path(p).resolve() for p in (encoder, decoder, cfg))  # CreateProcess wants absolute
    for name, path in (("EncoderApp", encoder), ("DecoderApp", decoder), ("cfg", cfg)):
        if not path.is_file():
            raise FileNotFoundError(f"VTM {name} not found: {path}")
    h, w = image.shape
    Path(work_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_root, prefix="vtm_") as tmp:
        tmp = Path(tmp)
        source, stream, enc_recon, recon = (tmp / n for n in ("input.yuv", "stream.bin", "encoder_recon.yuv", "recon.yuv"))
        source.write_bytes(np.ascontiguousarray(image).tobytes())
        encode = [str(encoder), "-c", str(cfg), "-i", str(source), "-b", str(stream), "-o", str(enc_recon),
                  *(f"--{key}={value}" for key, value in BIT_DEPTHS.items()),
                  "--InputChromaFormat=400", "--ChromaFormatIDC=400", f"--SourceWidth={w}", f"--SourceHeight={h}",
                  "--FramesToBeEncoded=1", f"--QP={int(qp)}", f"--Level={level}", f"--FrameRate={frame_rate}"]
        log = _run(encode, timeout_s)
        version = re.search(r"VTM Encoder Version[^\r\n]*", log)
        if version is None:
            raise RuntimeError(f"EncoderApp printed no 'VTM Encoder Version' line; output tail: {_tail(log)}")
        if not stream.is_file() or stream.stat().st_size == 0:  # before DecoderApp: the failure is the encoder's
            raise RuntimeError(f"EncoderApp wrote no bitstream or an empty one ({stream.name}); "
                               f"output tail: {_tail(log)}")
        bitstream = stream.read_bytes()
        decode = [str(decoder), "-b", str(stream), "-o", str(recon), "-d", "8"]
        dec_log = _run(decode, timeout_s)
        result_recon = _plane(recon, h, w, "DecoderApp output")
        result_enc_recon = _plane(enc_recon, h, w, "EncoderApp reconstruction")
    psnr_y = re.search(r"\[Y\s+([-+0-9.eEinf]+)\s+dB", log)
    dec_version = re.search(r"VTM Decoder Version[^\r\n]*", dec_log)
    provenance = {
        "encoder_version": version.group(0).strip(),
        "decoder_version": dec_version.group(0).strip() if dec_version else None,
        "encoder_sha256": _sha256(encoder), "decoder_sha256": _sha256(decoder), "cfg_sha256": _sha256(cfg),
        "encoder_command": encode, "decoder_command": decode, "qp": int(qp), "level": str(level),
        "frame_rate": frame_rate, "timeout_s": timeout_s,
        "bit_depths": dict(BIT_DEPTHS, DecoderOutputBitDepth=8), "chroma_format": "400",
    }
    return VtmResult(bitstream, result_recon, result_enc_recon, float(psnr_y.group(1)) if psnr_y else None, provenance)
