"""Packing a VTM bitstream (B05) into HEIF with GPAC, checked by a round trip at every call.

Contract: ../REFERENCE.md#b09 (one level above this repository). The file is what MP4Box writes,
never edited; it is the stored file of VVC for CR. Round trip: gpac extracts the stream, DecoderApp
decodes it with -d 8, and the result must equal the recon of B05 bit for bit.
"""

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

from imstudies.core.metrics import Refusal
from imstudies.lab.coders import isobmff
from imstudies.lab.coders.process import read_plane, run_tool
from imstudies.lab.coders.vtm import check_timeout_s


class PackedVvc(NamedTuple):
    """Outputs of B09: the HEIF file as written by MP4Box, provenance."""

    file: bytes
    provenance: dict


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpac_version(mp4box, timeout_s):
    """'26.08-DEV-rev171-gf9f54240-master' from `MP4Box -version` (printed on stderr)."""
    out = subprocess.run([str(mp4box), "-version"], capture_output=True, text=True, errors="replace", timeout=timeout_s)
    found = re.search(r"GPAC version (\S+)", out.stdout + out.stderr)
    if found is None:
        raise RuntimeError(f"MP4Box -version printed no 'GPAC version': {(out.stdout + out.stderr)[-500:]}")
    return found.group(1)


def _pack(stream, heic, mp4box, timeout_s):
    cmd = [str(mp4box), "-add-image", f"{stream}:primary", "-new", str(heic)]
    run_tool(cmd, timeout_s)
    return cmd


def pack(bitstream, recon, *, timeout_s, mp4box, gpac, decoder, work_root):
    """Pack `bitstream` into HEIF and check it back against `recon` - contract ../REFERENCE.md#b09."""
    if timeout_s is None:
        raise Refusal("parameter_not_assigned", "timeout_s")
    check_timeout_s(timeout_s)
    if not isinstance(bitstream, bytes) or not bitstream:
        raise RuntimeError("bitstream is empty or not bytes: the output of B05 is broken")
    if not isinstance(recon, np.ndarray) or recon.ndim != 2 or recon.dtype != np.uint8:
        raise RuntimeError("recon is not a 2-D uint8 array: the output of B05 is broken")
    mp4box, gpac, decoder = (Path(p).resolve() for p in (mp4box, gpac, decoder))
    for name, path in (("MP4Box", mp4box), ("gpac", gpac), ("DecoderApp", decoder)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    h, w = recon.shape
    Path(work_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_root, prefix="heif_") as tmp:
        tmp = Path(tmp)
        stream, heic, back, decoded = (tmp / n for n in ("stream.266", "image.heic", "back.vvc", "back.yuv"))
        stream.write_bytes(bitstream)
        pack_cmd = _pack(stream, heic, mp4box, timeout_s)
        if not heic.is_file() or heic.stat().st_size == 0:
            raise RuntimeError(f"MP4Box wrote no file: {heic.name}")
        file = heic.read_bytes()
        size = isobmff.ispe(file)
        if size != (w, h):
            raise RuntimeError(f"ispe of the HEIF file is {size}, expected (W, H) = {(w, h)}")
        # round trip, every call: extract, decode, compare with the recon of B05
        extract_cmd = [str(gpac), "-i", str(heic), "-o", str(back)]
        run_tool(extract_cmd, timeout_s)
        decode_cmd = [str(decoder), "-b", str(back), "-o", str(decoded), "-d", "8"]
        run_tool(decode_cmd, timeout_s)
        if not np.array_equal(read_plane(decoded, h, w, "DecoderApp output after the HEIF round trip"), recon):
            raise RuntimeError("round trip failed: the stream extracted from the HEIF file decodes to another image")
        extracted = back.read_bytes()
    sizes = isobmff.top_level_sizes(file)
    provenance = {
        "gpac_version": gpac_version(mp4box, timeout_s), "mp4box_sha256": _sha256(mp4box), "gpac_sha256": _sha256(gpac),
        "libgpac_sha256": _sha256(mp4box.parent / "libgpac.dll") if (mp4box.parent / "libgpac.dll").is_file() else None,
        "decoder_sha256": _sha256(decoder), "pack_command": pack_cmd, "extract_command": extract_cmd,
        "decode_command": decode_cmd, "timeout_s": timeout_s, "file_bytes": len(file),
        "mdat_bytes": sizes.get("mdat"), "free_bytes": sizes.get("free"), "box_bytes": sizes,
        "ftyp_brands": isobmff.ftyp_brands(file), "ispe": list(size), "pixi_bits": isobmff.pixi(file),
        "free_text": (isobmff.first(file, "free") or b"").rstrip(b"\x00").decode("latin-1") or None,
        "extracted_stream_bytes": len(extracted), "extracted_stream_sha256": hashlib.sha256(extracted).hexdigest(),
        "round_trip": "recon equal",
    }
    return PackedVvc(file, provenance)
