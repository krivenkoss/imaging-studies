"""Minimal reader of ISOBMFF/HEIF boxes: enough to check what a codec or packer wrote.

Reads the box tree of the meta item properties (meta > iprp > ipco) and the top level; does not
interpret sample tables or tracks. Used by the HEIF, AVIF and VVC-in-HEIF wrappers (B07-B09).
"""

import struct

CONTAINERS = {"meta": 4, "iprp": 0, "ipco": 0, "dinf": 0}  # child boxes start after this many payload bytes


def boxes(data, start=0, end=None, path=""):
    """Every box as (path, type, payload bytes, size with header), depth first; `path` joins parent types by '/'."""
    end = len(data) if end is None else end
    out, pos = [], start
    while pos + 8 <= end:
        size, kind = struct.unpack(">I4s", data[pos:pos + 8])
        kind = kind.decode("latin-1")
        header = 8
        if size == 1:
            size, header = struct.unpack(">Q", data[pos + 8:pos + 16])[0], 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            raise ValueError(f"broken box {kind!r} at byte {pos}: size {size}")
        payload = data[pos + header:pos + size]
        here = f"{path}/{kind}" if path else kind
        out.append((here, kind, payload, size))
        if kind in CONTAINERS:
            out += boxes(data, pos + header + CONTAINERS[kind], pos + size, here)
        pos += size
    return out


def first(data, kind):
    """Payload of the first box of type `kind`, or None."""
    return next((payload for _, k, payload, _ in boxes(data) if k == kind), None)


def ftyp_brands(data):
    payload = first(data, "ftyp")
    return [payload[0:4].decode("latin-1")] + [payload[i:i + 4].decode("latin-1") for i in range(8, len(payload), 4)]


def ispe(data):
    """(width, height) of the first ispe property (a FullBox)."""
    payload = first(data, "ispe")
    return struct.unpack(">II", payload[4:12]) if payload is not None else None


def pixi(data):
    """Bits per channel of the first pixi property (a FullBox), or None if there is none."""
    payload = first(data, "pixi")
    return list(payload[5:5 + payload[4]]) if payload is not None else None


def hevc_chroma_format_idc(data):
    """chroma_format_idc of the first hvcC (HEVCDecoderConfigurationRecord, byte 16, low 2 bits)."""
    payload = first(data, "hvcC")
    return payload[16] & 0x03 if payload is not None else None


def av1_monochrome(data):
    """monochrome flag of the first av1C (AV1CodecConfigurationRecord, byte 2, bit 4)."""
    payload = first(data, "av1C")
    return (payload[2] >> 4) & 1 if payload is not None else None


def top_level_sizes(data):
    """Total bytes of each top-level box type (header included)."""
    sizes = {}
    for path, kind, _, size in boxes(data):
        if "/" not in path:
            sizes[kind] = sizes.get(kind, 0) + size
    return sizes
