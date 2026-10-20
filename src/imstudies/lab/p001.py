"""Adapter of the P001 set (B10): a copy of the dental sources under codes D01..., the set registry and the code table.

Contract: ../REFERENCE.md#b10 (one level above this repository), version 0.2.0. Reads the declared
8-bit BMP and PNG sources, drops exact duplicates, assigns codes by ascending pixels_sha256 and writes data/P001/DNN.png,
data/registry/P001.json and <private_root>/P001_codes.json. Refusals, the registry and the console
carry codes and source indices (1-based, in the sorted list of source paths), never source names or
paths; only the code table outside the project holds them.

The BMP is read here, not by Pillow: Pillow turns an 8-bit BMP with a grey ramp palette into mode L
and misreads an 8-bit BMP without a colour table, so the kind of the source (palette / gray8) would
be lost. Only uncompressed 8-bit BMP is read; anything else is refused in words. A PNG is read by Pillow
after its header is checked: grey of 8 bit (L) or a palette (P) grey on the indices the image uses;
colour, alpha (including tRNS) and 16 bit are refused in words. pixels_sha256 is the
sha256 of the C-order uint8 bytes (as in the run files of cycles 1, 3, 5); a duplicate has the same
shape and the same pixels_sha256.
"""

import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, features

from imstudies.lab.runs import RunRefusal, sha256_file

CONTRACT = "B10 0.2.0"
PANORAMA_DIR = "Dental_panorama_original"
EXTRA_SOURCE = "Dental_raw/dental2.bmp"
N_UNIQUE = 16
MAX_PIXELS = 8_912_896  # level 5 of VVC (B05)
TILE = 512
V1_CROP = 512  # the centre crop of the V1 dental images (run_v1.py, cycle 1)
PNG_ARGS = {"compress_level": 6, "optimize": False}
PNG_CHUNKS = {"IHDR", "IDAT", "IEND"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_COLOUR = {0: "grey", 2: "RGB colour", 3: "palette", 4: "grey with alpha", 6: "RGB colour with alpha"}
SOURCE_SUFFIXES = (".bmp", ".png")


def source_paths(images_root):
    """Sorted paths of the B10 sources relative to images_root (POSIX); a missing folder or file is refused."""
    root = Path(images_root)
    folder = root / PANORAMA_DIR
    if not folder.is_dir():
        raise RunRefusal(f"the panorama folder of B10 is missing under images_root: {PANORAMA_DIR}")
    rels = [f"{PANORAMA_DIR}/{p.name}" for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SOURCE_SUFFIXES]
    if not (root / EXTRA_SOURCE).is_file():
        raise RunRefusal("the extra source of B10 in Dental_raw is missing under images_root")
    return sorted(rels + [EXTRA_SOURCE])


class Bmp8:
    """An uncompressed 8-bit BMP: pixel indices (a view of the file bytes) and the colour table, if any."""

    fmt = "bmp"

    def __init__(self, data):
        if len(data) < 54 or data[:2] != b"BM":
            raise ValueError("not a BMP file")
        offset, header = struct.unpack_from("<II", data, 10)
        if header < 40:
            raise ValueError(f"BMP header of {header} bytes is not read")
        width, height, _, bpp, compression = struct.unpack_from("<iiHHI", data, 18)
        if bpp != 8:
            raise ValueError(f"{bpp}-bit BMP: only 8-bit grey is read")
        if compression != 0:
            raise ValueError(f"compressed BMP (compression {compression}) is not read")
        if width <= 0 or height == 0:
            raise ValueError(f"BMP of size {width}x{height}")
        self.h, self.w = abs(height), width
        stride = (width + 3) // 4 * 4
        if offset < 14 + header or len(data) < offset + stride * self.h:
            raise ValueError("BMP pixel data shorter than its header says")
        n_table = (offset - 14 - header) // 4
        used = struct.unpack_from("<I", data, 46)[0]
        n_table = min(n_table, used) if used else n_table
        self.table = np.frombuffer(data, np.uint8, count=4 * n_table, offset=14 + header).reshape(n_table, 4)
        rows = np.frombuffer(data, np.uint8, count=stride * self.h, offset=offset).reshape(self.h, stride)[:, :width]
        self.index = rows[::-1] if height > 0 else rows  # bottom-up unless the height is negative
        self.kind = "palette" if n_table else "gray8"
        if n_table:
            used_idx = np.flatnonzero(np.bincount(self.index.ravel(), minlength=256))
            if used_idx.max() >= n_table:
                raise ValueError("a pixel index lies outside the colour table")
            b, g, r = (self.table[used_idx, k] for k in range(3))
            if not ((r == g).all() and (g == b).all()):
                raise ValueError("the palette is not grey on the indices the image uses")

    def values(self):
        """uint8 (H, W) brightness: through the grey palette, or the index itself without a table."""
        if self.kind == "palette":
            return np.ascontiguousarray(self.table[:, 2][self.index])
        return np.ascontiguousarray(self.index)


class Png8:
    """A PNG of 8-bit grey (mode L) or of a palette (mode P) grey on the indices the image uses.

    Unlike Bmp8 the pixels are decoded before the input limit is checked (Pillow decodes the whole image;
    its own decompression-bomb guard applies first)."""

    fmt = "png"

    def __init__(self, data):
        if data[:8] != PNG_SIGNATURE or png_chunk_types(data)[:1] != ["IHDR"]:
            raise ValueError("not a PNG file")
        width, height, depth, colour = struct.unpack_from(">IIBB", data, 16)
        if colour not in (0, 3):
            raise ValueError(f"PNG of {PNG_COLOUR.get(colour, f'colour type {colour}')}: only grey or a grey palette "
                             f"is read")
        if colour == 0 and depth != 8:
            raise ValueError(f"{depth}-bit grey PNG: only 8-bit grey is read")
        if "tRNS" in png_chunk_types(data):
            raise ValueError("PNG with transparency (tRNS): alpha is not read")
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            mode, index = im.mode, np.asarray(im)
            palette = im.getpalette() if mode == "P" else None
        expected = "L" if colour == 0 else "P"
        if mode != expected or index.shape != (height, width):
            raise ValueError(f"PNG decoded as mode {mode} {index.shape}, expected {expected} {(height, width)}")
        self.h, self.w = height, width
        self.kind = "gray8" if colour == 0 else "palette"
        if colour == 3:
            table = np.array(palette, dtype=np.uint8).reshape(-1, 3)
            used = np.flatnonzero(np.bincount(index.ravel(), minlength=256))
            if used.max() >= len(table):
                raise ValueError("a pixel index lies outside the colour table")
            r, g, b = table[used].T
            if not ((r == g).all() and (g == b).all()):
                raise ValueError("the palette is not grey on the indices the image uses")
            self._values = table[:, 0][index]
        else:
            self._values = index

    def values(self):
        return np.ascontiguousarray(self._values, dtype=np.uint8)


def read_sources(images_root, rels):
    """Every source parsed and checked (format, bit depth, grey palette), then the input limit, then the pixels."""
    parsed = []
    for n, rel in enumerate(rels, start=1):
        try:
            data = (Path(images_root) / rel).read_bytes()
            parsed.append(Png8(data) if rel.lower().endswith(".png") else Bmp8(data))
        except OSError as exc:
            raise RunRefusal(f"source {n} of {len(rels)}: cannot be read ({exc.strerror or type(exc).__name__})") from None
        except (ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            raise RunRefusal(f"source {n} of {len(rels)}: {exc}") from None
    for n, bmp in enumerate(parsed, start=1):
        if bmp.h * bmp.w > MAX_PIXELS:
            raise RunRefusal(f"source {n} of {len(rels)}: {bmp.h}x{bmp.w} = {bmp.h * bmp.w} pixels > the B10 limit "
                             f"{MAX_PIXELS}")
    return [(src.values(), src.kind, src.fmt) for src in parsed]


def unique_images(images_root, rels, read):
    """Images distinct by shape and pixels_sha256, coded D01... by ascending pixels_sha256."""
    groups = {}
    for n, (rel, (values, kind, fmt)) in enumerate(zip(rels, read), start=1):
        key = (values.shape, hashlib.sha256(values.tobytes()).hexdigest())
        entry = groups.setdefault(key, {"values": values, "pixels_sha256": key[1], "copies": []})
        entry["copies"].append({"index": n, "path": rel, "kind": kind, "format": fmt,
                                "file_sha256": sha256_file(Path(images_root) / rel)})
    for entry in groups.values():  # a unit whose copies differ in kind or format names both
        entry["kind"] = "+".join(sorted({c["kind"] for c in entry["copies"]}))
        entry["format"] = "+".join(sorted({c["format"] for c in entry["copies"]}))
    if len(groups) != N_UNIQUE:
        raise RunRefusal(f"{len(groups)} unique images from {len(rels)} source files ({len(rels) - len(groups)} "
                         f"duplicates); B10 expects {N_UNIQUE}")
    images = sorted(groups.values(), key=lambda e: (e["pixels_sha256"], e["values"].shape))
    for k, entry in enumerate(images, start=1):
        entry["code"] = f"D{k:02d}"
    return images


def png_chunk_types(data):
    types, pos = [], 8
    while pos + 8 <= len(data):
        length, kind = struct.unpack_from(">I4s", data, pos)
        types.append(kind.decode("latin-1"))
        pos += 12 + length
    return types


def png_bytes(values):
    """PNG of mode L, 8 bit, without text, time or ICC chunks; a foreign chunk is a failure."""
    buf = io.BytesIO()
    Image.fromarray(values, "L").save(buf, "PNG", **PNG_ARGS)
    data = buf.getvalue()
    extra = set(png_chunk_types(data)) - PNG_CHUNKS
    if extra:
        raise RuntimeError(f"PNG writer added chunks {sorted(extra)}")
    return data


def link_v1(images, v1_codes, v1_crops):
    """v1dNN -> DNN by the pixels of the V1 crop: the code file gives shape and corner, cycle 1 the crop sha256."""
    link, by_file = {}, {}
    for entry in v1_codes:
        code, shape, (top, left) = entry["code"], tuple(entry["full_shape"]), entry["crop_top_left"]
        crop = v1_crops.get(code)
        if crop is None or tuple(crop["full_shape"]) != shape or list(crop["crop_top_left"]) != [top, left]:
            raise RunRefusal(f"{code}: the code file and the run file of cycle 1 disagree on shape or crop")
        hits = [img["code"] for img in images if img["values"].shape == shape and hashlib.sha256(
            np.ascontiguousarray(img["values"][top:top + V1_CROP, left:left + V1_CROP]).tobytes()).hexdigest()
                == crop["pixels_sha256"]]
        if len(hits) != 1:
            raise RunRefusal(f"{code}: found at {len(hits)} images of the set, exactly one expected (B10)")
        link[code] = hits[0]
        same_file = {img["code"] for img in images for c in img["copies"] if c["file_sha256"] == entry.get("file_sha256")}
        if same_file and same_file != {hits[0]}:
            raise RunRefusal(f"{code}: linked to {hits[0]} by pixels but to {sorted(same_file)} by file sha256")
        by_file[code] = bool(same_file)
    return link, by_file


def tiles(h, w):
    return {"full_tiles": (h // TILE) * (w // TILE), "tile_rows": h // TILE, "tile_cols": w // TILE,
            "remainder_bottom_px": h % TILE, "remainder_right_px": w % TILE}


def leak_count(blobs, names):
    """Occurrences of source file names and stems in the blobs: plain or JSON-escaped, case-insensitive.

    A blob is decoded as UTF-8 with undecodable bytes dropped (PNG data included), so a name written in any
    case or as JSON \\u escapes of either case is found."""
    needles = set()
    for name in names:
        for text in (name, Path(name).stem):
            needles |= {form.lower() for form in (text, text.lower(), json.dumps(text)[1:-1],
                                                   json.dumps(text.lower())[1:-1])}
    texts = [blob.decode("utf-8", errors="ignore").lower() for blob in blobs]
    return sum(text.count(n) for text in texts for n in needles if n)


def build_copy(images_root, data_root, private_root, v1_codes, v1_crops, state, log=print):
    """Contract B10: copy, registry and code table. Returns the registry and the count of source names found in
    the written PNG files and the registry (the caller counts its console). Existing outputs are refused."""
    images_root, data_root, private_root = Path(images_root), Path(data_root), Path(private_root)
    for name, path in (("images_root", images_root), ("data_root", data_root), ("private_root", private_root)):
        if not path.is_dir():
            raise RunRefusal(f"{name} does not exist or is not a folder")
    out_dir, registry_path = data_root / "P001", data_root / "registry" / "P001.json"
    codes_path = private_root / "P001_codes.json"
    for what, path in (("the copy folder data/P001", out_dir), ("the registry data/registry/P001.json", registry_path),
                       ("the code table P001_codes.json in private_root", codes_path)):
        if path.exists():
            raise RunRefusal(f"{what} already exists: B10 never overwrites")
    rels = source_paths(images_root)
    images = unique_images(images_root, rels, read_sources(images_root, rels))
    names = [Path(r).name for r in rels]
    console = []

    def say(line):
        console.append(line)
        log(line)

    pngs = {img["code"]: png_bytes(img["values"]) for img in images}
    link, by_file = link_v1(images, v1_codes, v1_crops)
    registry = {
        "_what": "Registry of the P001 set (contract B10): one entry per unique image under its code; no source names "
                 "or paths - the code table lives outside the project.",
        "set": "P001", "contract": CONTRACT, **state,
        "n_source_files": len(rels), "n_duplicates": len(rels) - len(images), "n_unique": len(images),
        "pixels_sha256_rule": "sha256 of the C-order uint8 bytes; a duplicate has the same shape and pixels_sha256",
        "code_rule": "D01... by ascending pixels_sha256",
        "png": {"writer": f"Pillow {PIL.__version__}, zlib {features.version('zlib')}", "mode": "L",
                "args": PNG_ARGS, "chunks": sorted(PNG_CHUNKS)},
        "grid": {"tile": TILE, "origin": "top-left, no overlap",
                 "full_tiles": "floor(H/512) * floor(W/512); remainders at the bottom and right edges, px"},
        "limit_pixels": MAX_PIXELS,
        "images": [{"code": img["code"], "H": img["values"].shape[0], "W": img["values"].shape[1],
                    "pixels_sha256": img["pixels_sha256"], "png_sha256": hashlib.sha256(pngs[img["code"]]).hexdigest(),
                    "source_kind": img["kind"], "source_format": img["format"], "n_source_files": len(img["copies"]), "class": "orthopantomogram",
                    "role": "check", **tiles(*img["values"].shape),
                    "v1_codes": sorted(k for k, v in link.items() if v == img["code"])} for img in images],
        "v1_link": link,
        "v1_link_confirmed_by_file_sha256": by_file,
    }
    text = json.dumps(registry, ensure_ascii=False, indent=1)
    if leak_count([text.encode("utf-8")] + list(pngs.values()), names):
        raise RunRefusal("the registry or a PNG carries a source name: nothing written")
    out_dir.mkdir(parents=True)
    for img in images:
        path = out_dir / f"{img['code']}.png"
        with open(path, "xb") as f:
            f.write(pngs[img["code"]])
        with Image.open(path) as im:
            back_mode, back = im.mode, np.asarray(im)
        if back_mode != "L" or not np.array_equal(back, img["values"]):
            raise RuntimeError(f"{img['code']}: the PNG read back differs from the source pixels")
        say(f"  {img['code']}: {img['values'].shape[0]} x {img['values'].shape[1]}, {img['kind']}, {img['format']}, "
            f"source files {len(img['copies'])}, full 512 tiles {tiles(*img['values'].shape)['full_tiles']}")
    if leak_count(["\n".join(console).encode("utf-8")], names):
        raise RunRefusal("the console carries a source name: registry and code table not written")
    registry_path.parent.mkdir(exist_ok=True)
    with open(registry_path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    table = {"_what": "P001 code -> every source copy (path relative to images_root) and its sha256. Names identify "
                      "patients: this file stays outside the project and never enters a repository.",
             "contract": CONTRACT,
             "codes": {img["code"]: [{"path": c["path"], "file_sha256": c["file_sha256"]} for c in img["copies"]]
                       for img in images}}
    with open(codes_path, "x", encoding="utf-8", newline="\n") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)
    written = [p.read_bytes() for p in sorted(out_dir.iterdir())] + [registry_path.read_bytes()]
    return registry, leak_count(written, names)
