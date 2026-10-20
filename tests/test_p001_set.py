"""Lab tests of imstudies.lab.p001, the adapter of the P001 set (contract ../REFERENCE.md#b10 0.2.0), on synthetic BMP and PNG.

No medical file is used: the sets are built in tmp_path from random pixels under invented names. Both sides
of every decision are checked: grey palette (identity ramp and not) and a palette with colour on an
unused index are read, colour on a used index is refused; 8-bit BMP without a table is read as is, 24 and
16 bit are refused; PNG of a grey palette and of L are read, RGB, LA, 16-bit grey, tRNS and a colour
palette are refused; a BMP-PNG duplicate is one unit, the same name with other pixels is not a duplicate;
16 unique images are copied, 15 and 17 are refused; existing outputs are refused; the input limit is
checked on both sides; the V1 link by crop pixels; the leak count on both sides. The oracle of the
reader is Pillow's own decoding with a numpy palette lookup.
"""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.lab

PATIENT = "Петренко Тарас Іванович 2019-05-14"  # invented, like the real names: words and a date


def grey_palette(k=0):
    return [v for i in range(256) for v in ((13 + 3 * i + k) % 256,) * 3]


def write_palette_bmp(path, idx, palette):
    from PIL import Image

    im = Image.fromarray(idx.astype(np.uint8), "P")
    im.putpalette(palette)
    im.save(path, "BMP")
    return path


def write_palette_png(path, idx, palette, **save_args):
    from PIL import Image

    im = Image.fromarray(idx.astype(np.uint8), "P")
    im.putpalette(palette)
    im.save(path, "PNG", **save_args)
    return path


def write_gray8_bmp(path, values):
    """8-bit BMP without a colour table (pixel data right after the 40-byte header), bottom-up rows."""
    h, w = values.shape
    stride = (w + 3) // 4 * 4
    rows = np.zeros((h, stride), np.uint8)
    rows[:, :w] = values[::-1]
    header = struct.pack("<2sIHHI", b"BM", 54 + rows.size, 0, 0, 54) + struct.pack(
        "<IiiHHIIiiII", 40, w, h, 1, 8, 0, rows.size, 2835, 2835, 0, 0)
    path.write_bytes(header + rows.tobytes())
    return path


def pillow_values(path):
    from PIL import Image

    with Image.open(path) as im:
        idx = np.asarray(im)
        if im.mode == "L":
            return idx
        return np.array(im.getpalette(), dtype=np.uint8).reshape(-1, 3)[idx, 0]


def make_set(root, n_panorama=15, seed=0):
    """images_root with n_panorama sources in the panorama folder - palette BMPs (one 520 x 530 for the V1 link),
    source 01 a palette PNG and source 02 an L PNG, as in the real folder - and Dental_raw/dental2.bmp."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    pano, raw = root / "Dental_panorama_original", root / "Dental_raw"
    pano.mkdir(parents=True)
    raw.mkdir()
    for k in range(n_panorama):
        shape = (520, 530) if k == 0 else (12, 16 + k)
        idx = rng.integers(0, 60, shape)
        if k == 1:
            write_palette_png(pano / f"{PATIENT} {k:02d}.png", idx, grey_palette(k))
        elif k == 2:
            Image.fromarray(idx.astype(np.uint8), "L").save(pano / f"{PATIENT} {k:02d}.png", "PNG")
        else:
            write_palette_bmp(pano / f"{PATIENT} {k:02d}.bmp", idx, grey_palette(k))
    write_palette_bmp(raw / "dental2.bmp", rng.integers(0, 60, (14, 18)), grey_palette())
    write_palette_bmp(raw / "dental1.bmp", rng.integers(0, 60, (14, 18)), grey_palette())  # not a B10 source
    (pano / f"{PATIENT} 00.txt").write_text("not an image", encoding="utf-8")
    return root


def v1_for(p001, root):
    """One V1 entry for the 520 x 530 source: code file entry and the crop record of cycle 1."""
    path = root / "Dental_panorama_original" / f"{PATIENT} 00.bmp"
    values = pillow_values(path)
    top, left = (520 - 512) // 2, (530 - 512) // 2
    crop = hashlib.sha256(np.ascontiguousarray(values[top:top + 512, left:left + 512]).tobytes()).hexdigest()
    entry = {"code": "v1d01", "full_shape": [520, 530], "crop_top_left": [top, left],
             "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return [entry], {"v1d01": {"full_shape": [520, 530], "crop_top_left": [top, left], "pixels_sha256": crop}}


@pytest.fixture()
def p001():
    import importlib

    return importlib.import_module("imstudies.lab.p001")


@pytest.fixture()
def dirs(tmp_path):
    out = {"images": tmp_path / "Images", "data": tmp_path / "data", "private": tmp_path / "private"}
    out["data"].mkdir()
    out["private"].mkdir()
    return out


STATE = {"core_commit": "test", "core_tree": "test"}


def test_palette_identity_and_gray8_read_as_brightness(p001, tmp_path):
    from PIL import Image

    rng = np.random.default_rng(1)
    idx = rng.integers(0, 40, (7, 13)).astype(np.uint8)
    path = write_palette_bmp(tmp_path / "grey.bmp", idx, grey_palette(5))
    bmp = p001.Bmp8(path.read_bytes())
    lookup = np.array(grey_palette(5), dtype=np.uint8).reshape(-1, 3)[idx, 0]
    assert bmp.kind == "palette" and np.array_equal(bmp.values(), lookup) and np.array_equal(lookup, pillow_values(path))
    ramp = tmp_path / "ramp.bmp"
    Image.fromarray(idx, "L").save(ramp, "BMP")  # Pillow writes a grey ramp palette
    assert p001.Bmp8(ramp.read_bytes()).kind == "palette" and np.array_equal(p001.Bmp8(ramp.read_bytes()).values(), idx)
    gray8 = write_gray8_bmp(tmp_path / "gray8.bmp", idx)
    assert p001.Bmp8(gray8.read_bytes()).kind == "gray8" and np.array_equal(p001.Bmp8(gray8.read_bytes()).values(), idx)


def test_colour_palette_refused_only_on_a_used_index(p001, tmp_path):
    idx = np.full((6, 9), 3, np.uint8)
    idx[0, 0] = 4
    palette = grey_palette()
    palette[3 * 200:3 * 200 + 3] = [255, 0, 0]  # colour on an unused index: read
    assert p001.Bmp8(write_palette_bmp(tmp_path / "a.bmp", idx, palette).read_bytes()).kind == "palette"
    palette[3 * 4:3 * 4 + 3] = [90, 91, 90]  # colour on a used index: refused
    with pytest.raises(ValueError, match="palette is not grey on the indices the image uses"):
        p001.Bmp8(write_palette_bmp(tmp_path / "b.bmp", idx, palette).read_bytes())


def test_other_bit_depths_refused(p001, tmp_path):
    from PIL import Image

    rgb = tmp_path / "rgb.bmp"
    Image.fromarray(np.zeros((5, 6, 3), np.uint8), "RGB").save(rgb, "BMP")
    with pytest.raises(ValueError, match="24-bit BMP: only 8-bit grey is read"):
        p001.Bmp8(rgb.read_bytes())
    data = bytearray(write_gray8_bmp(tmp_path / "g.bmp", np.zeros((5, 6), np.uint8)).read_bytes())
    data[28:30] = struct.pack("<H", 16)
    with pytest.raises(ValueError, match="16-bit BMP"):
        p001.Bmp8(bytes(data))


def test_input_limit_both_sides(p001, tmp_path):
    from PIL import Image

    side = int(p001.MAX_PIXELS ** 0.5)  # side * side <= MAX_PIXELS < (side + 1) ** 2
    Image.new("L", (side, side)).save(tmp_path / "ok.bmp", "BMP")
    Image.new("L", (side + 1, side + 1)).save(tmp_path / "big.bmp", "BMP")
    (values, kind, fmt), = p001.read_sources(tmp_path, ["ok.bmp"])
    assert values.shape == (side, side)
    with pytest.raises(p001.RunRefusal, match=rf"source 2 of 2: {side + 1}x{side + 1} = \d+ pixels > the B10 limit"):
        p001.read_sources(tmp_path, ["ok.bmp", "big.bmp"])


def test_png_kinds_both_sides(p001, tmp_path):
    from PIL import Image

    idx = np.random.default_rng(2).integers(0, 40, (7, 13)).astype(np.uint8)
    grey = write_palette_png(tmp_path / "grey.png", idx, grey_palette(5))
    src = p001.Png8(grey.read_bytes())
    lookup = np.array(grey_palette(5), dtype=np.uint8).reshape(-1, 3)[idx, 0]
    assert (src.kind, src.fmt) == ("palette", "png") and np.array_equal(src.values(), lookup)
    assert np.array_equal(lookup, pillow_values(grey))
    Image.fromarray(idx, "L").save(tmp_path / "l.png", "PNG")
    src = p001.Png8((tmp_path / "l.png").read_bytes())
    assert (src.kind, src.fmt) == ("gray8", "png") and np.array_equal(src.values(), idx)
    palette = grey_palette()
    palette[3 * 200:3 * 200 + 3] = [255, 0, 0]  # colour on an unused index: read
    assert p001.Png8(write_palette_png(tmp_path / "unused.png", idx, palette).read_bytes()).kind == "palette"
    palette[3 * int(idx[0, 0]):3 * int(idx[0, 0]) + 3] = [90, 91, 90]  # colour on a used index: refused
    with pytest.raises(ValueError, match="palette is not grey on the indices the image uses"):
        p001.Png8(write_palette_png(tmp_path / "used.png", idx, palette).read_bytes())
    Image.fromarray(np.zeros((5, 6, 3), np.uint8), "RGB").save(tmp_path / "rgb.png", "PNG")
    with pytest.raises(ValueError, match="PNG of RGB colour: only grey or a grey palette is read"):
        p001.Png8((tmp_path / "rgb.png").read_bytes())
    Image.new("LA", (6, 5)).save(tmp_path / "la.png", "PNG")
    with pytest.raises(ValueError, match="PNG of grey with alpha"):
        p001.Png8((tmp_path / "la.png").read_bytes())
    Image.fromarray(np.zeros((5, 6), np.uint16)).save(tmp_path / "g16.png", "PNG")
    with pytest.raises(ValueError, match="16-bit grey PNG: only 8-bit grey is read"):
        p001.Png8((tmp_path / "g16.png").read_bytes())
    trns = write_palette_png(tmp_path / "trns.png", idx, grey_palette(), transparency=0)
    with pytest.raises(ValueError, match=r"PNG with transparency \(tRNS\)"):
        p001.Png8(trns.read_bytes())
    with pytest.raises(p001.RunRefusal, match="source 2 of 2: PNG of RGB colour"):
        p001.read_sources(tmp_path, ["l.png", "rgb.png"])


def test_refusal_names_the_index_not_the_file(p001, dirs):
    from PIL import Image

    root = make_set(dirs["images"])
    Image.fromarray(np.zeros((5, 6, 3), np.uint8), "RGB").save(root / "Dental_panorama_original" / f"{PATIENT} rgb.bmp")
    with pytest.raises(p001.RunRefusal) as info:
        p001.build_copy(root, dirs["data"], dirs["private"], *v1_for(p001, root), STATE)
    assert "source 16 of 17: 24-bit BMP" in str(info.value) and "Петренко" not in str(info.value)
    assert not (dirs["data"] / "P001").exists()


def test_copy_of_sixteen_with_duplicate_and_same_name(p001, dirs):
    from PIL import Image

    root = make_set(dirs["images"], n_panorama=14)
    pano = root / "Dental_panorama_original"
    Image.fromarray(pillow_values(pano / f"{PATIENT} 03.bmp"), "L").save(pano / "copy without the name.png")  # BMP-PNG
    write_palette_bmp(pano / "dental2.bmp", np.full((9, 9), 7, np.uint8), grey_palette())  # same name, other pixels
    lines = []
    registry, leaks = p001.build_copy(root, dirs["data"], dirs["private"], *v1_for(p001, root), STATE, log=lines.append)
    assert (registry["n_source_files"], registry["n_duplicates"], registry["n_unique"]) == (17, 1, 16)
    assert [i["code"] for i in registry["images"]] == [f"D{k:02d}" for k in range(1, 17)]
    shas = [i["pixels_sha256"] for i in registry["images"]]
    assert shas == sorted(shas) and sum(i["n_source_files"] for i in registry["images"]) == 17
    big = next(i for i in registry["images"] if (i["H"], i["W"]) == (520, 530))
    assert (big["full_tiles"], big["remainder_bottom_px"], big["remainder_right_px"]) == (1, 8, 18)
    assert registry["v1_link"] == {"v1d01": big["code"]} and registry["v1_link_confirmed_by_file_sha256"]["v1d01"]
    for img in registry["images"]:
        png = dirs["data"] / "P001" / f"{img['code']}.png"
        assert hashlib.sha256(png.read_bytes()).hexdigest() == img["png_sha256"]
        chunks = p001.png_chunk_types(png.read_bytes())
        assert chunks[0] == "IHDR" and chunks[-1] == "IEND" and set(chunks[1:-1]) == {"IDAT"}
        assert img["role"] == "check" and img["class"] == "orthopantomogram"
    source = pillow_values(pano / f"{PATIENT} 03.bmp")
    copy = next(i for i in registry["images"] if i["pixels_sha256"] == hashlib.sha256(source.tobytes()).hexdigest())
    assert copy["n_source_files"] == 2 and np.array_equal(pillow_values(dirs["data"] / "P001" / f"{copy['code']}.png"), source)
    assert (copy["source_kind"], copy["source_format"]) == ("gray8+palette", "bmp+png")
    formats = {(i["source_kind"], i["source_format"]) for i in registry["images"]}
    assert {("palette", "bmp"), ("palette", "png"), ("gray8", "png")} <= formats
    table = json.loads((dirs["private"] / "P001_codes.json").read_text(encoding="utf-8"))["codes"]
    assert sorted(c["path"] for c in table[copy["code"]]) == sorted(
        ["Dental_panorama_original/copy without the name.png", f"Dental_panorama_original/{PATIENT} 03.bmp"])
    assert leaks == 0 and p001.leak_count(["\n".join(lines).encode("utf-8")], [PATIENT + ".bmp", "dental2.bmp"]) == 0
    registry_text = (dirs["data"] / "registry" / "P001.json").read_text(encoding="utf-8")
    assert "Dental_panorama_original" not in registry_text and "dental2" not in registry_text
    with pytest.raises(p001.RunRefusal, match="the copy folder data/P001 already exists"):
        p001.build_copy(root, dirs["data"], dirs["private"], *v1_for(p001, root), STATE)


@pytest.mark.parametrize("n_panorama, found", [(14, 15), (16, 17)])
def test_other_than_sixteen_unique_refused_with_numbers(p001, dirs, n_panorama, found):
    root = make_set(dirs["images"], n_panorama=n_panorama)
    with pytest.raises(p001.RunRefusal, match=rf"^{found} unique images from {found} source files \(0 duplicates\); "
                                              r"B10 expects 16$"):
        p001.build_copy(root, dirs["data"], dirs["private"], *v1_for(p001, root), STATE)
    assert not (dirs["data"] / "P001").exists() and not (dirs["private"] / "P001_codes.json").exists()


def test_existing_outputs_refused(p001, dirs):
    root = make_set(dirs["images"])
    (dirs["data"] / "registry").mkdir()
    (dirs["data"] / "registry" / "P001.json").write_text("{}", encoding="utf-8")
    with pytest.raises(p001.RunRefusal, match="the registry data/registry/P001.json already exists"):
        p001.build_copy(root, dirs["data"], dirs["private"], *v1_for(p001, root), STATE)
    (dirs["data"] / "registry" / "P001.json").unlink()
    (dirs["private"] / "P001_codes.json").write_text("{}", encoding="utf-8")
    with pytest.raises(p001.RunRefusal, match="the code table P001_codes.json in private_root already exists"):
        p001.build_copy(root, dirs["data"], dirs["private"], *v1_for(p001, root), STATE)


def test_v1_code_not_found_is_a_stop(p001, dirs):
    root = make_set(dirs["images"])
    entries, crops = v1_for(p001, root)
    crops["v1d01"]["pixels_sha256"] = "0" * 64
    with pytest.raises(p001.RunRefusal, match=r"v1d01: found at 0 images of the set, exactly one expected"):
        p001.build_copy(root, dirs["data"], dirs["private"], entries, crops, STATE)
    assert not (dirs["data"] / "P001").exists()


def test_registry_carrying_a_source_name_writes_nothing(p001, dirs, monkeypatch):
    root = make_set(dirs["images"])
    monkeypatch.setattr(p001, "CONTRACT", f"B10 0.1.0 {PATIENT} 05")  # a source stem inside the registry text
    with pytest.raises(p001.RunRefusal, match="the registry or a PNG carries a source name: nothing written"):
        p001.build_copy(root, dirs["data"], dirs["private"], *v1_for(p001, root), STATE)
    assert not (dirs["data"] / "P001").exists() and not (dirs["data"] / "registry").exists()
    assert not (dirs["private"] / "P001_codes.json").exists()


def test_leak_count_both_sides(p001):
    name = PATIENT + " 03.bmp"
    assert p001.leak_count([b"code D03, 512 x 512"], [name]) == 0
    assert p001.leak_count([("x " + name.upper()).encode("utf-8")], [name]) >= 1  # case-insensitive, file name
    assert p001.leak_count([json.dumps({"p": PATIENT + " 03"}).encode("ascii")], [name]) == 1  # JSON-escaped stem
