"""Lab tests of imstudies.lab.p001_tiles (contract ../REFERENCE.md#b11) on synthetic tiles with known truth.

The oracle does not call the module: shares are counted in the test by Python loops over pixels and
over the 8 x 8 blocks with statistics.pstdev (another path than numpy std over reshaped blocks).
Tiles: all 255, all 0, flat grey 128, noise sigma = 10, half frame / half noise, and a chessboard of
4 x 4 squares (flat in 4 x 4 blocks, not in 8 x 8 - the block size is part of the measure). Also: tile count on
sizes not divisible by 512, a copy whose png_sha256 differs from the registry is refused, the page claim
is checked against its run-file entry and fails on swapped fields, the page carries the four parts of
the explanation.
B11 0.2.0: offsets of the centred grid written by hand (1504 x 2773, 1488 x 2789, odd remainder, multiple
of 512); with origin "top_left" the real copy gives the rows of the cycle 8 run file bit for bit (lab).
"""

import hashlib
import json
import statistics

import numpy as np
import pytest

pytestmark = pytest.mark.lab


@pytest.fixture(scope="module")
def tiles():
    import importlib

    return importlib.import_module("imstudies.lab.p001_tiles")


def oracle(tile):
    """Shares by plain loops: pixel thresholds, block population standard deviation, union counted per pixel."""
    rows = tile.tolist()
    n = 512 * 512
    white = sum(v >= 250 for row in rows for v in row)
    black = sum(v <= 5 for row in rows for v in row)
    flat_pixels, flat_blocks = set(), 0
    for by in range(0, 512, 8):
        for bx in range(0, 512, 8):
            block = [rows[y][x] for y in range(by, by + 8) for x in range(bx, bx + 8)]
            if statistics.pstdev(block) < 1.0:
                flat_blocks += 1
                flat_pixels |= {(y, x) for y in range(by, by + 8) for x in range(bx, bx + 8)}
    any_count = sum(1 for y in range(512) for x in range(512)
                    if rows[y][x] >= 250 or rows[y][x] <= 5 or (y, x) in flat_pixels)
    return {"share_white": white / n, "share_black": black / n, "share_flat": flat_blocks / 4096, "share_any": any_count / n}


def noise_tile(seed=1):
    rng = np.random.default_rng(seed)
    return np.clip(np.rint(128 + rng.normal(0, 10, (512, 512))), 0, 255).astype(np.uint8)


def checker4_tile():
    """4 x 4 squares of 100 and 104 in a chessboard: every 4 x 4 block flat, every 8 x 8 block std = 2."""
    y, x = np.mgrid[0:512, 0:512]
    return np.where((y // 4 + x // 4) % 2 == 0, 100, 104).astype(np.uint8)


@pytest.mark.parametrize("name", ["white", "black", "grey", "noise", "half", "checker4"])
def test_shares_against_the_loop_oracle(tiles, name):
    tile = {"white": np.full((512, 512), 255, np.uint8), "black": np.zeros((512, 512), np.uint8),
            "grey": np.full((512, 512), 128, np.uint8), "noise": noise_tile(), "checker4": checker4_tile()}.get(name)
    if name == "half":
        tile = noise_tile(2)
        tile[:256] = 255  # the upper half is frame
    got, want = tiles.tile_shares(tile), oracle(tile)
    for key in want:
        assert got[key] == pytest.approx(want[key], abs=1e-12), key
    known = {"white": (1, 0, 1, 1), "black": (0, 1, 1, 1), "grey": (0, 0, 1, 1)}
    if name in known:
        assert tuple(got[k] for k in ("share_white", "share_black", "share_flat", "share_any")) == known[name]
    if name == "noise":
        assert got["share_flat"] == 0 and got["share_any"] < 0.01  # the other side: textured tile is not background
    if name == "half":
        assert 0.5 <= got["share_any"] < 0.51 and got["share_flat"] == pytest.approx(0.5)
    if name == "checker4":  # the block size is part of the measure: 8 x 8 blocks of this tile are not flat
        assert got["share_flat"] == 0 and got["share_any"] == 0


def test_grid_on_sizes_not_divisible_by_512(tiles):
    assert len(tiles.grid(1504, 2773)) == 2 * 5 and tiles.grid(1504, 2773)[-1] == (1, 4, 4 * 512, 512)
    assert tiles.grid(511, 4000) == [] and len(tiles.grid(1024, 1024)) == 4 and len(tiles.grid(1535, 1023)) == 2


def test_centred_grid_offsets_by_hand(tiles):
    # (H, W) -> (x0, y0): remainders 213 and 480 -> (106, 240); 229 and 464 -> (114, 232); odd 3 and 3 -> (1, 1)
    assert tiles.offsets(1504, 2773, "center") == (106, 240)
    assert tiles.offsets(1488, 2789, "center") == (114, 232)
    assert tiles.offsets(1027, 1539, "center") == (1, 1)
    assert tiles.offsets(1024, 1536, "center") == (0, 0) and tiles.offsets(1504, 2773, "top_left") == (0, 0)
    centred = tiles.grid(1504, 2773, origin="center")
    assert len(centred) == 10 and centred[0] == (0, 0, 106, 240) and centred[-1] == (1, 4, 106 + 4 * 512, 240 + 512)
    assert tiles.grid(1504, 2773) == tiles.grid(1504, 2773, origin="top_left")  # the default is 0.1.0
    with pytest.raises(ValueError, match="origin 'middle'"):
        tiles.grid(1504, 2773, origin="middle")


def test_top_left_repeats_cycle_8_bitwise(tiles):
    """Regression on the real copy (lab): with origin "top_left" every share equals the run file of cycle 8."""
    from pathlib import Path

    core = Path(__file__).resolve().parents[1]
    if not (core / "lab_paths.json").is_file():
        pytest.skip("lab_paths.json is absent: the laboratory is not configured on this machine")
    raw = json.loads((core / "lab_paths.json").read_text(encoding="utf-8"))
    data_root, cache_root = (core / raw["data_root"]).resolve(), (core / raw["cache_root"]).resolve()
    cycle8 = cache_root / "cycles" / "8" / "p001_tiles.json"
    if not cycle8.is_file() or not (data_root / "registry" / "P001.json").is_file():
        pytest.skip("the copy of B10 or the run file of cycle 8 is absent on this machine")
    registry, images = tiles.load_copy(data_root)
    rows = tiles.measure(images, origin="top_left")
    old = json.loads(cycle8.read_text(encoding="utf-8"))["rows"]
    assert len(rows) == len(old) == 154
    assert rows == old  # tile ids, corners and the four shares, bit for bit


def write_copy(root, shapes, seed=0):
    """A synthetic copy of B10: data/P001/DNN.png and a registry with the fields B11 reads."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    (root / "P001").mkdir(parents=True)
    (root / "registry").mkdir()
    entries = []
    for k, (h, w) in enumerate(shapes, start=1):
        code = f"D{k:02d}"
        values = rng.integers(0, 256, (h, w)).astype(np.uint8)
        values[:, :40] = 255  # a frame strip
        Image.fromarray(values, "L").save(root / "P001" / f"{code}.png")
        entries.append({"code": code, "H": h, "W": w, "full_tiles": (h // 512) * (w // 512),
                        "png_sha256": hashlib.sha256((root / "P001" / f"{code}.png").read_bytes()).hexdigest()})
    (root / "registry" / "P001.json").write_text(json.dumps({"contract": "B10 0.2.0", "images": entries}), encoding="utf-8")
    return root


def test_copy_read_and_png_sha256_checked(tiles, tmp_path):
    root = write_copy(tmp_path / "data", [(600, 1100), (520, 530)])
    registry, images = tiles.load_copy(root)
    rows = tiles.measure(images)
    assert [i["n_tiles"] for i in tiles.per_image(registry, rows)] == [2, 1]
    assert [r["tile"] for r in rows] == ["D01_r00_c00", "D01_r00_c01", "D02_r00_c00"]
    png = root / "P001" / "D02.png"
    png.write_bytes(png.read_bytes() + b"\0")
    with pytest.raises(tiles.RunRefusal, match="D02: png_sha256 of the copy differs from the registry"):
        tiles.load_copy(root)
    with pytest.raises(tiles.RunRefusal, match="registry data/registry/P001.json of B10 0.2.0 does not exist"):
        tiles.load_copy(tmp_path / "empty")


def test_page_claim_against_the_run_file_entry(tiles):
    entry = {"code": "D07", "n_tiles": 10, "n_any_le_10": 4, "n_any_le_25": 6, "n_any_le_50": 9, "max_share_any": 0.734}
    claim = tiles.statement(entry)
    tiles.check_statement(claim, entry)
    swapped = entry | {"n_any_le_10": entry["n_any_le_50"], "n_any_le_50": entry["n_any_le_10"]}
    with pytest.raises(tiles.RunRefusal, match="D07: the page claim .* differs from the run file"):
        tiles.check_statement(tiles.statement(swapped), entry)
    with pytest.raises(tiles.RunRefusal, match="does not have the declared form"):
        tiles.check_statement(claim.replace("≤ 25 %", "< 25 %"), entry)


def test_page_carries_title_claim_and_four_part_explanation(tiles, tmp_path):
    values = np.full((600, 1100), 255, np.uint8)
    rows = [{"x": 0, "y": 0, "share_any": 1.0}, {"x": 512, "y": 0, "share_any": 0.25}]
    texts = tiles.draw_page(values, rows, "D01 — title", "claim", tmp_path / "page.png")
    assert (tmp_path / "page.png").is_file() and texts[:2] == ["D01 — title", "claim"]
    assert texts[2] == "На пальцах: " + tiles.EXPLANATION[0] and texts[3:6] == list(tiles.EXPLANATION[1:])
    assert len(tiles.EXPLANATION) == 4 and texts[6:] == ["100 %", "25 %"]
