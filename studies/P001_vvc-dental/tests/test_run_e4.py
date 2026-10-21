"""Lab tests of run_e4.py (study P001, E4 contract of STUDY.md section 7) on a synthetic copy of the set in tmp_path.

No pixel of P001 is read. The copy: a registry of two "images" 96 x 160 (D01, D02), a grid of 2 x 1 fragments
32 x 32 per image (sizes given by the test), PNG copies of random pixels; invented patient names lie in a private
table beside it, which the run must not read. VTM is a substitute coder with a given PSNR-HVS-M curve by QP
(D01 monotone, D02 not); JPEG, HEIF, AVIF are the real wrappers (lab_paths.json with the B12 environment needed).

- refinement: the crossing found (3 QP between the right nodes), no crossing both ways with the right reason,
  a non-monotone curve takes the first crossing from the high-quality side;
- the run: rows per image (19 grid + 3 refinement), whole images before fragments, slow first;
- resume after a stop between the grid and the refinement: every refinement runs once, rows equal an
  uninterrupted run;
- privacy: no invented name and no pixel array in the run file and the book;
- --plan equals a count by hand; missing registry, grid or copy refused in words.
"""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.lab

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
NAMES = ["Шевченко Тарас Григорович 1814-03-09.bmp", "Українка Леся 1871-02-25.png"]  # invented, like the real ones


@pytest.fixture(scope="module")
def e4():
    spec = importlib.util.spec_from_file_location("run_e4", STUDY / "run_e4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GRID = list(range(-12, 61, 4))
PLAN = {"vtm": {"f": [30, 50], "m_grid": GRID, "refine_step": 4, "threshold_db": 42.0},
        "jpeg": [50, 100], "heif": [50, 100], "avif": [50, 100], "tile": 32, "pool": 2,
        "controls": {"fragment_qps": [30], "whole_qp": 40}}
STATE = {"core_commit": "test", "core_tree": "test"}


def curve(unit, qp):
    """PSNR-HVS-M of the substitute VTM: D01 falls steadily (crossing between 24 and 28); D02 dips below 42 dB at
    QP 4, comes back at 8 and falls again (first crossing from the high-quality side: between 0 and 4)."""
    if unit.startswith("D02") and len(unit) == 3:
        return {4: 41.0, 8: 43.0}.get(qp, 60.0 - 0.6 * (qp + 12) if qp <= 0 else 48.0 - 0.5 * qp)
    return 72.0 - 0.9 * (qp + 12)


def fake_vtm(paths, unit, level, image, qp, work):
    h, w = image.shape
    size = 2000 - 20 * qp
    return {"key": f"{unit}|vtm|{qp}", "unit": unit, "level": level, "codec": "vtm", "param": qp, "status": "valid",
            "file_bytes": size, "cr_file": h * w / size, "bpp": 8 * size / (h * w), "psnr_db": curve(unit, qp) - 5,
            "psnr_hvs_m_db": curve(unit, qp), "haarpsi": 0.9, "seconds": 0.001, "seconds_metrics": 0.0,
            "file_sha256": hashlib.sha256(f"{unit}{qp}".encode()).hexdigest(), "recon_sha256": "r", "detail": None}


def synthetic_copy(root):
    from PIL import Image

    rng = np.random.default_rng(7)
    (root / "data" / "P001").mkdir(parents=True)
    (root / "data" / "registry").mkdir()
    entries, rows = [], []
    for code in ("D01", "D02"):
        values = rng.integers(0, 256, (96, 160)).astype(np.uint8)
        path = root / "data" / "P001" / f"{code}.png"
        Image.fromarray(values, "L").save(path)
        entries.append({"code": code, "H": 96, "W": 160, "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "full_tiles": 2})
        rows += [{"tile": f"{code}_r{r:02d}_c00", "code": code, "r": r, "c": 0, "x": 64, "y": 16 + 32 * r} for r in (0, 1)]
    (root / "data" / "registry" / "P001.json").write_text(json.dumps({"contract": "B10 0.2.0", "images": entries}),
                                                          encoding="utf-8")
    (root / "cycles9").mkdir()
    (root / "cycles9" / "p001_tiles.json").write_text(json.dumps({"rows": rows, "parameters": {"tile": 32}}),
                                                       encoding="utf-8")
    (root / "private").mkdir()
    (root / "private" / "P001_codes.json").write_text(
        json.dumps({"codes": {"D01": [{"path": NAMES[0]}], "D02": [{"path": NAMES[1]}]}}, ensure_ascii=False),
        encoding="utf-8")
    return root


TIMING = {"f": {"vtm": [(30, 8.0), (50, 2.0)], "jpeg": [(50, 0.001), (100, 0.002)], "heif": [(50, 0.05), (100, 0.08)],
                "avif": [(50, 0.04), (100, 0.04)]},
          "m": {"vtm": [(q, 600.0 - 8 * (q + 12)) for q in GRID], "jpeg": [(50, 0.01), (100, 0.02)],
                "heif": [(50, 0.8), (100, 1.2)], "avif": [(50, 0.4), (100, 0.4)]},
          "metrics": {"f": 0.05, "m": 0.9}, "k": 3.5, "refine_qps": [25, 26, 27]}


@pytest.fixture()
def setup(e4, tmp_path, monkeypatch):
    if not (ROOT / "lab_paths.json").is_file():
        pytest.skip("lab_paths.json is absent: the laboratory is not configured on this machine")
    raw = json.loads((ROOT / "lab_paths.json").read_text(encoding="utf-8"))
    python = raw.get("avif_libheif_python")
    if not python or not Path(python).is_file():
        pytest.skip("the B12 environment is not configured on this machine")
    root = synthetic_copy(tmp_path)
    paths = {"data_root": root / "data", "cache_root": root / "cache", "avif_libheif_python": Path(python),
             "private_root": root / "private", **{k: root / f"no_{k}" for k in ("encoder", "decoder", "cfg", "mp4box", "gpac")}}
    registry, tile_rows, tile = e4.load_inputs(paths["data_root"], root / "cycles9" / "p001_tiles.json", pins=(None, None))
    units = e4.load_units(paths["data_root"], registry, tile_rows, tile)
    monkeypatch.setattr(e4, "vtm_task", fake_vtm)
    return root, paths, registry, tile_rows, units


def run(e4, setup, out, **kw):
    root, paths, _, _, units = setup
    return e4.run(paths, units, PLAN, out, STATE, TIMING, "reg", "tiles", log=lambda *a: None, **kw)


def test_refinement_rule(e4):
    ref = e4.e4_book.refinement
    grid = [(q, 72 - 0.9 * (q + 12)) for q in GRID]  # 43.2 dB at QP 20, 39.6 dB at QP 24
    a = max(q for q, v in grid if v >= 42)
    assert a == 20
    assert ref(grid) == (list(range(a + 1, a + 4)), None) and a in GRID and a + 4 in GRID
    assert ref([(q, 50.0) for q in GRID]) == ([], "no_crossing: every node is at or above 42 dB")
    assert ref([(q, 30.0) for q in GRID]) == ([], "no_crossing: every node is below 42 dB")
    assert ref([(q, "+inf" if q < 0 else 30.0) for q in GRID]) == ([-3, -2, -1], None)  # lossless counts as above
    bumpy = [(q, curve("D02", q)) for q in GRID]
    assert ref(bumpy) == ([1, 2, 3], None)  # the first crossing from the high-quality side, not the later ones
    assert ref([(q, 30.0 if q < 20 else 50.0) for q in GRID]) == (
        [], "no_crossing: no fall below 42 dB from the high-quality side")


def test_run_rows_order_and_refinement(e4, setup, tmp_path):
    run_path, book_path = run(e4, setup, tmp_path / "out")
    record = json.loads(run_path.read_text(encoding="utf-8"))
    vtm = {u: sorted(r["param"] for r in record["rows"] if r["unit"] == u and r["codec"] == "vtm") for u in ("D01", "D02")}
    a = max(q for q in GRID if curve("D01", q) >= 42)
    assert vtm["D01"] == sorted(GRID + [a + 1, a + 2, a + 3]) and vtm["D02"] == sorted(GRID + [1, 2, 3])
    assert record["summary"]["refinement"]["D02"] == {"qps": [1, 2, 3], "reason": None}
    assert len(record["rows"]) == 2 * (19 + 3 + 6) + 4 * (2 + 6)
    whole = [r for r in record["rows"] if r.get("phase") == "grid"]
    assert {r["level"] for r in whole} == {"m"}
    second = sorted((r for r in record["rows"] if r.get("phase") in ("refine", "fragment")), key=lambda r: r["order"])
    assert [r["phase"] for r in second[:6]] == ["refine"] * 6  # whole images before fragments
    frag = [r for r in second if r["phase"] == "fragment"]
    assert [r["param"] for r in frag[:4]] == [30] * 4  # slow first: QP 30 expected 8 s, QP 50 2 s
    assert all(d["equal"] for d in record["summary"]["determinism"]) and len(record["summary"]["determinism"]) == 2
    with pytest.raises(e4.RunRefusal, match=e4.RUN_FILE):
        run(e4, setup, tmp_path / "out")


def keyed(run_path):
    record = json.loads(Path(run_path).read_text(encoding="utf-8"))
    return record, {r["key"]: (r["status"], r["file_sha256"], r["psnr_hvs_m_db"]) for r in record["rows"]}


def test_resume_between_grid_and_refinement_refines_once(e4, setup, tmp_path):
    whole, _ = run(e4, setup, tmp_path / "whole")
    out = tmp_path / "stopped"
    with pytest.raises(e4.RunStopped):
        run(e4, setup, out, stop_after=2 * (19 + 3))  # the last task of the whole-image stage, before any refinement
    entries = [json.loads(line) for line in (out / e4.JOURNAL_FILE).read_text(encoding="utf-8").splitlines()[1:]]
    assert not any(r.get("phase") == "refine" for e in entries for r in e["rows"])
    resumed, _ = run(e4, setup, out)
    entries = [json.loads(line) for line in (out / e4.JOURNAL_FILE).read_text(encoding="utf-8").splitlines()[1:]]
    refine_tasks = [e["task"] for e in entries if any(r.get("phase") == "refine" for r in e["rows"])]
    assert len(refine_tasks) == len(set(refine_tasks)) == 6
    record_r, rows_r = keyed(resumed)
    assert rows_r == keyed(whole)[1] and record_r["resumed_tasks"] == 2 * (19 + 3)
    assert len(record_r["rows"]) == len({r["key"] for r in record_r["rows"]})


def test_privacy_no_names_no_pixels(e4, setup, tmp_path):
    run_path, book_path = run(e4, setup, tmp_path / "out")
    texts = [run_path.read_text(encoding="utf-8")]
    with zipfile.ZipFile(book_path) as z:
        texts += [z.read(n).decode("utf-8", errors="ignore") for n in z.namelist()]
    for name in NAMES:
        for part in (name, Path(name).stem, json.dumps(name)[1:-1]):
            assert not any(part in t for t in texts), part

    def longest_int_list(obj):
        if isinstance(obj, list):
            own = len(obj) if obj and all(isinstance(x, int) and not isinstance(x, bool) for x in obj) else 0
            return max([own] + [longest_int_list(x) for x in obj])
        if isinstance(obj, dict):
            return max([0] + [longest_int_list(x) for x in obj.values()])
        return 0
    assert longest_int_list(json.loads(texts[0])) < 64  # no row of pixels anywhere


def test_plan_by_hand(e4, setup):
    _, _, registry, tile_rows, _ = setup
    got = e4.plan_counts(registry, tile_rows, PLAN, TIMING)
    assert [(i["code"], i["fragments"], i["vtm_whole"], i["vtm_fragments"], i["still_tasks"], i["tasks"])
            for i in got["images"]] == [("D01", 2, 22, 4, 9, 35), ("D02", 2, 22, 4, 9, 35)]
    assert got["totals"] == {"vtm": {"tasks": 52, "files": 52}, "jpeg": {"tasks": 6, "files": 12},
                             "heif": {"tasks": 6, "files": 12}, "avif": {"tasks": 6, "files": 12}}
    vtm_m = sum(600.0 - 8 * (q + 12) for q in GRID) + sum(600.0 - 8 * (q + 12) for q in (25, 26, 27))
    whole = vtm_m + 22 * 0.9 + (0.01 + 0.02 + 0.8 + 1.2 + 0.4 + 0.4) + 6 * 0.9
    frag = (8.0 + 2.0 + 2 * 0.05) + (0.001 + 0.002 + 0.05 + 0.08 + 0.04 + 0.04) + 6 * 0.05
    seconds = 2 * whole + 4 * frag
    assert got["hours_1"] == pytest.approx(seconds / 3600, rel=1e-12)
    assert got["hours_4"] == pytest.approx(seconds / 3.5 / 3600, rel=1e-12)


def test_missing_inputs_refused_in_words(e4, setup):
    root, paths, _, _, _ = setup
    tiles = root / "cycles9" / "p001_tiles.json"
    with pytest.raises(e4.RunRefusal, match="the grid cycles/9/p001_tiles.json does not exist"):
        e4.load_inputs(paths["data_root"], root / "none.json", pins=(None, None))
    (paths["data_root"] / "P001" / "D02.png").rename(root / "moved.png")
    with pytest.raises(e4.RunRefusal, match="D02: the copy data/P001/D02.png does not exist"):
        e4.load_inputs(paths["data_root"], tiles, pins=(None, None))
    with pytest.raises(e4.RunRefusal, match="registry data/registry/P001.json: sha256 differs from the pinned"):
        e4.load_inputs(paths["data_root"], tiles, pins=("0" * 64, None))
    (paths["data_root"] / "registry" / "P001.json").rename(root / "moved.json")
    with pytest.raises(e4.RunRefusal, match="the registry data/registry/P001.json does not exist"):
        e4.load_inputs(paths["data_root"], tiles, pins=(None, None))
