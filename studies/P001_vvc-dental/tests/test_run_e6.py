"""Lab tests of run_e6.py (study P001, E6 declared in STUDY.md section 6) on a synthetic copy - no pixel of P001.

The copy: 10 "images" 64 x 96 with two fragments 32 x 32 each, a registry and a grid; an E5 run file with the
features of the 20 fragments and their threshold PCC at 42 dB; an E4 run file with the points, whose sha256 are
those of a substitute coder (each codec its own deterministic file and reconstruction).
- the run: 14 pages, 56 points encoded again and equal to E4, claims checked, PNG files, pages.csv without a
  verdict, the book with the drop-down list of the four words on «Вердикт»; no list of integers in the run file;
- another reconstruction of one point - a refusal in words naming the fragment and the codec;
- the book: a changed cell is refused; verdicts and notes are carried by a rebuild; a page with a verdict gone from
  the disk or from the run file - a refusal by name.
"""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.lab

STUDY = Path(__file__).resolve().parents[1]
CODECS = ("vtm", "jpeg", "heif", "avif")


@pytest.fixture(scope="module")
def e6():
    spec = importlib.util.spec_from_file_location("run_e6", STUDY / "run_e6.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def substitute(paths, codec, image, pcc, work):
    """A coder of the test: a file of its own per codec and PCC, the reconstruction quantised per codec."""
    k = CODECS.index(codec) + 2
    file = f"{codec}|{pcc}|".encode() + hashlib.sha256(image.tobytes()).digest()
    return file, ((image // k) * k).astype(np.uint8)


@pytest.fixture()
def world(tmp_path, e6, monkeypatch):
    from PIL import Image

    rng = np.random.default_rng(16)
    (tmp_path / "data" / "P001").mkdir(parents=True)
    (tmp_path / "data" / "registry").mkdir()
    entries, tiles, fragments, units, rows = [], [], [], [], []
    for i in range(10):
        code = f"D{i + 1:02d}"
        values = rng.integers(0, 256, (64, 96)).astype(np.uint8)
        path = tmp_path / "data" / "P001" / f"{code}.png"
        Image.fromarray(values, "L").save(path)
        entries.append({"code": code, "H": 64, "W": 96, "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "full_tiles": 2})
        for r in (0, 1):
            tile = f"{code}_r0{r}_c00"
            tiles.append({"tile": tile, "code": code, "r": r, "c": 0, "x": 32, "y": 32 * r})
            array = np.ascontiguousarray(values[32 * r:32 * r + 32, 32:64])
            fragments.append({"unit": tile, "image": code, "mgm": float(rng.uniform(0.05, 0.9)),
                              "ln_cr_vtm": float(rng.uniform(1, 4)), "ln_r_jpeg": float(rng.uniform(0, 0.5))})
            for k, codec in enumerate(CODECS):
                pcc = 30 + k
                units.append({"unit": tile, "codec": codec, "t_db": 42.0, "judged": True, "reason": None, "pcc": pcc})
                file, recon = substitute(None, codec, array, pcc, None)
                rows.append({"unit": tile, "codec": codec, "param": pcc, "status": "valid", "psnr_hvs_m_db": 42.25 + k,
                             "psnr_db": 39.0 + k, "haarpsi": 0.9, "cr_file": 10.5 + k, "bpp": 0.8,
                             "file_sha256": hashlib.sha256(file).hexdigest(),
                             "recon_sha256": hashlib.sha256(recon.tobytes()).hexdigest()})
    registry = tmp_path / "data" / "registry" / "P001.json"
    registry.write_text(json.dumps({"contract": "B10 0.2.0", "images": entries}), encoding="utf-8")
    grid = tmp_path / "p001_tiles.json"
    grid.write_text(json.dumps({"rows": tiles, "parameters": {"tile": 32}}), encoding="utf-8")
    e5 = tmp_path / "e5_analysis.json"
    e5.write_text(json.dumps({"analysis": {"complexity": {"fragments": fragments}, "units": units}}), encoding="utf-8")
    e4 = tmp_path / "e4_main.json"
    e4.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    monkeypatch.setattr(e6, "encode", substitute)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    return {"e5": e5, "e4": e4, "data": tmp_path / "data", "grid": grid, "out": tmp_path / "cycles16",
            "pins": (sha(e5), sha(e4)), "registry_pins": (sha(registry), sha(grid))}


def run(e6, w):
    return e6.run(w["e5"], w["e4"], w["data"], w["grid"], w["out"], {"core_commit": "test", "core_tree": "test"}, {},
                  pins=w["pins"], registry_pins=w["registry_pins"], log=lambda *a: None)


def longest_int_list(obj):
    if isinstance(obj, list):
        own = len(obj) if obj and all(isinstance(x, int) and not isinstance(x, bool) for x in obj) else 0
        return max([own] + [longest_int_list(x) for x in obj])
    if isinstance(obj, dict):
        return max([0] + [longest_int_list(x) for x in obj.values()])
    return 0


def test_pages_book_and_csv(e6, world):
    run_path, book_path = run(e6, world)
    record = json.loads(run_path.read_text(encoding="utf-8"))
    pages = record["pages"]
    assert len(pages) == 14 and len({p["unit"] for p in pages}) == 14
    assert record["equal"] == {"points": 56, "file_and_recon_equal": 56}
    for p in pages:
        assert (world["out"] / p["file"]).is_file()
        e6.p001_e6.check_statement(p["claim"], p)
        assert p["claim"].endswith("у всех четырёх кодеков PSNR-HVS-M не ниже 42 дБ — по метрике искажения незаметны.")
    with open(world["out"] / e6.CSV_FILE, encoding="utf-8", newline="") as f:
        table = list(csv.reader(f))
    assert table[0] == list(e6.CSV_COLUMNS) and len(table) == 15 and "Вердикт" not in table[0]
    from openpyxl import load_workbook
    sheet = load_workbook(book_path)["Страницы"]
    lists = [dv for dv in sheet.data_validations.dataValidation if dv.type == "list"]
    assert len(lists) == 1 and lists[0].formula1 == '"ок,не тот класс,ложная тревога,пропуск"'
    assert str(lists[0].sqref) == "F3:F16" and sheet["F1"].value == "Вердикт" and sheet["G1"].value == "Примечание"
    assert sheet["A3"].hyperlink.target == pages[0]["file"]
    assert sheet["E3"].value == "\n".join(e6.p001_e6.EXPLANATION)
    assert longest_int_list(record) == 0  # no pixels in the run file
    with pytest.raises(e6.RunRefusal, match="already exists"):
        run(e6, world)


def test_another_reconstruction_is_refused_in_words(e6, world, monkeypatch):
    def changed(paths, codec, image, pcc, work):
        file, recon = substitute(paths, codec, image, pcc, work)
        if codec == "heif":
            recon = recon.copy()
            recon[0, 0] ^= 1
        return file, recon
    monkeypatch.setattr(e6, "encode", changed)
    with pytest.raises(e6.RunRefusal, match=r"_c00, HEIF PCC 32: encoded again, the reconstruction differs from E4"):
        run(e6, world)


def test_book_cell_check_and_verdicts_carried(e6, world):
    run_path, book_path = run(e6, world)
    from openpyxl import load_workbook
    book = load_workbook(book_path)
    pages = book["Страницы"]
    pages["F3"], pages["G3"] = "ок", None
    pages["F4"], pages["G4"] = "пропуск", "JPEG, корень"
    first, second = pages["A3"].value, pages["A4"].value
    book.save(book_path)
    carried = e6.rebuild_book(run_path, book_path)
    assert carried == {first: ("ок", None), second: ("пропуск", "JPEG, корень")}
    pages = load_workbook(book_path)["Страницы"]
    assert (pages["F3"].value, pages["F4"].value, pages["G4"].value) == ("ок", "пропуск", "JPEG, корень")
    book = load_workbook(book_path)
    book["Таблица"]["G3"] = 99
    book.save(book_path)
    with pytest.raises(e6.RunRefusal, match="book cell Таблица!R3C7"):
        e6.check_book(run_path, book_path)
    e6.build_book(run_path, book_path, carried)  # the cell is right again, the verdicts back
    (world["out"] / second).unlink()
    with pytest.raises(e6.RunRefusal, match=f"pages with a verdict are gone from the disk: \\['{second}'\\]"):
        e6.rebuild_book(run_path, book_path)
    record = json.loads(run_path.read_text(encoding="utf-8"))
    record["pages"] = [p for p in record["pages"] if p["file"] != first]
    run_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(e6.RunRefusal, match=f"pages with a verdict are gone from the run file: \\['{first}'\\]"):
        e6.rebuild_book(run_path, book_path)
