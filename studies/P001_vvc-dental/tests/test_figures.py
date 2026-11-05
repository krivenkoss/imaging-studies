"""Lab tests of figures.py (paper P001, cycle 19) on synthetic run files of E4, E5, E6 and the grid, a synthetic image
D10 and fragment, and a coder of the test's own - no pixel of the data, no real number.

- the files are created: eight figures (PDF, PNG, CSV), the captions, three tables (.xlsx, .csv), the index; an existing
  file is refused without --force; the figures are numbered in the order of their mention (cycles 22, 25): Fig. 2 is
  the fragments of the plots, the former Fig. 2 to Fig. 7 are Fig. 3 to Fig. 8 with their captions and index rows;
- two builds into different folders write the same bytes of every file; no PDF has CreationDate or ModDate; every xlsx
  has created, modified and every zip entry at 1980-01-01 00:00:00 and its cells equal its CSV (cycle 21);
- the choice of fragments by the declared rules, ties included: Fig. 1 (b) - the fragment of D10 with MGM closest to
  the median MGM of all fragments, (c) - the owner's choice D10_r00_c02 (cycle 21): a code not in E5, of another image
  or equal to (b) is refused; the plots of Fig. 3 and Fig. 4 - the smallest and the largest MGM; the frames of the grid
  of Fig. 1 - by the formula of B11;
- Fig. 2 (cycle 25): (a), (b) - the codes of the simple and the complex fragment of the plots of Fig. 3 and Fig. 4, the
  pixels of the loader (the grid) without interpolation, one row, the letters below the row and not over the images;
  its caption names (b) the same fragment as Fig. 8 by the number of the figure against the original in the numbering
  of the build - a reordered numbering moves it, a build whose captions disagree with the numbering is refused, and a
  figure against the original of another fragment is refused (cycle 26); "Figs. 3 and 4" of the caption take their
  numbers from the numbering as well (cycle 27);
- Table 4 (cycle 27): every cell equals the records of e5_fixed.json computed again by the test's own median and
  quartiles; the VVC row has "-" for r_fix; every digit of its head, cells and note belongs to an index row of Table 4;
  a unit below T in the run file is refused;
- the CSV of every figure equals the data of the run files; the series of Fig. 3 carry the data of their own codec;
  Fig. 3 has six panels, (e), (f) - HaarPSI of E4 on CR without threshold lines (cycle 25), its caption places the
  threshold lines in (a,b) (cycle 26);
- every series of the plots is a line with markers: the marker size equals the style, is 2.5 pt and at most 4 pt, at
  most markevery_max markers on a line; every series, threshold line and r = 1 has a legend entry; the labels of the
  logarithmic axes carry no "x 10^n"; an axis label with PCC has no brackets and no x axis of a plot has a negative
  limit or tick label (cycle 24);
- Fig. 4: in every codec panel the two series are the valid E4 rows of the simple and the complex fragment of Fig. 3,
  solid and dashed, the x axis from the style key of 0 to the largest PCC of the plan with the VVC points of QP below 0
  kept in the data and the CSV (cycle 24), the y axis of (a) VVC up to the style key of 80 dB with the points above it
  kept in the data and the CSV (cycle 23); Fig. 5: per codec one point per fragment at (MGM, threshold PCC of
  E5), small markers and no line, rho of E5 in the title; Fig. 6: 9 CI segments and 9 estimates per panel equal to
  the comparisons of E5 (patients) and the whole-image comparisons, the vertical line r = 1 (cycle 22);
- Fig. 8: the upper row - the original and the reconstructions; the lower row - the region repeated 4 times by pixels
  (the test's own index arithmetic), the region - the block with the largest MGM by the test's own Sobel sums, ties to
  the smaller row and column; a reconstruction that differs from E4 is refused; the titles of the panels do not overlap;
- in the PNG every image panel is a whole multiple of its pixels, pixel for pixel;
- no "VVC (VTM)" in any output file or figure text;
- every row of numbers_index.csv equals the key of its run file under its operation and format, by the test's own
  evaluator of keys; every table row of the index is the cell or a part of it; every digit of the captions belongs to
  an index row of its caption; Table 1 has no sha256;
- a decimal point only and no Cyrillic in the tables and in the texts of the figures;
- the words of the tables and captions of cycle 28: no "n = " before a number and no head "n <noun>" in any caption,
  head or note - "Number of patients", "Number of fragments", "VVC is better than ...", "16 patients and 16 whole
  images"; the index rows of those numbers keep their ids, keys and operations, Table 3 rows take the new head into
  their ids, and the caption of Fig. 6 has one row more - the whole images (owner, 2026-11-05);
- colour and line style differ for every pair of series; the style table has four colours, lines and markers.
"""

import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
import tomllib
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.lab

STUDY = Path(__file__).resolve().parents[1]
CODECS = ("vtm", "jpeg", "heif", "avif")
OTHER_CODECS = CODECS[1:]
LABELS ={"vtm": "VVC", "jpeg": "JPEG", "heif": "HEIF", "avif": "AVIF"}
PAIR_LABELS = {"vtm/jpeg": "VVC / JPEG", "vtm/heif": "VVC / HEIF", "vtm/avif": "VVC / AVIF"}
THRESHOLD_LABELS = {"T = 40 dB", "T = 41 dB", "T = 42 dB"}
CYRILLIC = re.compile("[Ѐ-ӿ]")
CHOSEN = "D08_r00_c04"  # the largest MGM of the synthetic fragments (ties with D16_r00_c04 - the smaller code)
SIMPLEST = "D01_r00_c00"  # the smallest MGM (ties with D09_r00_c00 - the smaller code)
THRESHOLD_PCC = {"vtm": 9, "jpeg": 67, "heif": 67, "avif": 67}  # the 12th point of every codec below
D10_SHAPE = (1100, 2100)  # H, W of the synthetic D10: a grid of 2 x 4 fragments
D10_MGM = (0.030, 0.035, 0.046, 0.050, 0.0437, 0.052, 0.055, 0.0405)  # its fragments r00_c00 ... r01_c03; the median
# of all fragments (0.041) is none of them, so a caption that prints the MGM of a fragment for the median is seen; by
# the rule (b) is r01_c03 and the next one r01_c00, the owner's (c) r00_c02 is the third - a rule in place of the
# owner's choice is seen
OWNER_C = "D10_r00_c02"  # Fig. 1 (c): the owner's choice, task of cycle 21
FRAGMENT_RHO = (0.7619, -0.98246, 0.2453, -0.95)  # rho of the threshold PCC on MGM by codec; -0.95 is printed -0.950
RHO_PRINTED = ("0.762", "-0.982", "0.245", "-0.950")
FIG4_SERIES = {"simple": f"simple, {SIMPLEST}", "complex": f"complex, {CHOSEN}"}  # the line labels of Fig. 4
NAMED = {"FIGURES[2]": "Dependences of PSNR-HVS-M (a,b), PSNR (c,d) and HaarPSI (e,f) on compression ratio",
         "FIGURES[3]": "Dependences of PSNR-HVS-M on PCC", "FIGURES[7]": f"The fragment {CHOSEN} with the largest MGM"}
# the figures named in the caption of Fig. 2 by the key of their number (cycles 26, 27) - known by their own captions


@pytest.fixture(scope="module")
def fg():
    spec = importlib.util.spec_from_file_location("figures", STUDY / "figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_fragment():
    """512 x 512: a smooth ramp, strong noise in the block (2, 1), weak noise in the block (0, 3)."""
    y, x = np.mgrid[0:512, 0:512]
    image = ((x + y) // 4).astype(np.uint8)
    rng = np.random.default_rng(20261027)
    image[256:384, 128:256] = rng.integers(0, 256, (128, 128))
    image[0:128, 384:512] = rng.integers(100, 140, (128, 128))
    return image


def simple_fragment():
    """512 x 512: the fragment of the simplest structure - a ramp along the rows, unlike the other fragments."""
    y, x = np.mgrid[0:512, 0:512]
    return ((x // 2 + y // 64) % 256).astype(np.uint8)


def whole_d10():
    y, x = np.mgrid[0:D10_SHAPE[0], 0:D10_SHAPE[1]]
    image = ((x // 9 + y // 5) % 256).astype(np.uint8)
    image[::7, ::3] = 255
    return image


def b11_grid(height, width, tile=512):
    """The grid of B11 0.2.0 written again: whole tiles centred, the odd pixel of the remainder right and below."""
    rows, cols = height // tile, width // tile
    y0, x0 = (height - tile * rows) // 2, (width - tile * cols) // 2
    return {f"D10_r{r:02d}_c{c:02d}": (x0 + tile * c, y0 + tile * r) for r in range(rows) for c in range(cols)}


def fake_coder(codec, image, pcc):
    step = CODECS.index(codec) + 2
    return f"{codec}:{pcc}".encode() + image[:2, :8].tobytes(), (image // step) * step


FIXED_PCC = {"vtm": 25, "jpeg": 57, "heif": 43, "avif": 66}


def fixed_run(images, n_fragments):
    """The fields of e5_fixed.json that Table 4 reads: per patient and per whole image the CR at the fixed PCC and the
    loss, r_fix per image, the summary (n, r_fix > 1 counted here) and the units below T (none); the patients and the
    whole images differ in every quantity."""
    patients, whole, ratios = [], [], []
    for i, im in enumerate(images):
        for k, c in enumerate(CODECS):
            patients.append({"level": "пациенты", "image": im, "codec": c, "n": 5, "own_n": 5,
                             "cr_fix": 9.0 + 0.37 * ((i * 7 + k * 3) % 16) + k, "cr_own": 12.0,
                             "loss": 0.7 + 0.013 * ((i * 5 + k) % 16)})
            whole.append({"level": "снимки", "image": im, "codec": c, "n": 1, "own_n": 1, "cr_fix": 11.0 + i / 4 + k,
                          "cr_own": 13.0, "loss": 0.75 + 0.011 * ((i * 3 + k * 2) % 16)})
            if c != "vtm":
                ratios += [{"level": "пациенты", "image": im, "pair": f"vtm/{c}", "r_fix": 0.9 + 0.02 * ((i * 3 + k) % 16)},
                           {"level": "снимки", "image": im, "pair": f"vtm/{c}", "r_fix": 0.95 + 0.015 * ((i * 5 + k * 2) % 16)}]
    summary = []
    for level in ("пациенты", "снимки"):
        summary += [{"level": level, "what": "потеря «фиксированный / свой»", "codec": c, "n": len(images), "above_1": None}
                    for c in CODECS]
        summary += [{"level": level, "what": "r_fix", "codec": f"vtm/{c}", "n": len(images),
                     "above_1": sum(x["r_fix"] > 1 for x in ratios if x["level"] == level and x["pair"] == f"vtm/{c}")}
                    for c in OTHER_CODECS]
    below = [{"level": lv, "codec": c, "n": n, "below": 0, "units": [], "images": []}
             for lv, n in (("фрагменты", n_fragments), ("снимки", len(images))) for c in CODECS]
    return {"t_db": 42.0, "analysis": {"fixed_pcc": FIXED_PCC, "patients": patients, "whole": whole, "ratios": ratios,
                                       "summary": summary, "below": below}}


def synthetic(cache):
    """E4: the header and the points of every fragment; E5, E6 and the grid: the fields the figures and tables read."""
    fragments = [{"unit": f"D{1 + i // 5:02d}_r00_c{i % 5:02d}", "image": f"D{1 + i // 5:02d}",
                  "mgm": round(0.02 + 0.001 * (i % 40), 6)} for i in range(80) if 1 + i // 5 != 10]  # ties: two each
    grid = b11_grid(*D10_SHAPE)
    fragments += [{"unit": unit, "image": "D10", "mgm": m} for unit, m in zip(sorted(grid), D10_MGM)]
    original, whole = original_fragment(), whole_d10()
    rows, units = [], []
    for f in fragments:
        for k, c in enumerate(CODECS):
            params = list(range(-2, 19)) if c == "vtm" else list(range(100, 39, -3))
            for j, p in enumerate(params):
                file_sha = recon_sha = "0" * 64
                if f["unit"] == CHOSEN and p == THRESHOLD_PCC[c]:
                    file, recon = fake_coder(c, original, p)
                    file_sha, recon_sha = hashlib.sha256(file).hexdigest(), hashlib.sha256(recon.tobytes()).hexdigest()
                rows.append({"unit": f["unit"], "codec": c, "param": p, "status": "valid", "cr_file": 2.0 + j + k / 10,
                             "psnr_hvs_m_db": 50.0 - 0.7 * j - k / 5 - 10 * f["mgm"] + 40 * (c == "vtm" and j < 4),
                             # every fragment a curve of its own; VVC at QP -2 to 1 above 80 dB, at QP -2 and -1 left
                             # of the x axis (Fig. 4 (a))
                             "psnr_db": 45.0 - 0.6 * j - k / 4 - f["mgm"], "haarpsi": 0.99 - 0.03 * j - k / 50 - f["mgm"],
                             "file_sha256": file_sha, "recon_sha256": recon_sha})
            rows.append({"unit": f["unit"], "codec": c, "param": 999, "status": "lossless", "cr_file": 1.5,
                         "psnr_hvs_m_db": "+inf", "psnr_db": "+inf", "haarpsi": 1.0, "file_sha256": "0" * 64,
                         "recon_sha256": "0" * 64})
            f[f"pcc_{c}"] = params[11] - (f["mgm"] > 0.05 and f["unit"] != CHOSEN)  # the threshold PCC of E5 (Fig. 5)
            units.append({"unit": f["unit"], "image": f["image"], "level": "f", "codec": c, "t_db": 42.0, "judged": True,
                          "reason": None, "pcc": f[f"pcc_{c}"], "cr": 2.0 + 11 + k / 10,
                          "psnr_hvs_m_db": 50.0 - 0.7 * 11 - k / 5, "psnr_db": 38.5 + k + f["mgm"],
                          "haarpsi": 0.91 + k / 100 - f["mgm"], "mgm": f["mgm"]})
    images = [f"D{i:02d}" for i in range(1, 17)]
    patients = [{"image": im, "codec": c, "t_db": 42.0, "median_cr": 10.0 + i + k} for i, im in enumerate(images)
                for k, c in enumerate(CODECS)]
    ratios = [{"image": im, "pair": f"vtm/{b}", "t_db": 42.0, "r": 1.1 + i / 100 + k / 10, "in_test": True}
              for i, im in enumerate(images) for k, b in enumerate(("jpeg", "heif", "avif"))]
    complexity_patients = [{"image": im, "mgm": round(0.06 - 0.002 * i + (0.0005 if i % 2 else 0), 6)}
                           for i, im in enumerate(images)]  # MGM not in the order of the codes

    def comparisons(level):
        out = []
        for t in (40.0, 41.0, 42.0):
            for k, b in enumerate(("jpeg", "heif", "avif", "jpeg_opt")):
                main = level == "пациенты" and t == 42.0 and b != "jpeg_opt"
                wider = 0.05 if level == "снимки" else 0.0  # the CI of the whole images differs from the patients'
                out.append({"level": level, "pair": f"vtm/{b}", "t_db": t, "n": 16, "ratio": 1.2 + k / 10 + t / 1000,
                            "ratio_low": 1.1 + k / 10 - wider, "ratio_high": 1.3 + k / 10 + wider, "p": 3.0517578125e-05,
                            "p_holm": 9.1552734375e-05 if b != "jpeg_opt" else None,
                            "role": "основное" if main else "описательно",
                            "verdict": (f"VVC выгоднее {b.upper()}" if b != "avif" else
                                        "«VVC выгоднее AVIF» не установлено — отрицательный или неопределённый результат")
                            if main else None})
        return out
    jpeg50 = {"name": "уровень 50 по Fidler et al.",
              "units": [{"unit": f["unit"], "level": "f", "psnr_db": 36.0 + f["mgm"] * 10, "haarpsi": 0.95 - f["mgm"],
                         "cr": 9.0 + f["mgm"] * 100} for f in fragments] +
                       [{"unit": im, "level": "m", "psnr_db": 99.0, "haarpsi": 0.5, "cr": 99.0} for im in images],
              "shares": [{"level": lv, "t_db": t, "n": n, "count": n - 1 - int(t) % 40, "share_pct": 0.0}
                         for lv, n in (("фрагменты", len(fragments)), ("снимки", 16)) for t in (40.0, 41.0, 42.0)],
              "patients": [{"image": im, "n": 10, "median_cr": 7.5 + i / 3, "median_cr_opt": 7.8 + i / 3}
                           for i, im in enumerate(images)]}
    rho = [{"level": "фрагменты", "x": "MGM", "key": f"pcc_{c}", "n": len(fragments), "rho": r}
           for c, r in zip(CODECS, FRAGMENT_RHO)] + \
          [{"level": "пациенты", "x": "MGM", "key": f"pcc_{c}", "n": 16, "rho": 0.1} for c in CODECS]
    e5 = {"thresholds_db": [40.0, 41.0, 42.0],
          "analysis": {"complexity": {"fragments": fragments, "patients": complexity_patients, "rho": rho}, "units": units,
                       "patients": patients, "ratios": ratios, "comparisons": comparisons("пациенты"),
                       "whole_comparisons": comparisons("снимки"), "jpeg50": jpeg50}}
    e4 = {"header": {"codec_versions": {"vtm_gpac_sha256": {"encoder": "a" * 64, "decoder": "b" * 64, "mp4box": "c" * 64},
                                        "jpeg": {"libjpeg_turbo": "3.1.4.1", "pillow": "12.3.0"},
                                        "heif": {"libde265": "libde265 HEVC decoder, version 1.1.3", "libheif": "1.23.6",
                                                 "pillow": "12.3.0", "pillow_heif": "1.8.1",
                                                 "x265": "x265 HEVC encoder (4.3+1-e9b8812)"},
                                        "avif_libheif": {"aom_encoder": "AOMedia Project AV1 Encoder v3.12.0",
                                                         "libheif": "1.19.7", "numpy": "2.5.4", "pillow_heif": "0.22.0",
                                                         "python": "3.13.15"}},
                     "plan": {"vtm": {"f": list(range(-12, 64))}, "jpeg": list(range(1, 101)),
                              "heif": list(range(0, 101)), "avif": list(range(0, 101))}},
          "rows": rows}
    e6 = {"pages": [{"unit": CHOSEN, "list": 1}, {"unit": "D01_r00_c00", "list": 2}]}
    tiles = {"parameters": {"origin": "center", "tile": 512},
             "rows": [{"tile": unit, "code": "D10", "r": int(unit[5:7]), "c": int(unit[9:11]), "x": x, "y": y}
                      for unit, (x, y) in sorted(grid.items())]}
    for name, data in (("cycles/13/e4_main.json", e4), ("cycles/17/e5_analysis.json", e5),
                       ("cycles/16/e6_pages.json", e6), ("cycles/9/p001_tiles.json", tiles),
                       ("cycles/26/e5_fixed.json", fixed_run(images, len(fragments)))):
        (cache / name).parent.mkdir(parents=True, exist_ok=True)
        (cache / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    arrays = {CHOSEN: original, SIMPLEST: simple_fragment(), "D10": whole}
    arrays |= {unit: whole[y:y + 512, x:x + 512].copy() for unit, (x, y) in grid.items()}
    return e4, e5, fragments, arrays


def coders(fg, monkeypatch, arrays, wrong=None):
    """The test's own loader and coder in place of those of E4; `wrong` - a codec whose reconstruction is altered."""
    def load_units(paths, codes):
        return {code: arrays[code].copy() for code in codes}

    def encode(paths, codec, image, pcc, work):
        assert np.array_equal(image, arrays[CHOSEN]) and pcc == THRESHOLD_PCC[codec]
        file, recon = fake_coder(codec, image, pcc)
        if codec == wrong:
            recon = recon.copy()
            recon[0, 0] ^= 1
        return file, recon
    monkeypatch.setattr(fg, "load_units", load_units)
    monkeypatch.setattr(fg, "encode", encode)


@pytest.fixture()
def built(fg, tmp_path, monkeypatch):
    cache, papers = tmp_path / "cache", tmp_path / "papers"
    e4, e5, fragments, arrays = synthetic(cache)
    coders(fg, monkeypatch, arrays)
    pins = {key: hashlib.sha256((cache / name).read_bytes()).hexdigest()
            for key, name in (("e4_pin", "cycles/13/e4_main.json"), ("fixed_pin", "cycles/26/e5_fixed.json"))}
    out, index = fg.build(cache, papers, paths={"cache_root": cache}, **pins, log=lambda *a: None)
    return {"fg": fg, "cache": cache, "papers": papers, "out": out, "e4": e4, "e5": e5, "fragments": fragments,
            "pins": pins, "arrays": arrays, "monkeypatch": monkeypatch}


@pytest.fixture()
def drawn(fg, tmp_path):
    """The synthetic run files and arrays without a build - for the tests that draw a figure themselves."""
    cache = tmp_path / "cache"
    e4, e5, fragments, arrays = synthetic(cache)
    return {"fg": fg, "cache": cache, "e4": e4, "e5": e5, "fragments": fragments, "arrays": arrays}


def sources(built):
    fg, cache = built["fg"], built["cache"]
    return [fg.Source(name, cache / name) for name in ("cycles/13/e4_main.json", "cycles/17/e5_analysis.json",
                                                       "cycles/16/e6_pages.json", "cycles/9/p001_tiles.json")]


def style_of(fg):
    return fg.load_style()


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def typical_by_rule(fragments):
    """(the median, [(b) - the MGM closest to the median, (c) - the owner's choice], the next one by the rule)."""
    median = statistics.median(f["mgm"] for f in fragments)
    ranked = sorted((abs(f["mgm"] - median), f["unit"]) for f in fragments if f["image"] == "D10")
    return median, [ranked[0][1], OWNER_C], ranked[1][1]


def test_files_created_and_existing_files_need_force(fg, built):
    for name in fg.OUTPUTS:
        assert (built["out"] / name).is_file() and (built["out"] / name).stat().st_size > 0, name
    with pytest.raises(fg.RunRefusal, match="--force overwrites them"):
        fg.build(built["cache"], built["papers"], paths={"cache_root": built["cache"]}, **built["pins"],
                 log=lambda *a: None)
    fg.build(built["cache"], built["papers"], paths={"cache_root": built["cache"]}, **built["pins"], force=True,
             log=lambda *a: None)
    assert not list(built["cache"].glob("p001_figures_*"))  # the work folder of the coders is removed


def test_two_builds_write_the_same_bytes_and_no_time_of_the_build(built, tmp_path):
    from openpyxl import load_workbook

    fg = built["fg"]
    again, _ = fg.build(built["cache"], tmp_path / "again", paths={"cache_root": built["cache"]}, **built["pins"],
                        log=lambda *a: None)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731
    first, second = ({name: digest(out / name) for name in fg.OUTPUTS} for out in (built["out"], again))
    assert first == second and len(first) == 34
    trim = lambda row: row[:max([k + 1 for k, v in enumerate(row) if v != ""], default=0)]  # noqa: E731
    checked = {".pdf": 0, ".xlsx": 0}
    for name in fg.OUTPUTS:
        path = built["out"] / name
        if path.suffix == ".pdf":
            data = path.read_bytes()
            assert b"/CreationDate" not in data and b"/ModDate" not in data, name
        elif path.suffix == ".xlsx":
            with zipfile.ZipFile(path) as book:
                assert {entry.date_time for entry in book.infolist()} == {(1980, 1, 1, 0, 0, 0)}, name
                core = book.read("docProps/core.xml").decode("utf-8")
            assert sorted(re.findall(r"<dcterms:(created|modified)\b[^>]*>([^<]*)<", core)) == \
                [("created", "1980-01-01T00:00:00Z"), ("modified", "1980-01-01T00:00:00Z")], name
            cells = [trim(["" if v is None else v for v in row])
                     for row in load_workbook(path).active.iter_rows(values_only=True)]
            assert cells == [trim(row) for row in read_csv(path.with_suffix(".csv"))], name
        else:
            continue
        checked[path.suffix] += 1
    assert checked == {".pdf": 8, ".xlsx": 4}


def test_fragment_choice_by_the_rules_with_ties(fg):
    fragments = [{"unit": u, "image": u[:3], "mgm": m} for u, m in
                 (("D10_r00_c01", 0.75), ("D10_r00_c00", 0.25), ("D10_r01_c00", 0.5625), ("D01_r00_c00", 0.5),
                  ("D02_r00_c00", 0.5), ("D03_r00_c00", 0.125), ("D04_r00_c00", 0.875), ("D05_r00_c00", 0.125),
                  ("D06_r00_c00", 0.875))]
    median, picks = fg.typical_fragments(fragments, "D10", 2)
    assert median == 0.5 and [x["unit"] for x in picks] == ["D10_r01_c00", "D10_r00_c00"]  # 0.0625, then 0.25 by code
    median, picks = fg.typical_fragments(fragments[:3] + fragments[3:5], "D10", 3)
    assert [x["unit"] for x in picks] == ["D10_r01_c00", "D10_r00_c00", "D10_r00_c01"]  # equal distances: smaller code
    simple, complex_ = fg.extreme_fragments(fragments)
    assert (simple["unit"], complex_["unit"]) == ("D03_r00_c00", "D04_r00_c00")
    with pytest.raises(fg.RunRefusal, match="D10 has 3 fragments in E5, 4 are needed"):
        fg.typical_fragments(fragments, "D10", 4)


def test_fig1_initial_data_by_the_rules(drawn):
    built = drawn
    fg = built["fg"]
    style = style_of(fg)
    e4, e5, e6, tiles = sources(built)
    median, want, next_by_rule = typical_by_rule(built["fragments"])
    assert len({want[0], want[1], next_by_rule}) == 3  # the synthetic data tell the owner's (c) from the rule
    plt, _ = fg.setup(style)
    fig, data, series, dpi, panels, caption = fg.fig1(e5, tiles, built["arrays"], style, plt, "png")
    a, b, c = fig.axes
    grid = b11_grid(*D10_SHAPE)
    frames = sorted((p.get_x() + 0.5, p.get_y() + 0.5, p.get_width(), p.get_height(), p.get_edgecolor()) for p in a.patches)
    assert sorted(f[:4] for f in frames) == sorted((x, y, 512, 512) for x, y in grid.values())  # B11
    mark = matplotlib_colour(style["figures"]["fig1_mark_color"])
    marked = sorted((f[0], f[1]) for f in frames if tuple(f[4]) == mark)
    assert marked == sorted(grid[u] for u in want)
    assert sorted(t.get_text() for t in a.texts) == ["b", "c"]
    inside = lambda px, py: [u for u, (x, y) in grid.items() if x <= px < x + 512 and y <= py < y + 512]  # noqa: E731
    assert {t.get_text(): inside(*t.get_position()) for t in a.texts} == {"b": [want[0]], "c": [want[1]]}
    for ax, unit in ((b, want[0]), (c, want[1])):
        (shown,) = ax.get_images()
        x, y = grid[unit]
        assert np.array_equal(np.asarray(shown.get_array())[..., 0], built["arrays"]["D10"][y:y + 512, x:x + 512]), unit
    (shown,) = a.get_images()
    assert np.array_equal(np.asarray(shown.get_array())[..., 1], built["arrays"]["D10"])
    rows = [r for r in data if r.get("kind") == "tile"]
    assert {r["unit"]: r["panel"] for r in rows if r["panel"]} == {want[0]: "b", want[1]: "c"}
    assert [r["mgm"] for r in data if r.get("kind") == "median MGM"] == [median]
    mgm = {f["unit"]: f["mgm"] for f in built["fragments"]}
    assert caption["value"] == (
        "Fig. 1. The dental image D10 with the grid of its 512 × 512 pixel fragments (a); its fragment with MGM closest "
        f"to the median MGM of the {len(built['fragments'])} fragments ({median:.4f}), {want[0]}, MGM = "
        f"{mgm[want[0]]:.4f} (b); and the fragment from the upper middle part of the image, {OWNER_C}, MGM = "
        f"{mgm[OWNER_C]:.4f} (c).")
    plt.close(fig)


def test_fig1_c_is_the_owners_choice_and_a_wrong_code_is_refused(drawn, monkeypatch):
    fg, fragments = drawn["fg"], drawn["fragments"]
    median, want, _ = typical_by_rule(fragments)
    assert fg.FIG1_C == OWNER_C
    got, picks = fg.fig1_fragments(fragments)
    assert got == median and [x["unit"] for x in picks] == want
    for code, reason in (("D10_r02_c00", "D10_r02_c00 is found 0 times in analysis.complexity.fragments of E5"),
                         ("D02_r00_c03", "D02_r00_c03 is a fragment of D02, not of D10"),
                         (want[0], f"{want[0]} is the fragment (b) by the rule as well")):
        monkeypatch.setattr(fg, "FIG1_C", code)
        with pytest.raises(fg.RunRefusal, match=re.escape(f"Fig. 1 (c): {reason}")):
            fg.fig1_fragments(fragments)


def test_fig1_is_one_row_of_one_height_in_the_pdf_and_whole_multiples_in_the_png(drawn):
    built = drawn
    fg = built["fg"]
    style = style_of(fg)
    _, e5, _, tiles = sources(built)
    plt, _ = fg.setup(style)
    H, W = D10_SHAPE
    for target in ("pdf", "png"):
        fig = fg.fig1(e5, tiles, built["arrays"], style, plt, target)[0]
        width_cm, height_cm = (v * 2.54 for v in fig.get_size_inches())
        assert abs(width_cm - 18.1) < 0.01 and height_cm <= 6.0, (target, width_cm, height_cm)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        a, b, c = (ax.get_window_extent(renderer) for ax in fig.axes)
        assert a.x1 < b.x0 and b.x1 < c.x0, target  # one row, in the order (a), (b), (c)
        assert max(a.y0, b.y0, c.y0) < min(a.y1, b.y1, c.y1), target
        letters = [x for x in fig.texts if x.get_text() in ("(a)", "(b)", "(c)")]
        assert sorted(x.get_text() for x in letters) == ["(a)", "(b)", "(c)"]
        for letter in letters:
            box = letter.get_window_extent(renderer)
            assert box.y1 < min(a.y0, b.y0, c.y0), (target, letter.get_text())  # below the rasters, not over them
            assert not any(box.overlaps(x) for x in (a, b, c)), (target, letter.get_text())
        if target == "pdf":
            heights = [x.height for x in (a, b, c)]
            assert max(heights) - min(heights) < 1e-6 and abs(a.y0 - b.y0) < 1e-6 and abs(b.y0 - c.y0) < 1e-6
            assert abs((b.x0 - a.x1) - (c.x0 - b.x1)) < 1e-6  # equal gaps
            assert abs(a.width / a.height - W / H) < 1e-9 and abs(b.width - b.height) < 1e-6
            left, right = a.x0, fig.bbox.width - c.x1
            assert abs(left - right) < 1e-6 and left < 0.01 * fig.bbox.width  # the row spans the width, margin to margin
        else:
            boxes = [tuple(round(v) for v in (x.x0, x.y0, x.width, x.height)) for x in (a, b, c)]
            assert all(abs(v - round(v)) < 1e-6 for x in (a, b, c) for v in (x.x0, x.y0, x.width, x.height))
            assert boxes[0][2:] == (W, H) and boxes[1][2:] == boxes[2][2:] == (512 * 3, 512 * 3)
            assert abs((a.y0 + a.height / 2) - (b.y0 + b.height / 2)) <= 0.5 and b.y0 == c.y0  # centred, not stretched
        plt.close(fig)


def matplotlib_colour(text):
    from matplotlib.colors import to_rgba

    return to_rgba(text)


def test_figure_csv_equal_the_run_files(built):
    by_unit = {f["unit"]: f["mgm"] for f in built["fragments"]}
    fig2 = read_csv(built["out"] / "figures" / "fig2.csv")
    head = fig2[0]
    fragments = {r[head.index("panel")]: (r[head.index("structure")], r[head.index("unit")], float(r[head.index("mgm")]))
                 for r in fig2[1:] if r[head.index("kind")] == "fragment"}
    assert fragments == {"a": ("simple", SIMPLEST, by_unit[SIMPLEST]), "b": ("complex", CHOSEN, by_unit[CHOSEN])}
    assert len(fig2) == 1 + 2 + 2  # the fragments, the boxes of the PNG - no pixel
    fig3 = read_csv(built["out"] / "figures" / "fig3.csv")
    head = fig3[0]
    assert head == ["panel", "fragment", "codec", "pcc", "cr", "psnr_hvs_m_db", "psnr_db", "haarpsi"]  # haarpsi - cycle 25
    for unit, panels in ((SIMPLEST, "ace"), (CHOSEN, "bdf")):
        for panel, field in zip(panels, ("psnr_hvs_m_db", "psnr_db", "haarpsi")):
            want = sorted((r["codec"], r["param"], r["cr_file"], r[field]) for r in built["e4"]["rows"]
                          if r["unit"] == unit and r["status"] == "valid")
            got = sorted((r[head.index("codec")], int(r[head.index("pcc")]), float(r[head.index("cr")]),
                          float(r[head.index(field)])) for r in fig3[1:] if r[head.index("panel")] == panel)
            assert got == want and {r[head.index("fragment")] for r in fig3[1:] if r[head.index("panel")] == panel} == {unit}
    fig4 = read_csv(built["out"] / "figures" / "fig4.csv")
    assert fig4[0] == ["panel", "codec", "fragment", "structure", "pcc", "psnr_hvs_m_db"]
    want = sorted((letter, r["codec"], r["unit"], what, r["param"], r["psnr_hvs_m_db"]) for letter, c in zip("abcd", CODECS)
                  for unit, what in ((SIMPLEST, "simple"), (CHOSEN, "complex")) for r in built["e4"]["rows"]
                  if r["unit"] == unit and r["codec"] == c and r["status"] == "valid")
    assert sorted((r[0], r[1], r[2], r[3], int(r[4]), float(r[5])) for r in fig4[1:]) == want
    fig5 = read_csv(built["out"] / "figures" / "fig5.csv")
    assert fig5[0] == ["panel", "codec", "fragment", "mgm", "threshold_pcc"]
    want = sorted((letter, c, f["unit"], f["mgm"], f[f"pcc_{c}"]) for letter, c in zip("abcd", CODECS)
                  for f in built["fragments"])
    assert sorted((r[0], r[1], r[2], float(r[3]), int(r[4])) for r in fig5[1:]) == want
    a = built["e5"]["analysis"]
    fig6 = read_csv(built["out"] / "figures" / "fig6.csv")
    assert fig6[0] == ["panel", "level", "pair", "t_db", "n", "ratio", "ratio_low", "ratio_high"]
    want = sorted((letter, what, x["pair"], x["t_db"], x["n"], x["ratio"], x["ratio_low"], x["ratio_high"])
                  for letter, what, name in (("a", "patients", "comparisons"), ("b", "whole images", "whole_comparisons"))
                  for x in a[name] if x["pair"] != "vtm/jpeg_opt")
    assert sorted((r[0], r[1], r[2], float(r[3]), int(r[4]), *map(float, r[5:])) for r in fig6[1:]) == want
    mgm = {p["image"]: p["mgm"] for p in a["complexity"]["patients"]}
    fig7 = read_csv(built["out"] / "figures" / "fig7.csv")
    assert fig7[0] == ["panel", "series", "patient", "mgm", "value"]
    want = sorted([("a", p["codec"], p["image"], mgm[p["image"]], p["median_cr"]) for p in a["patients"]] +
                  [("b", x["pair"], x["image"], mgm[x["image"]], x["r"]) for x in a["ratios"]])
    assert sorted((r[0], r[1], r[2], float(r[3]), float(r[4])) for r in fig7[1:]) == want
    fig8 = read_csv(built["out"] / "figures" / "fig8.csv")
    head = fig8[0]
    region = [r for r in fig8[1:] if r[head.index("kind")] == "region"]
    assert len(region) == 1 and (region[0][head.index("row")], region[0][head.index("column")]) == ("2", "1")
    panels = [r for r in fig8[1:] if r[head.index("kind")] == "panel"]
    assert [(r[head.index("codec")], int(r[head.index("pcc")])) for r in panels] == list(THRESHOLD_PCC.items())
    assert len(fig8) == 1 + 16 + 1 + 4 + 10  # blocks, region, panels, the boxes of the PNG - no pixel
    fig1 = read_csv(built["out"] / "figures" / "fig1.csv")
    head = fig1[0]
    assert len([r for r in fig1[1:] if r[head.index("kind")] == "tile"]) == 8 and len(fig1) == 1 + 8 + 1 + 3 + 3


def plotted(built):
    """Figures 3, 7 and 4 drawn - the plots of lines with markers: [(axes, the labels of its data series, the labels of
    its reference lines, legend texts)]; the six panels of Fig. 3 first."""
    fg = built["fg"]
    style = style_of(fg)
    e4, e5, _, _ = sources(built)
    plt, _ = fg.setup(style)
    f3 = fg.fig3(e4, e5, style, plt)[0]
    f7 = fg.fig7(e5, style, plt)[0]
    f4 = fg.fig4(e4, e5, style, plt)[0]
    shared = {t.get_text() for t in f3.legends[0].get_texts()}
    top, bottom = f7.axes
    assert len(f3.axes) == 6  # PSNR-HVS-M, PSNR and HaarPSI (cycle 25)
    panels = [(ax, set(LABELS.values()), THRESHOLD_LABELS if k < 2 else set(), shared) for k, ax in enumerate(f3.axes)]
    panels += [(top, set(LABELS.values()), set(), {t.get_text() for t in top.get_legend().get_texts()}),
               (bottom, set(PAIR_LABELS.values()), {"r = 1"}, {t.get_text() for t in bottom.get_legend().get_texts()})]
    above = {t.get_text() for t in f4.legends[0].get_texts()}
    panels += [(ax, set(FIG4_SERIES.values()), THRESHOLD_LABELS, {t.get_text() for t in ax.get_legend().get_texts()} | above)
               for ax in f4.axes]
    return style, plt, [f3, f7, f4], panels


def test_every_series_is_a_line_with_small_markers_and_in_the_legend(built):
    style, plt, figs, panels = plotted(built)
    size, most = style["page"]["marker_size_pt"], style["page"]["markevery_max"]
    assert size == 2.5
    for ax, series, references, legend in panels:
        lines = {line.get_label(): line for line in ax.get_lines()}
        assert series <= set(lines) and references <= set(lines)
        for name in series:
            line = lines[name]
            assert line.get_linestyle() not in ("None", "none", "", " ") and line.get_linewidth() > 0, name
            assert line.get_marker() not in ("None", "none", "", " ", None), name
            assert line.get_markersize() == size <= 4, (name, line.get_markersize())
            step = line.get_markevery()
            n = len(line.get_xdata())
            assert isinstance(step, int) and 0 < len(range(0, n, step)) <= most, (name, step, n)
            x = list(line.get_xdata())
            assert x == sorted(x), name  # points joined in ascending x
        assert series | references <= legend, legend
    assert {t for _, _, _, legend in panels[:6] for t in legend} == set(LABELS.values()) | THRESHOLD_LABELS
    for fig in figs:
        plt.close(fig)


def test_log_axis_labels_are_plain_numbers(built):
    style, plt, figs, panels = plotted(built)
    figs[0].canvas.draw()
    for ax, *_ in panels[:6]:
        assert ax.get_xscale() == "log"
        labels = [t.get_text() for t in ax.get_xticklabels(which="both") if t.get_text()]
        assert labels and all(re.fullmatch(r"\d+(\.\d+)?", t) for t in labels), labels
        assert not any(s in t for t in labels for s in ("×", "times", "^", "$")), labels
    for fig in figs:
        plt.close(fig)


def test_pcc_axis_labels_without_brackets_and_no_negative_x(drawn):
    """Owner, cycle 24: "PCC" on the plots without its meaning in brackets; no negative value on an x axis."""
    fg = drawn["fg"]
    style = style_of(fg)
    e4, e5, _, _ = sources(drawn)
    plt, _ = fg.setup(style)
    figs = {"fig3": fg.fig3(e4, e5, style, plt)[0], "fig4": fg.fig4(e4, e5, style, plt)[0],
            "fig5": fg.fig5(e5, style, plt)[0], "fig6": fg.fig6(e5, style, plt)[0], "fig7": fg.fig7(e5, style, plt)[0]}
    pcc = []
    for name, fig in figs.items():
        fig.canvas.draw()
        ticks = []
        for ax in fig.axes:
            pcc += [(name, label) for label in (ax.get_xlabel(), ax.get_ylabel()) if "PCC" in label]
            low, high = sorted(ax.get_xlim())
            assert low >= 0, (name, ax.get_xlim())
            ticks += [t.get_text() for t in ax.get_xticklabels(which="both")  # the labels inside the axis - drawn
                      if t.get_text() and low <= t.get_position()[0] <= high]
        assert ticks and not any(t.startswith(("-", "−")) for t in ticks), (name, ticks)  # matplotlib's minus is U+2212
        plt.close(fig)
    assert pcc == [("fig4", "PCC")] * 4 + [("fig5", "Threshold PCC, 42 dB")] * 4  # no bracket, and none left out


def drawn_plot(drawn, name):
    fg = drawn["fg"]
    style = style_of(fg)
    e4, e5, _, _ = sources(drawn)
    plt, _ = fg.setup(style)
    args = (e4, e5) if name in ("fig3", "fig4") else (e5,)
    return style, plt, getattr(fg, name)(*args, style, plt)


def test_fig2_the_fragments_of_the_plots_pixel_for_pixel_and_letters_below(drawn):
    """Fig. 2 (owner, cycle 25): (a) the simple, (b) the complex fragment - the very fragments of Fig. 3 and Fig. 4; the
    pixels of the loader (the grid) as they are, no mark on them; one row, the letters below it."""
    fg = drawn["fg"]
    style = style_of(fg)
    e4, e5, _, tiles = sources(drawn)
    plt, _ = fg.setup(style)
    fig, data, series, dpi, panels, caption = fg.fig2(e5, tiles, drawn["arrays"], style, plt, CHOSEN)
    f3, d3 = fg.fig3(e4, e5, style, plt)[:2]
    f4 = fg.fig4(e4, e5, style, plt)[0]
    units = [p["unit"] for p in panels]
    assert units == [SIMPLEST, CHOSEN]
    in_fig3 = {}
    for r in d3:
        in_fig3.setdefault(r["panel"], set()).add(r["fragment"])
    assert in_fig3 == {letter: {units[k % 2]} for k, letter in enumerate("abcdef")}
    in_fig4 = {x.get_label() for ax in f4.axes for x in ax.get_lines() if not x.get_label().startswith("T = ")}
    assert in_fig4 == {f"simple, {units[0]}", f"complex, {units[1]}"}
    for ax, unit in zip(fig.axes, units):
        (shown,) = ax.get_images()
        got = np.asarray(shown.get_array())
        assert got.dtype == np.uint8 and got.shape == (512, 512, 3) and shown.get_interpolation() == "none", unit
        assert all(np.array_equal(got[..., ch], drawn["arrays"][unit]) for ch in range(3)), unit
        assert not ax.patches and not ax.texts, unit  # nothing drawn over the image
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    a, b = (ax.get_window_extent(renderer) for ax in fig.axes)
    k = style["figures"]["fig2_scale"]
    assert [(round(x.width), round(x.height)) for x in (a, b)] == [(512 * k, 512 * k)] * 2
    assert a.x1 < b.x0 and a.y0 == b.y0  # one row
    width = {"column": style["page"]["column_cm"], "double": style["page"]["double_column_cm"]}[style["figures"]["fig2_width"]]
    assert abs(fig.get_size_inches()[0] * 2.54 - width) < 0.01
    letters = sorted(fig.texts, key=lambda x: x.get_position()[0])
    assert [x.get_text() for x in letters] == ["(a)", "(b)"]
    for letter, box in zip(letters, (a, b)):
        extent = letter.get_window_extent(renderer)
        assert extent.y1 < box.y0 and box.x0 < (extent.x0 + extent.x1) / 2 < box.x1, letter.get_text()
    mgm = {f["unit"]: f["mgm"] for f in drawn["fragments"]}
    assert caption["value"] == (
        f"Fig. 2. The fragments of the plots of Figs. 3 and 4, 512 × 512 pixels: of simple structure {SIMPLEST}, "
        f"MGM = {mgm[SIMPLEST]:.4f} (a) and of complex structure {CHOSEN}, MGM = {mgm[CHOSEN]:.4f} (b), the same "
        "fragment as in Fig. 8.")
    for x in (fig, f3, f4):
        plt.close(x)


def test_fig2_names_the_figure_against_the_original_by_the_numbering(built):
    """Fig. 2 (tasks of cycles 26 and 27): the numbers of the figures of the plots and of the figure against the original
    in the caption of Fig. 2 are the numbers the build gives those figures - with the plots of Fig. 3 and Fig. 4 swapped
    in the order of mention the caption names "Figs. 4 and 3"; with the last two figures swapped it names Fig. 7, and
    the build refuses captions that keep their old numbers; a figure against the original of another fragment than
    Fig. 2 (b) is refused."""
    fg, monkeypatch = built["fg"], built["monkeypatch"]
    style = style_of(fg)
    _, e5, _, tiles = sources(built)
    plt, _ = fg.setup(style)
    caption = lambda compared: fg.fig2(e5, tiles, built["arrays"], style, plt, compared)  # noqa: E731
    first = caption(CHOSEN)
    mgm = {f["unit"]: f["mgm"] for f in built["fragments"]}
    assert first[5]["value"].startswith("Fig. 2. The fragments of the plots of Figs. 3 and 4, ")
    assert first[5]["value"].endswith(f"{CHOSEN}, MGM = {mgm[CHOSEN]:.4f} (b), the same fragment as in Fig. 8.")
    plt.close(first[0])
    with pytest.raises(fg.RunRefusal, match=f"Fig. 2 \\(b\\) is {CHOSEN}, the figure against the original shows "
                                            f"{SIMPLEST}: the caption names them the same fragment"):
        caption(SIMPLEST)
    order = list(fg.FIGURES)
    assert order[2:4] == ["metrics on CR", "PSNR-HVS-M on PCC"] and order[7] == "against the original"
    plots = order[:2] + order[3:1:-1] + order[4:]
    monkeypatch.setattr(fg, "FIGURES", tuple(plots))
    swapped = caption(CHOSEN)
    assert swapped[5]["value"].startswith("Fig. 2. The fragments of the plots of Figs. 4 and 3, ")
    assert swapped[5]["value"].endswith(" (b), the same fragment as in Fig. 8.")
    plt.close(swapped[0])
    order[6], order[7] = order[7], order[6]
    monkeypatch.setattr(fg, "FIGURES", tuple(order))
    moved = caption(CHOSEN)
    assert moved[5]["value"].startswith("Fig. 2. The fragments of the plots of Figs. 3 and 4, ")
    assert moved[5]["value"].endswith(" (b), the same fragment as in Fig. 7.")
    plt.close(moved[0])
    with pytest.raises(fg.RunRefusal, match="the caption of the figure 'against the original' does not begin with "
                                            "'Fig. 7.' - its place in FIGURES"):
        fg.build(built["cache"], built["papers"], paths={"cache_root": built["cache"]}, **built["pins"], force=True,
                 log=lambda *a: None)


def test_fig3_haarpsi_panels_equal_e4_without_threshold_lines(drawn):
    """Fig. 3 (owner, cycle 25): the third row (e), (f) - HaarPSI of the valid E4 rows on CR of the simple and the complex
    fragment, a line per codec on the logarithmic CR axis, no threshold line below the first row."""
    style, plt, (fig, *_) = drawn_plot(drawn, "fig3")
    assert len(fig.axes) == 6
    for ax, unit in ((fig.axes[4], SIMPLEST), (fig.axes[5], CHOSEN)):
        lines = {x.get_label(): x for x in ax.get_lines()}
        assert set(lines) == set(LABELS.values()), set(lines)
        for c in CODECS:
            want = sorted((r["cr_file"], r["haarpsi"]) for r in drawn["e4"]["rows"]
                          if r["unit"] == unit and r["codec"] == c and r["status"] == "valid")
            assert list(zip(lines[LABELS[c]].get_xdata(), lines[LABELS[c]].get_ydata())) == want, (unit, c)
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_xscale()) == ("CR", "HaarPSI", "log")
    assert [ax.get_title(loc="left") for ax in fig.axes] == [f"({x}) {what} structure" for x, what in
                                                            zip("abcdef", ("simple", "complex") * 3)]
    assert not [x for ax in fig.axes[2:] for x in ax.get_lines() if x.get_label().startswith("T = ")]
    plt.close(fig)


def test_fig4_two_fragments_per_codec_solid_and_dashed_over_the_plan(drawn):
    style, plt, (fig, *_) = drawn_plot(drawn, "fig4")
    plan = drawn["e4"]["header"]["plan"]
    assert len(fig.axes) == 4
    for k, (ax, c) in enumerate(zip(fig.axes, CODECS)):
        assert ax.get_title(loc="left") == f"({'abcd'[k]}) {LABELS[c]}"
        lines = {x.get_label(): x for x in ax.get_lines()}
        for unit, what, dash in ((SIMPLEST, "simple", "-"), (CHOSEN, "complex", "--")):
            line = lines[FIG4_SERIES[what]]
            want = sorted((r["param"], r["psnr_hvs_m_db"]) for r in drawn["e4"]["rows"]
                          if r["unit"] == unit and r["codec"] == c and r["status"] == "valid")
            assert list(zip(line.get_xdata(), line.get_ydata())) == want, (c, what)
            assert line.get_linestyle() == dash and line.get_marker() == style["codecs"][c]["marker"], (c, what)
            assert matplotlib_colour(line.get_color()) == matplotlib_colour(style["codecs"][c]["color"]), (c, what)
        values = plan["vtm"]["f"] if c == "vtm" else plan[c]
        # the x axis from the style key of 0 (cycle 24); in (a) the points of QP below 0 are still in the data
        assert ax.get_xlim() == (style["figures"]["fig4_pcc_min"], max(values)) == (0, max(values)), c
        assert ax.get_xlabel() == "PCC", c
        if c == "vtm":
            assert all(min(lines[name].get_xdata()) < 0 for name in FIG4_SERIES.values())
        # the y axis: the bottom - matplotlib's automatic margin of 5 % over all the data of the panel, written again;
        # the top - the same in (b)-(d), the style key of 80 dB in (a) (cycle 23), the points above it still in the data
        ys = [y for x in ax.get_lines() for y in x.get_ydata()]
        low, high = min(ys), max(ys)
        assert ax.get_ylim()[0] == pytest.approx(low - 0.05 * (high - low)), c
        if c == "vtm":
            assert ax.get_ylim()[1] == style["figures"]["fig4_vvc_ymax_db"] == 80 < high, (ax.get_ylim(), high)
        else:
            assert ax.get_ylim()[1] == pytest.approx(high + 0.05 * (high - low)), c
    plt.close(fig)


def test_fig5_one_point_per_fragment_without_lines_and_rho_of_e5(drawn):
    style, plt, (fig, *_) = drawn_plot(drawn, "fig5")
    assert len(fig.axes) == 4
    for k, (ax, c) in enumerate(zip(fig.axes, CODECS)):
        (points,) = ax.get_lines()
        assert points.get_linestyle() == "None" and points.get_marker() == style["codecs"][c]["marker"], c
        assert points.get_markersize() == 2.5, c
        assert matplotlib_colour(points.get_color()) == matplotlib_colour(style["codecs"][c]["color"]), c
        want = sorted((f["mgm"], f[f"pcc_{c}"]) for f in drawn["fragments"])
        assert sorted(zip(points.get_xdata(), points.get_ydata())) == want and len(want) == len(drawn["fragments"]), c
        assert ax.get_title(loc="left") == f"({'abcd'[k]}) {LABELS[c]}, ρ = {RHO_PRINTED[k]}"
        assert all(float(y).is_integer() for y in ax.get_yticks()), c  # PCC is a whole number
    plt.close(fig)


def test_fig6_ci_segments_and_estimates_equal_e5_and_the_line_r_1(drawn):
    style, plt, (fig, *_) = drawn_plot(drawn, "fig6")
    a = drawn["e5"]["analysis"]
    second = {f"vtm/{b}": b for b in ("jpeg", "heif", "avif")}
    segments = 0
    for ax, name in zip(fig.axes, ("comparisons", "whole_comparisons")):
        lines = ax.get_lines()
        unity = [x for x in lines if [float(v) for v in x.get_xdata()] == [1.0, 1.0] and x.get_linestyle() == "--"]
        bars = [x for x in lines if x.get_linestyle() == "-"]
        dots = [x for x in lines if x.get_linestyle() == "None"]
        assert len(unity) == 1 and len(bars) == len(dots) == 9 and len(lines) == 19, name
        labels = {round(y): t.get_text() for y, t in zip(fig.axes[0].get_yticks(), fig.axes[0].get_yticklabels())}
        got = {}
        for x in bars:
            (y0, y1), (x0, x1) = x.get_ydata(), x.get_xdata()
            assert y0 == y1 and x.get_marker() in ("None", "none", None, ""), name
            got[y0] = [labels[round(y0)], matplotlib_colour(x.get_color()), x0, x1]
        for x in dots:
            (y,), (r,) = x.get_ydata(), x.get_xdata()
            assert x.get_markersize() == 2.5 and matplotlib_colour(x.get_color()) == got[y][1], name
            got[y].append(r)
        want = sorted((f"VVC / {LABELS[second[x['pair']]]}, {x['t_db']:.0f} dB",
                       matplotlib_colour(style["codecs"][second[x["pair"]]]["color"]),
                       x["ratio_low"], x["ratio_high"], x["ratio"]) for x in a[name] if x["pair"] in second)
        assert sorted(tuple(v) for v in got.values()) == want, name
        segments += len(bars)
    assert segments == 18
    plt.close(fig)


def test_fig5_and_fig6_refuse_an_n_that_differs_from_the_printed_one(drawn):
    fg = drawn["fg"]
    style = style_of(fg)
    plt, _ = fg.setup(style)
    _, e5, _, _ = sources(drawn)
    for x in e5.data["analysis"]["complexity"]["rho"]:
        x["n"] = 7 if x["level"] == "фрагменты" else x["n"]
    with pytest.raises(fg.RunRefusal, match=re.escape(f"Fig. 5: rho of E5 is over 7 fragments, the panels show "
                                                      f"{len(drawn['fragments'])}")):
        fg.fig5(e5, style, plt)
    _, e5, _, _ = sources(drawn)
    for x in e5.data["analysis"]["whole_comparisons"]:
        x["n"] = 15
    with pytest.raises(fg.RunRefusal, match=re.escape("Fig. 6: n = 16 patients and n = 15 whole images")):
        fg.fig6(e5, style, plt)
    plt.close("all")


def sobel_mgm(block):
    """MGM written again: the 3 x 3 Sobel sums by kernel offsets on the inner pixels, the mean of the modulus."""
    x = block.astype(np.float64) / 255.0
    h, w = x.shape
    kx = ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))
    gx = sum(kx[i][j] * x[i:h - 2 + i, j:w - 2 + j] for i in range(3) for j in range(3))
    gy = sum(kx[j][i] * x[i:h - 2 + i, j:w - 2 + j] for i in range(3) for j in range(3))
    return float(np.hypot(gx, gy).mean())


def expected_region(image):
    scores = [(sobel_mgm(image[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128]), r, c) for r in range(4) for c in range(4)]
    best = max(s for s, _, _ in scores)
    return min((r, c) for s, r, c in scores if s == best)


def by_repetition(part, k):
    rows, cols = np.arange(part.shape[0] * k) // k, np.arange(part.shape[1] * k) // k
    return part[rows][:, cols]


def drawn_fig8(built, tmp_path):
    fg = built["fg"]
    style = style_of(fg)
    e4, e5, e6, _ = sources(built)
    unit, original, recons = fg.comparison(e4, e5, e6, built["arrays"], {"cache_root": built["cache"]}, tmp_path,
                                           lambda *a: None)
    plt, _ = fg.setup(style)
    return plt, unit, original, fg.fig8(e5, e6, unit, original, recons, style, plt)


def test_fig8_panels_are_the_reconstructions_and_their_region_by_repetition(built, tmp_path):
    plt, unit, original, (fig, data, series, dpi, panels, caption) = drawn_fig8(built, tmp_path)
    assert unit == CHOSEN and np.array_equal(original, built["arrays"][CHOSEN])
    r, c = expected_region(original)
    assert (r, c) == (2, 1)
    images = [original] + [fake_coder(k, original, THRESHOLD_PCC[k])[1] for k in CODECS]
    axes = np.array(fig.axes).reshape(5, 2).T  # per column: the upper panel, then the lower
    for k, image in enumerate(images):
        for ax, want in ((axes[0, k], image),
                         (axes[1, k], by_repetition(image[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128], 4))):
            (shown,) = ax.get_images()
            got = np.asarray(shown.get_array())
            assert got.dtype == np.uint8 and got.shape == (512, 512, 3), k
            assert all(np.array_equal(got[..., ch], want) for ch in range(3)), k  # pixel for pixel, grey levels as they are
            assert shown.get_interpolation() == "none", k
        (frame,) = axes[0, k].patches
        assert (frame.get_x(), frame.get_y(), frame.get_width(), frame.get_height()) == (c * 128 - 0.5, r * 128 - 0.5, 128, 128)
    titles = [ax.get_title() for ax in axes[0]]
    assert titles[0].split("\n")[0] == "Original"
    assert [t.split("\n")[0] for t in titles[1:]] == ["VVC", "JPEG", "HEIF", "AVIF"]
    assert all(t.split("\n")[1].startswith(f"PCC {THRESHOLD_PCC[k]}, ") for k, t in zip(CODECS, titles[1:]))
    plt.close(fig)


def test_fig8_titles_do_not_overlap(built, tmp_path):
    plt, _, _, (fig, *_) = drawn_fig8(built, tmp_path)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = sorted((ax.title.get_window_extent(renderer) for ax in fig.axes if ax.get_title()), key=lambda b: b.x0)
    assert len(boxes) == 5
    for left, right in zip(boxes, boxes[1:]):
        assert left.x1 < right.x0, (left, right)
    width, height = fig.canvas.get_width_height()
    assert all(b.x0 >= 0 and b.x1 <= width and b.y1 <= height for b in boxes)
    plt.close(fig)


def test_fig8_region_ties_go_to_the_smaller_row_then_column(fg):
    rng = np.random.default_rng(5)
    noise = rng.integers(0, 256, (128, 128)).astype(np.uint8)
    flat = np.full((512, 512), 77, np.uint8)
    assert fg.region_of(flat)[:2] == (0, 0)
    for blocks, want in ((((3, 0), (1, 3)), (1, 3)), (((2, 3), (2, 0)), (2, 0))):
        image = flat.copy()
        for r, c in blocks:
            image[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128] = noise
        assert fg.region_of(image)[:2] == want == expected_region(image)


def test_fig8_refuses_a_reconstruction_that_differs_from_e4(built, tmp_path):
    fg = built["fg"]
    e4, e5, e6, _ = sources(built)
    coders(fg, built["monkeypatch"], built["arrays"], wrong="heif")
    with pytest.raises(fg.RunRefusal, match="heif PCC 67: encoded again, the reconstruction differs from E4 by sha256"):
        fg.comparison(e4, e5, e6, built["arrays"], {"cache_root": built["cache"]}, tmp_path, lambda *a: None)


def test_png_image_panels_are_whole_multiples_pixel_for_pixel(built):
    from PIL import Image

    original = built["arrays"][CHOSEN]
    r, c = expected_region(original)
    region = lambda image: by_repetition(image[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128], 4)  # noqa: E731
    recon = {k: fake_coder(k, original, THRESHOLD_PCC[k])[1] for k in CODECS}
    grid = b11_grid(*D10_SHAPE)
    whole = built["arrays"]["D10"]
    _, want, _ = typical_by_rule(built["fragments"])
    crop = {u: whole[grid[u][1]:grid[u][1] + 512, grid[u][0]:grid[u][0] + 512] for u in want}
    expected = {("fig1", "a"): (whole, True), ("fig1", "b"): (crop[want[0]], False), ("fig1", "c"): (crop[want[1]], False),
                ("fig2", "a"): (built["arrays"][SIMPLEST], False), ("fig2", "b"): (original, False),
                ("fig8", "upper original"): (original, True), ("fig8", "lower original"): (region(original), False)}
    for k in CODECS:
        expected[("fig8", f"upper {k}")] = (recon[k], True)
        expected[("fig8", f"lower {k}")] = (region(recon[k]), False)
    checked = 0
    for name in ("fig1", "fig2", "fig8"):
        png = np.asarray(Image.open(built["out"] / "figures" / f"{name}.png").convert("RGB"))
        rows = read_csv(built["out"] / "figures" / f"{name}.csv")
        head = rows[0]
        for row in (dict(zip(head, x)) for x in rows[1:] if x[head.index("kind")] == "png panel"):
            array, marks = expected[(name, row["panel"])]
            x, y, w, h, k = (int(row[f]) for f in ("png_x", "png_y", "png_w", "png_h", "scale"))
            assert row["scale"] == str(k) and k >= 1 and (w, h) == (array.shape[1] * k, array.shape[0] * k), row
            got, want_ = png[y:y + h, x:x + w], by_repetition(array, k)
            flat = (got[..., 0] == got[..., 1]) & (got[..., 1] == got[..., 2])
            assert np.array_equal(got[..., 0][flat], want_[flat]), row
            assert float((~flat).mean()) <= (0.05 if marks else 0.0), row
            checked += 1
    assert checked == 15


def test_no_vvc_vtm_in_any_output(built, tmp_path):
    from openpyxl import load_workbook

    fg = built["fg"]
    style = style_of(fg)
    assert style["codecs"]["vtm"]["label"] == "VVC"
    for path in built["out"].rglob("*"):
        if path.suffix in (".csv", ".txt"):
            assert "VVC (VTM)" not in path.read_text(encoding="utf-8"), path.name
        elif path.suffix == ".xlsx":
            for row in load_workbook(path).active.iter_rows(values_only=True):
                assert not any(isinstance(v, str) and "VVC (VTM)" in v for v in row), path.name
    e4, e5, e6, tiles = sources(built)
    plt, _ = fg.setup(style)
    figs = [fg.fig1(e5, tiles, built["arrays"], style, plt, "png")[0], fg.fig1(e5, tiles, built["arrays"], style, plt, "pdf")[0],
            fg.fig2(e5, tiles, built["arrays"], style, plt, CHOSEN)[0], fg.fig3(e4, e5, style, plt)[0],
            fg.fig4(e4, e5, style, plt)[0], fg.fig5(e5, style, plt)[0], fg.fig6(e5, style, plt)[0],
            fg.fig7(e5, style, plt)[0], drawn_fig8(built, tmp_path)[3][0]]
    texts = [t.get_text() for fig in figs for t in fig.findobj(match=lambda o: hasattr(o, "get_text"))]
    assert texts and not any("VVC (VTM)" in t or CYRILLIC.search(t) for t in texts)
    for fig in figs:
        plt.close(fig)


def test_style_table_four_colours_lines_markers(built):
    fg = built["fg"]
    codecs = style_of(fg)["codecs"]
    for field in ("color", "line", "marker", "label"):
        assert len({codecs[c][field] for c in CODECS}) == 4, field
    assert [codecs[c]["color"] for c in CODECS] == ["#0072B2", "#D55E00", "#000000", "#CC79A7"]


def percentile(values, q):
    """Linear percentile, written again."""
    v = sorted(values)
    pos = (len(v) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


TOKEN = re.compile(r"([A-Za-z_]\w*)(?:\[([^\]]*)\])?")


def resolve(data, key):
    node = data
    for name, inside in TOKEN.findall(key):
        node = [x[name] for x in node] if isinstance(node, list) else node[name]
        if inside.isdigit():
            node = node[int(inside)]
        elif inside:
            conditions = [c.split("=", 1) for c in inside.split(",")]
            node = [x for x in node if all(str(x[k]) == v for k, v in conditions)]
    return node


def version_number(text):
    """The first word that starts with a digit (or v and a digit) and has a dot, up to a dash - written again."""
    for word in re.split(r"[ ()]+", text):
        if re.match(r"v?\d+\.", word):
            return word.lstrip("v").split("-")[0]
    raise AssertionError(text)


def only(values):
    assert len(set(values)) == 1, values
    return values[0]


def apply(op, node):
    numbers = lambda xs: [math.inf if x == "+inf" else x for x in xs]  # noqa: E731
    return {"value": lambda: node, "count": lambda: len(node), "median": lambda: percentile(numbers(node), 50),
            "q25": lambda: percentile(numbers(node), 25), "q75": lambda: percentile(numbers(node), 75),
            "min": lambda: min(numbers(node)), "max": lambda: max(numbers(node)),
            "range": lambda: f"{min(node)} to {max(node)}",
            "list": lambda: "; ".join(f"{x:.0f}" for x in node),
            "verdict": lambda: ("descriptive" if node is None else "VVC is better than " + node[len("VVC выгоднее "):]
                                if node.startswith("VVC выгоднее ") else "not established"),
            "name": lambda: "level 50 by Fidler et al." if node.startswith("уровень 50 по Fidler") else "JPEG PCC 50",
            "version": lambda: version_number(node), "same": lambda: only(node),
            }[op]()


def printed(value, fmt):
    if value is None:
        return "-"
    if fmt == "text":
        return str(value)
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return format(value, fmt)


def test_every_index_value_equals_its_key(built):
    index = read_csv(built["out"] / "tables" / "numbers_index.csv")
    head = index[0]
    assert head == ["id", "value", "file", "sha256", "key", "op", "format"]
    checked, numbers = 0, []
    captions = (built["out"] / "figures" / "captions.txt").read_text(encoding="utf-8").strip().split("\n\n")
    for row in index[1:]:
        item = dict(zip(head, row))
        if item["key"] == "constant":
            assert item["file"] == built["fg"].SCRIPT and item["op"] == "text"
            continue
        if item["op"] == "figure number":  # the number of a figure named in a caption (cycles 26, 27): the caption under
            # that number is that figure, known by its own words
            assert (item["file"], item["sha256"]) == (built["fg"].SCRIPT, ""), item["id"]
            assert captions[int(item["value"]) - 1].startswith(f"Fig. {item['value']}. {NAMED[item['key']]}"), item["id"]
            numbers.append(item["id"])
            continue
        style = item["file"] == "core/studies/P001_vvc-dental/figures_style.toml"  # a style key printed (cycle 23)
        path = STUDY / "figures_style.toml" if style else built["cache"] / item["file"]
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), item["id"]
        data = (tomllib.loads if style else json.loads)(path.read_text(encoding="utf-8"))
        assert printed(apply(item["op"], resolve(data, item["key"])), item["format"]) == item["value"], item["id"]
        checked += 1
    assert checked > 200 and numbers == ["Caption Fig. 2 / plots, first figure", "Caption Fig. 2 / plots, second figure",
                                         "Caption Fig. 2 / same fragment, figure"]
    ids = [r[0] for r in index[1:]]
    assert len(ids) == len(set(ids))
    for prefix in ("Fig. 1 / ", "Fig. 2 / ", "Fig. 3 (a) / ", "Fig. 3 (e) / ", "Fig. 4 (b) / ", "Fig. 5 (d) / ",
                   "Fig. 6 (a) / ", "Fig. 7 (b) / ", "Fig. 8 / ", "Table 1 / ", "Table 2 / note / ", "Table 3 / ",
                   "Table 4 / head / ", "Table 4 / row 2 / ", "Table 4 / note / ", "Caption Fig. 1 / ", "Caption Fig. 2 / ",
                   "Caption Fig. 6 / ", "Caption Fig. 8 / ", "Caption Table 3 / ", "Caption Table 4 / "):
        assert any(i.startswith(prefix) for i in ids), prefix
    # the former Fig. 2 to Fig. 7 are Fig. 3 to Fig. 8 (cycle 25): their rows moved, none is left under the old number
    for moved, old in (("Fig. {} / threshold lines, dB", 2), ("Caption Fig. {} / axis top (a)", 3), ("Fig. {} / n of rho", 4),
                       ("Fig. {} (b) / n whole images", 5), ("Fig. {} / x: MGM of the patients", 6),
                       ("Caption Fig. {} / n patients", 6), ("Fig. {} / fragment", 7), ("Caption Fig. {} / zoom", 7)):
        assert moved.format(old + 1) in ids and moved.format(old) not in ids, moved
    assert "Fig. 4 (a) / PCC range of the plan" in ids and not [i for i in ids if "x: PCC range" in i]  # owner, cycle 24


def table_place(name, table):
    """(row, column, a part or not) of the cell an index id names after "Table n / ": "row r / <head>[ / part]",
    "head / k[ / part]" or "note[ / part]"; a head may hold " / " (Table 4), so it is matched against the head row."""
    kind, _, rest = name.partition(" / ")
    if kind == "note":
        return len(table) - 1, 0, rest != ""
    if kind == "head":
        k, _, part = rest.partition(" / ")
        return 0, int(k) - 1, part != ""
    head = max((h for h in table[0] if rest == h or rest.startswith(h + " / ")), key=len)
    return int(kind.removeprefix("row ")), table[0].index(head), rest != head


def test_every_table_row_of_the_index_is_its_cell(built):
    tables = {n: read_csv(built["out"] / "tables" / f"table{n}.csv") for n in (1, 2, 3, 4)}
    rows = 0
    for row in read_csv(built["out"] / "tables" / "numbers_index.csv")[1:]:
        label, _, name = row[0].partition(" / ")
        if not label.startswith("Table "):
            continue
        table = tables[int(label.removeprefix("Table "))]
        r, c, part = table_place(name, table)
        if name.startswith("note"):
            assert len(table[-1]) == 1, row[0]
        text = table[r][c]
        assert (row[1] in text) if part else (text == row[1]), (row[0], row[1], text)
        rows += 1
    assert rows > 100


def test_every_digit_of_the_captions_is_an_index_row(built):
    captions = (built["out"] / "figures" / "captions.txt").read_text(encoding="utf-8").strip().split("\n\n")
    labels = [re.match(r"(Fig\. \d|Table \d)\. ", c).group(1) for c in captions]
    assert labels == [f"Fig. {n}" for n in range(1, 9)] + ["Table 1", "Table 2", "Table 3", "Table 4"]
    assert captions[-1] == (  # the task of cycle 27 with the words of cycle 28
        "Table 4. At one fixed PCC per codec for all units: CR over 16 patients; the ratio of CR at the fixed PCC to "
        "CR at the threshold PCC of the unit for T = 42 dB and r_fix = CR(VVC) / CR(codec) for the patients and the "
        "whole images.")
    index = read_csv(built["out"] / "tables" / "numbers_index.csv")[1:]
    mgm = {f["unit"]: f["mgm"] for f in built["fragments"]}
    assert captions[1:3] == [  # the forms of the task of cycle 25 with the words of the task of cycle 26
        f"Fig. 2. The fragments of the plots of Figs. 3 and 4, 512 × 512 pixels: of simple structure {SIMPLEST}, "
        f"MGM = {mgm[SIMPLEST]:.4f} (a) and of complex structure {CHOSEN}, MGM = {mgm[CHOSEN]:.4f} (b), the same "
        "fragment as in Fig. 8.",
        "Fig. 3. Dependences of PSNR-HVS-M (a,b), PSNR (c,d) and HaarPSI (e,f) on compression ratio (CR) for the "
        f"fragments of simple structure {SIMPLEST}, MGM = {mgm[SIMPLEST]:.4f} (a,c,e) and complex structure {CHOSEN}, "
        f"MGM = {mgm[CHOSEN]:.4f} (b,d,f); horizontal lines in (a,b) - the thresholds T = 40, 41 and 42 dB."]
    assert captions[3:6] == [  # the forms of the task of cycle 22, renumbered in cycle 25
        "Fig. 4. Dependences of PSNR-HVS-M on PCC for VVC (a), JPEG (b), HEIF (c) and AVIF (d) for the fragments of "
        f"simple structure {SIMPLEST} (solid lines) and complex structure {CHOSEN} (dashed lines); horizontal lines - "
        "the thresholds T = 40, 41 and 42 dB; in (a) the axes are limited to PCC ≥ 0 and 80 dB.",
        f"Fig. 5. Threshold PCC at T = 42 dB versus MGM for the {len(built['fragments'])} fragments: VVC (a), JPEG (b), "
        "HEIF (c) and AVIF (d); ρ - Spearman rank correlation.",
        "Fig. 6. The effect r = CR(VVC) / CR(codec) (Hodges-Lehmann estimate) with its 95% CI for the patients (a) and "
        "the whole images (b) at T = 40, 41 and 42 dB; 16 patients and 16 whole images; dashed line - r = 1."]  # cycle 28
    assert captions[6] == (  # cycle 28
        "Fig. 7. Dependences of the median CR at T = 42 dB (a) and of r = CR(VVC) / CR(codec) (b) on the MGM of the "
        "patient (the median over the fragments of the image) for 16 patients; dashed line - r = 1.")
    assert captions[10] == (  # cycle 28
        "Table 3. At the threshold T = 42 dB by codec and at the point JPEG PCC 50: CR over 16 patients; the threshold "
        f"PCC, PSNR and HaarPSI over {len(built['fragments'])} fragments; at JPEG PCC 50 - the share of the units with "
        "PSNR-HVS-M ≥ 42 dB.")
    assert captions[7].startswith("Fig. 8. The fragment D08_r00_c04 with the largest MGM")
    for label, text in zip(labels, captions):
        rest = text.removeprefix(f"{label}. ")
        values = sorted((r[1] for r in index if r[0].startswith(f"Caption {label} / ")), key=len, reverse=True)
        for value in values:
            assert value in rest, (label, value)
            rest = rest.replace(value, "", 1)
        assert not re.search(r"\d", rest), (label, rest)
        assert "VVC (VTM)" not in text and "Dependencies" not in text and not CYRILLIC.search(text)


def test_tables_decimal_point_no_cyrillic_no_sha256(built):
    from openpyxl import load_workbook

    for n in (1, 2, 3, 4):
        for line in read_csv(built["out"] / "tables" / f"table{n}.csv"):
            for value in line:
                assert not re.search(r"\d,\d", value) and not CYRILLIC.search(value), (n, value)
        sheet = load_workbook(built["out"] / "tables" / f"table{n}.xlsx").active
        for row in sheet.iter_rows(values_only=True):
            assert not any(isinstance(v, str) and (CYRILLIC.search(v) or re.search(r"\d,\d", v)) for v in row)
    table1 = read_csv(built["out"] / "tables" / "table1.csv")
    for value in (v for line in table1 for v in line):
        assert not re.search(r"[0-9a-f]{16}", value) and "numpy" not in value and "python" not in value, value
    assert table1[1][:2] == ["VVC", "VTM 23.0 all-intra + GPAC MP4Box 26.08-DEV"]
    assert table1[3][1] == "pillow-heif 1.8.1 / libheif 1.23.6 / x265 4.3+1"
    assert table1[4][1] == "pillow-heif 0.22.0 / libheif 1.19.7 / libaom 3.12.0"
    table2 = read_csv(built["out"] / "tables" / "table2.csv")
    assert table2[0][3] == "Patients, 42 dB" and table2[0][4] == "Whole images, 40 dB"
    assert table2[1] == ["VVC / JPEG", "1.240 (1.100–1.300)", "1.241 (1.100–1.300)", "1.242 (1.100–1.300)",
                         "1.240 (1.050–1.350)", "1.241 (1.050–1.350)", "1.242 (1.050–1.350)", "9.16e-05",
                         "VVC is better than JPEG"]  # cycle 28
    assert table2[2][-1] == "VVC is better than HEIF" and table2[3][-1] == "not established"
    assert table2[4][-2:] == ["-", "-"]
    assert table2[5] == [  # cycle 28
        "Note. Effect - the Hodges-Lehmann estimate of r = CR(VVC) / CR(codec) with its 95% CI; 16 patients and 16 whole "
        "images in every comparison; exact two-sided Wilcoxon signed-rank test, p = 3.05e-05 in every comparison; "
        "p (Holm) = 9.16e-05 for the three main comparisons (42 dB, patients)."]
    table3 = read_csv(built["out"] / "tables" / "table3.csv")
    n = len(built["fragments"])
    assert table3[0][4] == f"Threshold PCC for T = 42 dB, median (min–max), {n} fragments"
    pcc = sorted(u["pcc"] for u in built["e5"]["analysis"]["units"] if u["codec"] == "vtm")
    assert table3[1][0] == "VVC" and table3[1][4] == f"{statistics.median(pcc):g} ({pcc[0]}–{pcc[-1]})"
    assert table3[5][0] == "JPEG PCC 50 (level 50 by Fidler et al.)" and table3[5][4] == "50"
    assert table3[5][-1] == f"{n - 3} of {n}; 13 of 16"


def test_table4_from_the_records_of_e5_fixed_by_another_path(built, tmp_path):
    """Table 4 (task of cycle 27): every cell from the records of e5_fixed.json by the test's own median, quartiles and
    count of r_fix > 1 (not the functions of the build); the VVC row has "-" for r_fix; every digit of the head, the
    cells and the note belongs to an index row of its cell; a unit below T in the run file is refused."""
    fg = built["fg"]
    fx = json.loads((built["cache"] / "cycles/26/e5_fixed.json").read_text(encoding="utf-8"))["analysis"]
    table = read_csv(built["out"] / "tables" / "table4.csv")
    m3 = lambda v: f"{percentile(v, 50):.3f} ({min(v):.3f}–{max(v):.3f})"  # noqa: E731
    of = lambda name, level, c, field: [x[field] for x in fx[name] if x["codec" if name != "ratios" else "pair"] == c
                                        and (level is None or x["level"] == level)]  # noqa: E731
    assert table[0] == ["Codec", "Fixed PCC", "CR at the fixed PCC, median (q25–q75), 16 patients",  # cycle 28
                        "CR fixed / CR own, median (min–max): patients; whole images",
                        "r_fix = CR(VVC) / CR(codec), median (min–max): patients; whole images",
                        "r_fix > 1, of 16: patients; whole images"]
    for row, c in zip(table[1:5], CODECS):
        crs = of("patients", None, c, "cr_fix")
        assert len(crs) == 16
        want = [LABELS[c], str(FIXED_PCC[c]),
                f"{percentile(crs, 50):.2f} ({percentile(crs, 25):.2f}–{percentile(crs, 75):.2f})",
                f"{m3(of('patients', None, c, 'loss'))}; {m3(of('whole', None, c, 'loss'))}"]
        if c == "vtm":
            want += ["-", "-"]
        else:
            rp, rw = (of("ratios", level, f"vtm/{c}", "r_fix") for level in ("пациенты", "снимки"))
            want += [f"{m3(rp)}; {m3(rw)}", f"{sum(r > 1 for r in rp)}; {sum(r > 1 for r in rw)}"]
        assert row == want, c
    assert table[5:] == [[f"Note. PSNR-HVS-M ≥ 42 dB for all {len(built['fragments'])} fragments and 16 whole images at "
                          "the fixed PCC of every codec; descriptive analysis, declared after the main results."]]
    found = {}
    for row in read_csv(built["out"] / "tables" / "numbers_index.csv")[1:]:
        if row[0].startswith("Table 4 / "):
            found.setdefault(table_place(row[0].removeprefix("Table 4 / "), table)[:2], []).append(row[1])
    for r, line_ in enumerate(table):
        for c, text in enumerate(line_):
            rest = text
            for value in sorted(found.get((r, c), []), key=len, reverse=True):
                assert value in rest, (r, c, value)
                rest = rest.replace(value, "", 1)
            assert not re.search(r"\d", rest), (r, c, rest)
    below = json.loads((built["cache"] / "cycles/26/e5_fixed.json").read_text(encoding="utf-8"))
    below["analysis"]["below"][5]["below"] = 1
    (tmp_path / "below.json").write_text(json.dumps(below, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(fg.RunRefusal, match="Table 4: a unit lies below T at the fixed PCC"):
        fg.table4(fg.Source("cycles/26/e5_fixed.json", tmp_path / "below.json"), style_of(fg))


def test_no_n_before_a_number_in_captions_heads_and_notes(built):
    """Cycle 28 (owner, 2026-11-05): a count is the number with its noun - no "n = " before a number and no head
    "n <noun>" in any caption, head or note; the numbers keep their index rows (value, key, operation), the rows of
    Table 3 carry the new head in their ids, the caption of Fig. 6 has one row more - its whole images."""
    captions = (built["out"] / "figures" / "captions.txt").read_text(encoding="utf-8").strip().split("\n\n")
    tables = [read_csv(built["out"] / "tables" / f"table{n}.csv") for n in (1, 2, 3, 4)]
    notes = [t[-1][0] for t in tables if len(t[-1]) == 1]
    assert len(captions) == 12 and len(notes) == 2  # the notes of Table 2 and Table 4
    for text in captions + [h for t in tables for h in t[0]] + notes:
        assert not re.search(r"\bn = \d", text) and not re.match(r"n [a-z]", text), text
    assert (tables[2][0][1], tables[2][0][3]) == ("Number of patients", "Number of fragments")
    n = str(len(built["fragments"]))
    index = {r[0]: (r[1], r[4], r[5]) for r in read_csv(built["out"] / "tables" / "numbers_index.csv")[1:]}
    want = {"Caption Fig. 6 / n": ("16", "analysis.comparisons.n", "same"),
            "Caption Fig. 6 / n whole images": ("16", "analysis.whole_comparisons.n", "same"),
            "Caption Fig. 7 / n patients": ("16", "analysis.complexity.patients", "count"),
            "Caption Table 3 / n patients": ("16", "analysis.patients[codec=vtm,t_db=42.0]", "count"),
            "Caption Table 3 / n fragments": (n, "analysis.units[level=f,codec=vtm,t_db=42.0,judged=True]", "count"),
            "Caption Table 4 / n patients": ("16", "analysis.summary[level=пациенты].n", "same"),
            "Table 4 / head / 3 / n patients": ("16", "analysis.summary[level=пациенты].n", "same"),
            "Table 2 / note / n patients": ("16", "analysis.comparisons.n", "same"),
            "Table 2 / note / n whole images": ("16", "analysis.whole_comparisons.n", "same"),
            "Table 2 / row 1 / Conclusion": ("VVC is better than JPEG", "analysis.comparisons[8].verdict", "verdict"),
            "Table 2 / row 2 / Conclusion": ("VVC is better than HEIF", "analysis.comparisons[9].verdict", "verdict"),
            "Table 2 / row 3 / Conclusion": ("not established", "analysis.comparisons[10].verdict", "verdict"),
            "Table 3 / row 5 / Number of patients": ("16", "analysis.jpeg50.patients", "count"),
            "Table 3 / row 5 / Number of fragments": (n, "analysis.jpeg50.units[level=f]", "count")}
    for k, c in enumerate(CODECS, 1):
        want[f"Table 3 / row {k} / Number of patients"] = ("16", f"analysis.patients[codec={c},t_db=42.0]", "count")
        want[f"Table 3 / row {k} / Number of fragments"] = (
            n, f"analysis.units[level=f,codec={c},t_db=42.0,judged=True]", "count")
    assert {i: index.get(i) for i in want} == want
    assert not [i for i in index if i.startswith("Table 3 / row ") and i.endswith((" / n patients", " / n fragments"))]
    assert sorted(i.removeprefix("Caption Fig. 6 / ") for i in index if i.startswith("Caption Fig. 6 / ")) == \
        sorted(("CI level", "T 1", "T 2", "T 3", "n", "n whole images", "reference"))
