"""Lab tests of figures.py (paper P001, cycle 19) on synthetic run files of E4, E5, E6 and the grid, a synthetic image
D10 and fragment, and a coder of the test's own - no pixel of the data, no real number.

- the files are created: four figures (PDF, PNG, CSV), the captions, three tables (.xlsx, .csv), the index; an existing
  file is refused without --force;
- the choice of fragments by the declared rules, ties included: Fig. 1 - the two fragments of D10 with MGM closest to
  the median MGM of all fragments; Fig. 2 - the smallest and the largest MGM; the frames of the grid of Fig. 1 - by the
  formula of B11;
- the CSV of every figure equals the data of the run files; the series of Fig. 2 carry the data of their own codec;
- every series of the plots is a line with markers: the marker size equals the style, is 2.5 pt and at most 4 pt, at
  most markevery_max markers on a line; every series, threshold line and r = 1 has a legend entry; the labels of the
  logarithmic axes carry no "x 10^n";
- Fig. 4: the upper row - the original and the reconstructions; the lower row - the region repeated 4 times by pixels
  (the test's own index arithmetic), the region - the block with the largest MGM by the test's own Sobel sums, ties to
  the smaller row and column; a reconstruction that differs from E4 is refused; the titles of the panels do not overlap;
- in the PNG every image panel is a whole multiple of its pixels, pixel for pixel;
- no "VVC (VTM)" in any output file or figure text;
- every row of numbers_index.csv equals the key of its run file under its operation and format, by the test's own
  evaluator of keys; every table row of the index is the cell or a part of it; every digit of the captions belongs to
  an index row of its caption; Table 1 has no sha256;
- a decimal point only and no Cyrillic in the tables and in the texts of the figures;
- colour and line style differ for every pair of series; the style table has four colours, lines and markers.
"""

import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.lab

STUDY = Path(__file__).resolve().parents[1]
CODECS = ("vtm", "jpeg", "heif", "avif")
LABELS = {"vtm": "VVC", "jpeg": "JPEG", "heif": "HEIF", "avif": "AVIF"}
PAIR_LABELS = {"vtm/jpeg": "VVC / JPEG", "vtm/heif": "VVC / HEIF", "vtm/avif": "VVC / AVIF"}
THRESHOLD_LABELS = {"T = 40 dB", "T = 41 dB", "T = 42 dB"}
CYRILLIC = re.compile("[Ѐ-ӿ]")
CHOSEN = "D08_r00_c04"  # the largest MGM of the synthetic fragments (ties with D16_r00_c04 - the smaller code)
SIMPLEST = "D01_r00_c00"  # the smallest MGM (ties with D09_r00_c00 - the smaller code)
THRESHOLD_PCC = {"vtm": 31, "jpeg": 67, "heif": 67, "avif": 67}  # the 12th point of every codec below
D10_SHAPE = (1100, 2100)  # H, W of the synthetic D10: a grid of 2 x 4 fragments
D10_MGM = (0.030, 0.035, 0.0405, 0.0437, 0.046, 0.050, 0.052, 0.055)  # its fragments r00_c00 ... r01_c03; the median
# of all fragments (0.041) is none of them, so a caption that prints the MGM of a fragment for the median is seen


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
            params = list(range(20, 41)) if c == "vtm" else list(range(100, 39, -3))
            for j, p in enumerate(params):
                file_sha = recon_sha = "0" * 64
                if f["unit"] == CHOSEN and p == THRESHOLD_PCC[c]:
                    file, recon = fake_coder(c, original, p)
                    file_sha, recon_sha = hashlib.sha256(file).hexdigest(), hashlib.sha256(recon.tobytes()).hexdigest()
                rows.append({"unit": f["unit"], "codec": c, "param": p, "status": "valid", "cr_file": 2.0 + j + k / 10,
                             "psnr_hvs_m_db": 50.0 - 0.7 * j - k / 5, "psnr_db": 45.0 - 0.6 * j - k / 4 - f["mgm"],
                             "file_sha256": file_sha, "recon_sha256": recon_sha})
            rows.append({"unit": f["unit"], "codec": c, "param": 999, "status": "lossless", "cr_file": 1.5,
                         "psnr_hvs_m_db": "+inf", "psnr_db": "+inf", "file_sha256": "0" * 64, "recon_sha256": "0" * 64})
            units.append({"unit": f["unit"], "image": f["image"], "level": "f", "codec": c, "t_db": 42.0, "judged": True,
                          "reason": None, "pcc": params[11] - (f["mgm"] > 0.05 and f["unit"] != CHOSEN), "cr": 2.0 + 11 + k / 10,
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
                out.append({"level": level, "pair": f"vtm/{b}", "t_db": t, "n": 16, "ratio": 1.2 + k / 10 + t / 1000,
                            "ratio_low": 1.1 + k / 10, "ratio_high": 1.3 + k / 10, "p": 3.0517578125e-05,
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
    e5 = {"thresholds_db": [40.0, 41.0, 42.0],
          "analysis": {"complexity": {"fragments": fragments, "patients": complexity_patients}, "units": units,
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
                       ("cycles/16/e6_pages.json", e6), ("cycles/9/p001_tiles.json", tiles)):
        (cache / name).parent.mkdir(parents=True, exist_ok=True)
        (cache / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    arrays = {CHOSEN: original, "D10": whole}
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
    pin = hashlib.sha256((cache / "cycles/13/e4_main.json").read_bytes()).hexdigest()
    out, index = fg.build(cache, papers, paths={"cache_root": cache}, e4_pin=pin, log=lambda *a: None)
    return {"fg": fg, "cache": cache, "papers": papers, "out": out, "e4": e4, "e5": e5, "fragments": fragments,
            "pin": pin, "arrays": arrays, "monkeypatch": monkeypatch}


def sources(built):
    fg, cache = built["fg"], built["cache"]
    return [fg.Source(name, cache / name) for name in ("cycles/13/e4_main.json", "cycles/17/e5_analysis.json",
                                                       "cycles/16/e6_pages.json", "cycles/9/p001_tiles.json")]


def style_of(fg):
    return fg.tomllib.loads(fg.STYLE_FILE.read_text(encoding="utf-8"))


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def typical_by_rule(fragments):
    median = statistics.median(f["mgm"] for f in fragments)
    ranked = sorted((abs(f["mgm"] - median), f["unit"]) for f in fragments if f["image"] == "D10")
    return median, [u for _, u in ranked[:2]]


def test_files_created_and_existing_files_need_force(fg, built):
    for name in fg.OUTPUTS:
        assert (built["out"] / name).is_file() and (built["out"] / name).stat().st_size > 0, name
    with pytest.raises(fg.RunRefusal, match="--force overwrites them"):
        fg.build(built["cache"], built["papers"], paths={"cache_root": built["cache"]}, e4_pin=built["pin"],
                 log=lambda *a: None)
    fg.build(built["cache"], built["papers"], paths={"cache_root": built["cache"]}, e4_pin=built["pin"], force=True,
             log=lambda *a: None)
    assert not list(built["cache"].glob("p001_figures_*"))  # the work folder of the coders is removed


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


def test_fig1_initial_data_by_the_rules(built):
    fg = built["fg"]
    style = style_of(fg)
    e4, e5, e6, tiles = sources(built)
    median, want = typical_by_rule(built["fragments"])
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
    for ax, unit in ((b, want[0]), (c, want[1])):
        (shown,) = ax.get_images()
        x, y = grid[unit]
        assert np.array_equal(np.asarray(shown.get_array())[..., 0], built["arrays"]["D10"][y:y + 512, x:x + 512]), unit
    (shown,) = a.get_images()
    assert np.array_equal(np.asarray(shown.get_array())[..., 1], built["arrays"]["D10"])
    rows = [r for r in data if r.get("kind") == "tile"]
    assert {r["unit"]: r["panel"] for r in rows if r["panel"]} == {want[0]: "b", want[1]: "c"}
    assert [r["mgm"] for r in data if r.get("kind") == "median MGM"] == [median]
    assert want[0] in caption["value"] and want[1] in caption["value"] and f"{median:.4f}" in caption["value"]
    plt.close(fig)


def test_fig1_is_one_row_of_one_height_in_the_pdf_and_whole_multiples_in_the_png(built):
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
    fig2 = read_csv(built["out"] / "figures" / "fig2.csv")
    head = fig2[0]
    for unit, panels in ((SIMPLEST, "ac"), (CHOSEN, "bd")):
        for panel, field in zip(panels, ("psnr_hvs_m_db", "psnr_db")):
            want = sorted((r["codec"], r["param"], r["cr_file"], r[field]) for r in built["e4"]["rows"]
                          if r["unit"] == unit and r["status"] == "valid")
            got = sorted((r[head.index("codec")], int(r[head.index("pcc")]), float(r[head.index("cr")]),
                          float(r[head.index(field)])) for r in fig2[1:] if r[head.index("panel")] == panel)
            assert got == want and {r[head.index("fragment")] for r in fig2[1:] if r[head.index("panel")] == panel} == {unit}
    a = built["e5"]["analysis"]
    mgm = {p["image"]: p["mgm"] for p in a["complexity"]["patients"]}
    fig3 = read_csv(built["out"] / "figures" / "fig3.csv")
    assert fig3[0] == ["panel", "series", "patient", "mgm", "value"]
    want = sorted([("a", p["codec"], p["image"], mgm[p["image"]], p["median_cr"]) for p in a["patients"]] +
                  [("b", x["pair"], x["image"], mgm[x["image"]], x["r"]) for x in a["ratios"]])
    assert sorted((r[0], r[1], r[2], float(r[3]), float(r[4])) for r in fig3[1:]) == want
    fig4 = read_csv(built["out"] / "figures" / "fig4.csv")
    head = fig4[0]
    region = [r for r in fig4[1:] if r[head.index("kind")] == "region"]
    assert len(region) == 1 and (region[0][head.index("row")], region[0][head.index("column")]) == ("2", "1")
    panels = [r for r in fig4[1:] if r[head.index("kind")] == "panel"]
    assert [(r[head.index("codec")], int(r[head.index("pcc")])) for r in panels] == list(THRESHOLD_PCC.items())
    assert len(fig4) == 1 + 16 + 1 + 4 + 10  # blocks, region, panels, the boxes of the PNG - no pixel
    fig1 = read_csv(built["out"] / "figures" / "fig1.csv")
    head = fig1[0]
    assert len([r for r in fig1[1:] if r[head.index("kind")] == "tile"]) == 8 and len(fig1) == 1 + 8 + 1 + 3 + 3


def plotted(built):
    """Figures 2 and 3 drawn: [(axes, the labels of its data series, the labels of its reference lines, legend texts)]."""
    fg = built["fg"]
    style = style_of(fg)
    e4, e5, _, _ = sources(built)
    plt, _ = fg.setup(style)
    f2 = fg.fig2(e4, e5, style, plt)[0]
    f3 = fg.fig3(e5, style, plt)[0]
    shared = {t.get_text() for t in f2.legends[0].get_texts()}
    top, bottom = f3.axes
    panels = [(ax, set(LABELS.values()), THRESHOLD_LABELS if k < 2 else set(), shared) for k, ax in enumerate(f2.axes)]
    panels += [(top, set(LABELS.values()), set(), {t.get_text() for t in top.get_legend().get_texts()}),
               (bottom, set(PAIR_LABELS.values()), {"r = 1"}, {t.get_text() for t in bottom.get_legend().get_texts()})]
    return style, plt, [f2, f3], panels


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
    assert {t for _, _, _, legend in panels[:4] for t in legend} == set(LABELS.values()) | THRESHOLD_LABELS
    for fig in figs:
        plt.close(fig)


def test_log_axis_labels_are_plain_numbers(built):
    style, plt, figs, panels = plotted(built)
    figs[0].canvas.draw()
    for ax, *_ in panels[:4]:
        assert ax.get_xscale() == "log"
        labels = [t.get_text() for t in ax.get_xticklabels(which="both") if t.get_text()]
        assert labels and all(re.fullmatch(r"\d+(\.\d+)?", t) for t in labels), labels
        assert not any(s in t for t in labels for s in ("×", "times", "^", "$")), labels
    for fig in figs:
        plt.close(fig)


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


def drawn_fig4(built, tmp_path):
    fg = built["fg"]
    style = style_of(fg)
    e4, e5, e6, _ = sources(built)
    unit, original, recons = fg.comparison(e4, e5, e6, built["arrays"], {"cache_root": built["cache"]}, tmp_path,
                                           lambda *a: None)
    plt, _ = fg.setup(style)
    return plt, unit, original, fg.fig4(e5, e6, unit, original, recons, style, plt)


def test_fig4_panels_are_the_reconstructions_and_their_region_by_repetition(built, tmp_path):
    plt, unit, original, (fig, data, series, dpi, panels, caption) = drawn_fig4(built, tmp_path)
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


def test_fig4_titles_do_not_overlap(built, tmp_path):
    plt, _, _, (fig, *_) = drawn_fig4(built, tmp_path)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = sorted((ax.title.get_window_extent(renderer) for ax in fig.axes if ax.get_title()), key=lambda b: b.x0)
    assert len(boxes) == 5
    for left, right in zip(boxes, boxes[1:]):
        assert left.x1 < right.x0, (left, right)
    width, height = fig.canvas.get_width_height()
    assert all(b.x0 >= 0 and b.x1 <= width and b.y1 <= height for b in boxes)
    plt.close(fig)


def test_fig4_region_ties_go_to_the_smaller_row_then_column(fg):
    rng = np.random.default_rng(5)
    noise = rng.integers(0, 256, (128, 128)).astype(np.uint8)
    flat = np.full((512, 512), 77, np.uint8)
    assert fg.region_of(flat)[:2] == (0, 0)
    for blocks, want in ((((3, 0), (1, 3)), (1, 3)), (((2, 3), (2, 0)), (2, 0))):
        image = flat.copy()
        for r, c in blocks:
            image[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128] = noise
        assert fg.region_of(image)[:2] == want == expected_region(image)


def test_fig4_refuses_a_reconstruction_that_differs_from_e4(built, tmp_path):
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
    _, want = typical_by_rule(built["fragments"])
    crop = {u: whole[grid[u][1]:grid[u][1] + 512, grid[u][0]:grid[u][0] + 512] for u in want}
    expected = {("fig1", "a"): (whole, True), ("fig1", "b"): (crop[want[0]], False), ("fig1", "c"): (crop[want[1]], False),
                ("fig4", "upper original"): (original, True), ("fig4", "lower original"): (region(original), False)}
    for k in CODECS:
        expected[("fig4", f"upper {k}")] = (recon[k], True)
        expected[("fig4", f"lower {k}")] = (region(recon[k]), False)
    checked = 0
    for name in ("fig1", "fig4"):
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
    assert checked == 13


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
            fg.fig2(e4, e5, style, plt)[0], fg.fig3(e5, style, plt)[0],
            drawn_fig4(built, tmp_path)[3][0]]
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
            "verdict": lambda: ("descriptive" if node is None else "VVC better than " + node[len("VVC выгоднее "):]
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
    checked = 0
    for row in index[1:]:
        item = dict(zip(head, row))
        if item["key"] == "constant":
            assert item["file"] == built["fg"].SCRIPT and item["op"] == "text"
            continue
        path = built["cache"] / item["file"]
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), item["id"]
        data = json.loads(path.read_text(encoding="utf-8"))
        assert printed(apply(item["op"], resolve(data, item["key"])), item["format"]) == item["value"], item["id"]
        checked += 1
    assert checked > 200
    ids = [r[0] for r in index[1:]]
    assert len(ids) == len(set(ids))
    for prefix in ("Fig. 1 / ", "Fig. 2 (a) / ", "Fig. 3 (b) / ", "Fig. 4 / ", "Table 1 / ", "Table 2 / note / ",
                   "Table 3 / ", "Caption Fig. 1 / ", "Caption Table 3 / "):
        assert any(i.startswith(prefix) for i in ids), prefix


def test_every_table_row_of_the_index_is_its_cell(built):
    tables = {n: read_csv(built["out"] / "tables" / f"table{n}.csv") for n in (1, 2, 3)}
    rows = 0
    for row in read_csv(built["out"] / "tables" / "numbers_index.csv")[1:]:
        parts = row[0].split(" / ")
        if not parts[0].startswith("Table "):
            continue
        table = tables[int(parts[0].removeprefix("Table "))]
        if parts[1] == "note":
            assert row[1] in table[-1][0] and len(table[-1]) == 1, row[0]
        else:
            text = table[int(parts[1].removeprefix("row "))][table[0].index(parts[2])]
            assert (text == row[1]) if len(parts) == 3 else (row[1] in text), (row[0], row[1], text)
        rows += 1
    assert rows > 100


def test_every_digit_of_the_captions_is_an_index_row(built):
    captions = (built["out"] / "figures" / "captions.txt").read_text(encoding="utf-8").strip().split("\n\n")
    labels = [re.match(r"(Fig\. \d|Table \d)\. ", c).group(1) for c in captions]
    assert labels == ["Fig. 1", "Fig. 2", "Fig. 3", "Fig. 4", "Table 1", "Table 2", "Table 3"]
    index = read_csv(built["out"] / "tables" / "numbers_index.csv")[1:]
    assert captions[1].startswith("Fig. 2. Dependences of PSNR-HVS-M (a,b) and PSNR (c,d) on compression ratio (CR)")
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

    for n in (1, 2, 3):
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
                         "1.240 (1.100–1.300)", "1.241 (1.100–1.300)", "1.242 (1.100–1.300)", "9.16e-05",
                         "VVC better than JPEG"]
    assert table2[3][-1] == "not established" and table2[4][-2:] == ["-", "-"]
    assert "n = 16 patients" in table2[5][0] and "p = 3.05e-05" in table2[5][0] and "p (Holm) = 9.16e-05" in table2[5][0]
    table3 = read_csv(built["out"] / "tables" / "table3.csv")
    n = len(built["fragments"])
    assert table3[0][4] == f"Threshold PCC for T = 42 dB, median (min–max), {n} fragments"
    pcc = sorted(u["pcc"] for u in built["e5"]["analysis"]["units"] if u["codec"] == "vtm")
    assert table3[1][0] == "VVC" and table3[1][4] == f"{statistics.median(pcc):g} ({pcc[0]}–{pcc[-1]})"
    assert table3[5][0] == "JPEG PCC 50 (level 50 by Fidler et al.)" and table3[5][4] == "50"
    assert table3[5][-1] == f"{n - 3} of {n}; 13 of 16"
