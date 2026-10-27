"""Lab tests of figures.py (paper P001, cycle 18) on synthetic run files of E4, E5 and E6 and a synthetic fragment
with a coder of the test's own - no pixel of the data, no real number.

- the files are created: three figures (PDF, PNG, CSV), three tables (.xlsx, .csv), the index; an existing file is
  refused without --force;
- the CSV of every figure equals the data of the run files chosen by the declared rule (Fig. 1 - the 77th fragment by
  ascending MGM, ties to the smaller code; Fig. 2 - the patients in ascending MGM); the series of Fig. 1 carry the
  data of their own codec;
- every series of the plots is a line with markers: the marker size equals the style and is at most 4 pt, at most
  markevery_max markers on a line; every series, threshold line and r = 1 has a legend entry; the labels of the
  logarithmic axis carry no "x 10^n";
- Fig. 3: the upper row - the original and the reconstructions; the lower row - the region of every panel repeated
  4 times by pixels (the test's own index arithmetic), the region - the block of the 4 x 4 grid with the largest MGM by
  the test's own Sobel sums, ties to the smaller row and column; a reconstruction that differs from E4 is refused;
- every row of numbers_index.csv: its value equals the key of its run file under its operation and format, by the
  test's own evaluator of keys; every table row of the index is the cell or a part of it; Table 1 has no sha256;
- a decimal point only and no Cyrillic in the tables and in the texts of the figures;
- colour and line style differ for every pair of series; the style table has four colours, lines and markers.
"""

import csv
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.lab

STUDY = Path(__file__).resolve().parents[1]
CODECS = ("vtm", "jpeg", "heif", "avif")
LABELS = {"vtm": "VVC (VTM)", "jpeg": "JPEG", "heif": "HEIF", "avif": "AVIF"}
PAIR_LABELS = {"vtm/jpeg": "VVC / JPEG", "vtm/heif": "VVC / HEIF", "vtm/avif": "VVC / AVIF"}
CYRILLIC = re.compile("[Ѐ-ӿ]")
CHOSEN = "D08_r00_c04"  # the largest MGM of the synthetic fragments (ties with D16_r00_c04 - the smaller code)
THRESHOLD_PCC = {"vtm": 31, "jpeg": 67, "heif": 67, "avif": 67}  # the 12th point of every codec below


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


def fake_coder(codec, image, pcc):
    step = CODECS.index(codec) + 2
    return f"{codec}:{pcc}".encode() + image[:2, :8].tobytes(), (image // step) * step


def synthetic(cache):
    """E4: the header and the points of every fragment; E5 and E6: the fields the figures and tables read."""
    fragments = [{"unit": f"D{1 + i // 5:02d}_r00_c{i % 5:02d}", "image": f"D{1 + i // 5:02d}",
                  "mgm": round(0.02 + 0.001 * (i % 40), 6)} for i in range(80)]  # ties of MGM: two fragments each
    original = original_fragment()
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
                             "psnr_hvs_m_db": 50.0 - 0.7 * j - k / 5, "file_sha256": file_sha, "recon_sha256": recon_sha})
            rows.append({"unit": f["unit"], "codec": c, "param": 999, "status": "lossless", "cr_file": 1.5,
                         "psnr_hvs_m_db": "+inf", "file_sha256": "0" * 64, "recon_sha256": "0" * 64})
            units.append({"unit": f["unit"], "image": f["image"], "level": "f", "codec": c, "t_db": 42.0, "judged": True,
                          "reason": None, "pcc": params[11], "cr": 2.0 + 11 + k / 10,
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
                         for lv, n in (("фрагменты", 80), ("снимки", 16)) for t in (40.0, 41.0, 42.0)],
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
    for name, data in (("cycles/13/e4_main.json", e4), ("cycles/17/e5_analysis.json", e5), ("cycles/16/e6_pages.json", e6)):
        (cache / name).parent.mkdir(parents=True, exist_ok=True)
        (cache / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return e4, e5, fragments, original


def coders(fg, monkeypatch, original, wrong=None):
    """The test's own loader and coder in place of those of E4; `wrong` - a codec whose reconstruction is altered."""
    def load_fragment(paths, unit):
        assert unit == CHOSEN, unit
        return original.copy()

    def encode(paths, codec, image, pcc, work):
        assert np.array_equal(image, original) and pcc == THRESHOLD_PCC[codec]
        file, recon = fake_coder(codec, image, pcc)
        if codec == wrong:
            recon = recon.copy()
            recon[0, 0] ^= 1
        return file, recon
    monkeypatch.setattr(fg, "load_fragment", load_fragment)
    monkeypatch.setattr(fg, "encode", encode)


@pytest.fixture()
def built(fg, tmp_path, monkeypatch):
    cache, papers = tmp_path / "cache", tmp_path / "papers"
    e4, e5, fragments, original = synthetic(cache)
    coders(fg, monkeypatch, original)
    pin = hashlib.sha256((cache / "cycles/13/e4_main.json").read_bytes()).hexdigest()
    out, index = fg.build(cache, papers, paths={"cache_root": cache}, e4_pin=pin, log=lambda *a: None)
    return {"fg": fg, "cache": cache, "papers": papers, "out": out, "e4": e4, "e5": e5, "fragments": fragments,
            "pin": pin, "original": original, "monkeypatch": monkeypatch}


def sources(built):
    fg, cache = built["fg"], built["cache"]
    return [fg.Source(name, cache / name) for name in ("cycles/13/e4_main.json", "cycles/17/e5_analysis.json",
                                                       "cycles/16/e6_pages.json")]


def style_of(fg):
    return fg.tomllib.loads(fg.STYLE_FILE.read_text(encoding="utf-8"))


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def test_files_created_and_existing_files_need_force(fg, built):
    for name in fg.OUTPUTS:
        assert (built["out"] / name).is_file() and (built["out"] / name).stat().st_size > 0, name
    with pytest.raises(fg.RunRefusal, match="--force overwrites them"):
        fg.build(built["cache"], built["papers"], paths={"cache_root": built["cache"]}, e4_pin=built["pin"],
                 log=lambda *a: None)
    fg.build(built["cache"], built["papers"], paths={"cache_root": built["cache"]}, e4_pin=built["pin"], force=True,
             log=lambda *a: None)
    assert not list(built["cache"].glob("p001_figures_*"))  # the work folder of the coders is removed


def test_figure_csv_equal_the_run_files(built):
    chosen = sorted(built["fragments"], key=lambda f: (f["mgm"], f["unit"]))[76]["unit"]
    fig1 = read_csv(built["out"] / "figures" / "fig1.csv")
    want = sorted(((r["codec"], r["param"], r["cr_file"], r["psnr_hvs_m_db"]) for r in built["e4"]["rows"]
                   if r["unit"] == chosen and r["status"] == "valid"))
    head = fig1[0]
    got = sorted((r[head.index("codec")], int(r[head.index("pcc")]), float(r[head.index("cr")]),
                  float(r[head.index("psnr_hvs_m_db")])) for r in fig1[1:])
    assert {r[head.index("fragment")] for r in fig1[1:]} == {chosen} and got == want
    a = built["e5"]["analysis"]
    mgm = {p["image"]: p["mgm"] for p in a["complexity"]["patients"]}
    fig2 = read_csv(built["out"] / "figures" / "fig2.csv")
    assert fig2[0] == ["panel", "series", "patient", "mgm", "value"]
    want = sorted([("a", p["codec"], p["image"], mgm[p["image"]], p["median_cr"]) for p in a["patients"]] +
                  [("b", x["pair"], x["image"], mgm[x["image"]], x["r"]) for x in a["ratios"]])
    assert sorted((r[0], r[1], r[2], float(r[3]), float(r[4])) for r in fig2[1:]) == want
    fig3 = read_csv(built["out"] / "figures" / "fig3.csv")
    head = fig3[0]
    region = [r for r in fig3[1:] if r[head.index("kind")] == "region"]
    assert len(region) == 1 and (region[0][head.index("row")], region[0][head.index("column")]) == ("2", "1")
    panels = [r for r in fig3[1:] if r[head.index("kind")] == "panel"]
    assert [(r[head.index("codec")], int(r[head.index("pcc")])) for r in panels] == list(THRESHOLD_PCC.items())
    assert len(fig3) == 1 + 16 + 1 + 4  # blocks, region, panels - no pixel


def test_fig1_series_carry_their_own_codec(built):
    fg = built["fg"]
    style = style_of(fg)
    e4, e5, _ = sources(built)
    plt, _ = fg.setup(style)
    fig, data, _ = fg.fig1(e4, e5, style, plt)
    chosen = data[0]["fragment"]
    lines = {line.get_label(): line for line in fig.axes[0].get_lines() if line.get_label() in LABELS.values()}
    for c in CODECS:
        want = sorted(r["cr_file"] for r in built["e4"]["rows"] if r["unit"] == chosen and r["codec"] == c
                      and r["status"] == "valid")
        assert sorted(lines[LABELS[c]].get_xdata()) == want, c
    pairs = {(line.get_color(), line.get_linestyle()) for line in lines.values()}
    assert len(pairs) == 4  # colour and line style differ for every pair of series
    texts = [t.get_text() for t in fig.findobj(match=lambda o: hasattr(o, "get_text"))]
    assert texts and not any(CYRILLIC.search(t) for t in texts)
    plt.close(fig)


def plotted(built):
    """Figures 1 and 2 drawn: [(axes, the labels of its data series, the labels of its reference lines)]."""
    fg = built["fg"]
    style = style_of(fg)
    e4, e5, _ = sources(built)
    plt, _ = fg.setup(style)
    f1, _, _ = fg.fig1(e4, e5, style, plt)
    f2, _, _ = fg.fig2(e5, style, plt)
    top, bottom = f2.axes
    return style, plt, [f1, f2], [(f1.axes[0], set(LABELS.values()), {"T = 40 dB", "T = 41 dB", "T = 42 dB"}),
                                  (top, set(LABELS.values()), set()), (bottom, set(PAIR_LABELS.values()), {"r = 1"})]


def test_every_series_is_a_line_with_small_markers_and_in_the_legend(built):
    style, plt, figs, panels = plotted(built)
    size, most = style["page"]["marker_size_pt"], style["page"]["markevery_max"]
    for ax, series, references in panels:
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
        legend = {t.get_text() for t in ax.get_legend().get_texts()}
        assert legend == series | references, legend
    for fig in figs:
        plt.close(fig)


def test_log_axis_labels_are_plain_numbers(built):
    style, plt, figs, panels = plotted(built)
    ax = panels[0][0]
    assert ax.get_xscale() == "log"
    figs[0].canvas.draw()
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


def test_fig3_panels_are_the_reconstructions_and_their_region_by_repetition(built, tmp_path):
    fg = built["fg"]
    style = style_of(fg)
    e4, e5, e6 = sources(built)
    unit, original, panels = fg.comparison(e4, e5, e6, {"cache_root": built["cache"]}, tmp_path, lambda *a: None)
    assert unit == CHOSEN and np.array_equal(original, built["original"])
    plt, _ = fg.setup(style)
    fig, data, _ = fg.fig3(e5, e6, unit, original, panels, style, plt)
    r, c = expected_region(built["original"])
    assert (r, c) == (2, 1)
    images = [built["original"]] + [fake_coder(k, built["original"], THRESHOLD_PCC[k])[1] for k in CODECS]
    axes = np.array(fig.axes).reshape(2, 5)
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
    assert titles[0] == "Original" and all(f"PCC {THRESHOLD_PCC[k]}" in t for k, t in zip(CODECS, titles[1:]))
    plt.close(fig)


def test_fig3_region_ties_go_to_the_smaller_row_then_column(fg):
    rng = np.random.default_rng(5)
    noise = rng.integers(0, 256, (128, 128)).astype(np.uint8)
    flat = np.full((512, 512), 77, np.uint8)
    assert fg.region_of(flat)[:2] == (0, 0)
    for blocks, want in ((((3, 0), (1, 3)), (1, 3)), (((2, 3), (2, 0)), (2, 0))):
        image = flat.copy()
        for r, c in blocks:
            image[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128] = noise
        assert fg.region_of(image)[:2] == want == expected_region(image)


def test_fig3_refuses_a_reconstruction_that_differs_from_e4(built, tmp_path):
    fg = built["fg"]
    e4, e5, e6 = sources(built)
    coders(fg, built["monkeypatch"], built["original"], wrong="heif")
    with pytest.raises(fg.RunRefusal, match="heif PCC 67: encoded again, the reconstruction differs from E4 by sha256"):
        fg.comparison(e4, e5, e6, {"cache_root": built["cache"]}, tmp_path, lambda *a: None)


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
    assert checked > 150
    ids = [r[0] for r in index[1:]]
    assert len(ids) == len(set(ids))
    for prefix in ("Fig. 1 / ", "Fig. 2 (a) / ", "Fig. 2 (b) / ", "Fig. 3 / ", "Table 1 / ", "Table 2 / note / ", "Table 3 / "):
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
    assert table3[5][0] == "JPEG PCC 50 (level 50 by Fidler et al.)" and table3[5][-1] == "77 of 80; 13 of 16"
