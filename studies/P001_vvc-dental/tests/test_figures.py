"""Lab tests of figures.py (paper P001, cycle 17) on synthetic run files of E4 and E5 - no pixel, no real number.

- the files are created: three figures (PDF, PNG, CSV), four tables (.xlsx, .csv), the index; an existing file is
  refused without --force;
- the CSV of every figure equals the data of the run files chosen by the declared rule (Fig. 1 - the 77th fragment by
  ascending MGM, ties to the smaller code); the plotted series of Fig. 1 carry the data of their own codec;
- every row of numbers_index.csv: its value equals the key of its run file under its operation and format, by the
  test's own evaluator of keys; its sha256 is that of the file;
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

import pytest

pytestmark = pytest.mark.lab

STUDY = Path(__file__).resolve().parents[1]
CODECS = ("vtm", "jpeg", "heif", "avif")
LABELS = {"vtm": "VVC (VTM)", "jpeg": "JPEG", "heif": "HEIF", "avif": "AVIF"}
CYRILLIC = re.compile("[Ѐ-ӿ]")


@pytest.fixture(scope="module")
def fg():
    spec = importlib.util.spec_from_file_location("figures", STUDY / "figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic(cache):
    """E4: the header and the points of every fragment; E5: the fields the figures and tables read."""
    fragments = [{"unit": f"D{1 + i // 5:02d}_r00_c{i % 5:02d}", "image": f"D{1 + i // 5:02d}",
                  "mgm": round(0.02 + 0.001 * (i % 40), 6)} for i in range(80)]  # ties of MGM: two fragments each
    rows, units = [], []
    for f in fragments:
        for k, c in enumerate(CODECS):
            params = range(20, 41) if c == "vtm" else range(100, 39, -3)
            for j, p in enumerate(params):
                rows.append({"unit": f["unit"], "codec": c, "param": p, "status": "valid", "cr_file": 2.0 + j + k / 10,
                             "psnr_hvs_m_db": 50.0 - 0.7 * j - k / 5})
            rows.append({"unit": f["unit"], "codec": c, "param": 999, "status": "lossless", "cr_file": 1.5,
                         "psnr_hvs_m_db": "+inf"})
            units.append({"unit": f["unit"], "image": f["image"], "level": "f", "codec": c, "t_db": 42.0, "judged": True,
                          "cr": 2.0 + 11 + k / 10, "psnr_hvs_m_db": 50.0 - 0.7 * 11 - k / 5, "psnr_db": 38.5 + k + f["mgm"],
                          "haarpsi": 0.91 + k / 100 - f["mgm"], "mgm": f["mgm"]})
    images = [f"D{i:02d}" for i in range(1, 17)]
    patients = [{"image": im, "codec": c, "t_db": 42.0, "median_cr": 10.0 + i + k} for i, im in enumerate(images)
                for k, c in enumerate(CODECS)]
    ratios = [{"image": im, "pair": f"vtm/{b}", "t_db": 42.0, "r": 1.1 + i / 100 + k / 10, "in_test": True}
              for i, im in enumerate(images) for k, b in enumerate(("jpeg", "heif", "avif"))]

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
              "shares": [{"level": lv, "t_db": t, "n": n, "count": n - 1, "share_pct": 100.0 * (n - 1) / n}
                         for lv, n in (("фрагменты", 154), ("снимки", 16)) for t in (40.0, 41.0, 42.0)],
              "patients": [{"image": im, "n": 10, "median_cr": 7.5 + i / 3, "median_cr_opt": 7.8 + i / 3}
                           for i, im in enumerate(images)]}
    e5 = {"thresholds_db": [40.0, 41.0, 42.0],
          "analysis": {"complexity": {"fragments": fragments}, "units": units, "patients": patients, "ratios": ratios,
                       "comparisons": comparisons("пациенты"), "whole_comparisons": comparisons("снимки"),
                       "jpeg50": jpeg50}}
    e4 = {"header": {"codec_versions": {"vtm_gpac_sha256": {"encoder": "a" * 64, "decoder": "b" * 64, "mp4box": "c" * 64},
                                        "jpeg": {"pillow": "12.3.0", "libjpeg_turbo": "3.1.4.1"},
                                        "heif": {"libheif": "1.23.6", "x265": "4.3"},
                                        "avif_libheif": {"libheif": "1.19.7", "aom": "3.12.0"}},
                     "plan": {"vtm": {"f": list(range(-12, 64))}, "jpeg": list(range(1, 101)),
                              "heif": list(range(0, 101)), "avif": list(range(0, 101))}},
          "rows": rows}
    for name, data in (("cycles/13/e4_main.json", e4), ("cycles/17/e5_analysis.json", e5)):
        (cache / name).parent.mkdir(parents=True, exist_ok=True)
        (cache / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return e4, e5, fragments


@pytest.fixture()
def built(fg, tmp_path):
    cache, papers = tmp_path / "cache", tmp_path / "papers"
    e4, e5, fragments = synthetic(cache)
    pin = hashlib.sha256((cache / "cycles/13/e4_main.json").read_bytes()).hexdigest()
    out, index = fg.build(cache, papers, e4_pin=pin, log=lambda *a: None)
    return {"fg": fg, "cache": cache, "papers": papers, "out": out, "e4": e4, "e5": e5, "fragments": fragments,
            "pin": pin}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def test_files_created_and_existing_files_need_force(fg, built):
    for name in fg.OUTPUTS:
        assert (built["out"] / name).is_file() and (built["out"] / name).stat().st_size > 0, name
    with pytest.raises(fg.RunRefusal, match="--force overwrites them"):
        fg.build(built["cache"], built["papers"], e4_pin=built["pin"], log=lambda *a: None)
    fg.build(built["cache"], built["papers"], e4_pin=built["pin"], force=True, log=lambda *a: None)


def test_figure_csv_equal_the_run_files(built):
    chosen = sorted(built["fragments"], key=lambda f: (f["mgm"], f["unit"]))[76]["unit"]
    fig1 = read_csv(built["out"] / "figures" / "fig1.csv")
    want = sorted(((r["codec"], r["param"], r["cr_file"], r["psnr_hvs_m_db"]) for r in built["e4"]["rows"]
                   if r["unit"] == chosen and r["status"] == "valid"))
    head = fig1[0]
    got = sorted((r[head.index("codec")], int(r[head.index("pcc")]), float(r[head.index("cr")]),
                  float(r[head.index("psnr_hvs_m_db")])) for r in fig1[1:])
    assert {r[head.index("fragment")] for r in fig1[1:]} == {chosen} and got == want
    marked = [r for r in fig1[1:] if r[head.index("threshold_42")] == "1"]
    assert len(marked) == 4 and {r[head.index("codec")] for r in marked} == set(CODECS)
    fig3 = read_csv(built["out"] / "figures" / "fig3.csv")
    assert sorted((r[0], r[1], float(r[2]), float(r[3])) for r in fig3[1:]) == sorted(
        (u["unit"], u["codec"], u["mgm"], u["cr"]) for u in built["e5"]["analysis"]["units"])
    fig2 = read_csv(built["out"] / "figures" / "fig2.csv")
    points = [r for r in fig2[1:] if r[1] != "estimate"]
    assert sorted((r[0], r[1], float(r[2])) for r in points) == sorted(
        (x["pair"], x["image"], x["r"]) for x in built["e5"]["analysis"]["ratios"])


def test_fig1_series_carry_their_own_codec(built):
    fg = built["fg"]
    style = fg.tomllib.loads(fg.STYLE_FILE.read_text(encoding="utf-8"))
    e4 = fg.Source("cycles/13/e4_main.json", built["cache"] / "cycles/13/e4_main.json")
    e5 = fg.Source("cycles/17/e5_analysis.json", built["cache"] / "cycles/17/e5_analysis.json")
    plt = fg.setup(style)
    fig, data, _ = fg.fig1(e4, e5, style, plt)
    chosen = data[0]["fragment"]
    lines = {line.get_label(): line for line in fig.axes[0].get_lines() if not line.get_label().startswith("_")}
    for c in CODECS:
        want = sorted(r["cr_file"] for r in built["e4"]["rows"] if r["unit"] == chosen and r["codec"] == c
                      and r["status"] == "valid")
        assert sorted(lines[LABELS[c]].get_xdata()) == want, c
    pairs = {(line.get_color(), line.get_linestyle()) for line in lines.values()}
    assert len(pairs) == 4  # colour and line style differ for every pair of series
    texts = [t.get_text() for t in fig.findobj(match=lambda o: hasattr(o, "get_text"))]
    assert texts and not any(CYRILLIC.search(t) for t in texts)
    plt.close(fig)


def test_style_table_four_colours_lines_markers(built):
    fg = built["fg"]
    codecs = fg.tomllib.loads(fg.STYLE_FILE.read_text(encoding="utf-8"))["codecs"]
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


def apply(op, node):
    numbers = lambda xs: [math.inf if x == "+inf" else x for x in xs]  # noqa: E731
    return {"value": lambda: node, "count": lambda: len(node), "median": lambda: percentile(numbers(node), 50),
            "q25": lambda: percentile(numbers(node), 25), "q75": lambda: percentile(numbers(node), 75),
            "min": lambda: min(numbers(node)), "max": lambda: max(numbers(node)),
            "join": lambda: "; ".join(f"{k} {v}" for k, v in node.items()),
            "join8": lambda: "; ".join(f"{k} {v[:8]}" for k, v in node.items()),
            "range": lambda: f"{min(node)} to {max(node)}",
            "list": lambda: "; ".join(f"{x:.0f}" for x in node),
            "en": lambda: {"пациенты": "patients", "снимки": "whole images", "фрагменты": "fragments"}[node],
            "verdict": lambda: ("descriptive" if node is None else "VVC better than " + node[len("VVC выгоднее "):]
                                if node.startswith("VVC выгоднее ") else "not established"),
            "name": lambda: "level 50 by Fidler et al." if node.startswith("уровень 50 по Fidler") else "JPEG PCC 50",
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
    assert len(ids) == len(set(ids)) and any(i.startswith("Fig. 2 / ") for i in ids) and any(i.startswith("Table 4") for i in ids)


def test_decimal_point_and_no_cyrillic_in_tables(built):
    for n in (1, 2, 3, 4):
        for line in read_csv(built["out"] / "tables" / f"table{n}.csv"):
            for value in line:
                assert not re.search(r"\d,\d", value) and not CYRILLIC.search(value), (n, value)
        from openpyxl import load_workbook
        sheet = load_workbook(built["out"] / "tables" / f"table{n}.xlsx").active
        for row in sheet.iter_rows(values_only=True):
            assert not any(isinstance(v, str) and (CYRILLIC.search(v) or re.search(r"\d,\d", v)) for v in row)
    table2 = read_csv(built["out"] / "tables" / "table2.csv")
    assert table2[7][1:] == ["42", "VVC/JPEG", "16", "1.242", "1.100", "1.300", "3.05e-05", "9.16e-05",
                             "VVC better than JPEG"]  # rows 1-3 at 40 dB, 4-6 at 41, 7-9 at 42 (patients)
    assert table2[9][-1] == "not established" and table2[1][-1] == "descriptive" and table2[1][8] == "-"
