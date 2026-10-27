"""Figures and tables of the paper P001 from the run files of E4, E5 and E6 - one command. Run: python studies/P001_vvc-dental/figures.py   (--force: overwrite files that exist)

Reads <cache_root>/cycles/13/e4_main.json (pinned sha256), cycles/17/e5_analysis.json and cycles/16/e6_pages.json,
and the style figures_style.toml beside this script; writes into <papers_root>/P001_vvc-dental/: figures/fig1..fig3
(.pdf, .png at the dpi of the style, .csv of the plotted data - no pixel) and tables/table1..table3 (.xlsx, .csv) and
tables/numbers_index.csv - every table cell (every number of a composed cell) and every figure series: the value as
printed, the run file (relative to cache_root), its sha256, the key in the file, the operation and the format (the
source of NUMBERS.md). The composition is the one declared in the task of cycle 18; all text is English with a
decimal point; PCC names the parameter of compression; lines of different colour and style carry small markers
(STUDY.md section 7). Fig. 3 - the original fragment and the four codecs at the threshold of 42 dB - is drawn from
reconstructions encoded again with the wrappers of E4 (the function of run_e6.py), each equal to its E4 row by sha256
of the file and of the reconstruction, otherwise a refusal in words; it is a medical image and stays outside Git.
A number is never typed by hand: it is read from a run file.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np

from imstudies.core.complexity import mgm
from imstudies.lab.runs import RunRefusal, sha256_file

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
STYLE_FILE = STUDY / "figures_style.toml"
PAPER = "P001_vvc-dental"
E4_FILE, E5_FILE, E6_FILE = "cycles/13/e4_main.json", "cycles/17/e5_analysis.json", "cycles/16/e6_pages.json"
TILES_FILE = "cycles/9/p001_tiles.json"
E4_SHA256 = "44a1733512fdfbb9d02d210ff81d492b9a5d929ec8d547d84676d47350875f2a"  # A12
SCRIPT = "core/studies/P001_vvc-dental/figures.py"
CODECS = ("vtm", "jpeg", "heif", "avif")
OTHERS = ("jpeg", "heif", "avif")
T_MAIN = 42.0
FIG1_RANK = 77  # the fragment of Fig. 1: the 77th of 154 by ascending MGM, ties - the smaller code
FRAGMENT = 512  # Fig. 3: the whole fragment in the upper row
BLOCKS = 4      # Fig. 3: the region is a block of the BLOCKS x BLOCKS grid of the fragment (128 x 128)
ZOOM = 4        # Fig. 3: the lower row - the region repeated ZOOM times along each axis, no interpolation
PCC_MEANING = {"vtm": "QP", "jpeg": "quality", "heif": "quality", "avif": "quality"}  # STUDY.md section 5
VTM_IMPLEMENTATION = "VTM 23.0 all-intra + GPAC MP4Box 26.08-DEV"  # STUDY.md section 5: E4 holds only the sha256
VERSION_FIELDS = {  # Table 1: the version numbers taken from header.codec_versions of E4 - (group, ((label, field), ...))
    "jpeg": ("jpeg", (("Pillow", "pillow"), ("libjpeg-turbo", "libjpeg_turbo"))),
    "heif": ("heif", (("pillow-heif", "pillow_heif"), ("libheif", "libheif"), ("x265", "x265"))),
    "avif": ("avif_libheif", (("pillow-heif", "pillow_heif"), ("libheif", "libheif"), ("libaom", "aom_encoder"))),
}
VERSION = re.compile(r"\d+(?:\.\d+)+(?:\+\d+)?")  # "1.23.6"; "4.3+1" of "x265 HEVC encoder (4.3+1-e9b8812)"
CONTAINER = {"vtm": "HEIF (vvc1)", "jpeg": "JFIF", "heif": "HEIF (hvc1)", "avif": "AVIF"}
OUTPUTS = [f"figures/fig{k}.{ext}" for k in (1, 2, 3) for ext in ("pdf", "png", "csv")] + \
          [f"tables/table{k}.{ext}" for k in (1, 2, 3) for ext in ("xlsx", "csv")] + ["tables/numbers_index.csv"]
INDEX_COLUMNS = ("id", "value", "file", "sha256", "key", "op", "format")


# ---------------------------------------------------------------- values as printed, with their source

def show(value, fmt):
    """The printed form: a decimal point always; inf as 'inf'; None as '-'."""
    if value is None:
        return "-"
    if fmt == "text":
        return str(value)
    if isinstance(value, str) and value in ("+inf", "-inf", "nan"):
        value = float(value.lstrip("+"))
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return format(value, fmt)


class Source:
    def __init__(self, name, path):
        self.name, self.path = name, Path(path)
        self.sha256 = sha256_file(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8"))


def cell(value, fmt, source, key, op="value"):
    """A printed cell and where it comes from (a row of the index once an id is given)."""
    return {"value": show(value, fmt), "file": source.name if source else SCRIPT,
            "sha256": source.sha256 if source else "", "key": key, "op": op, "format": fmt}


def constant(text):
    return cell(text, "text", None, "constant", "text")


def joined(*pieces, also=()):
    """A cell printed from pieces - text and (name, cell); each cell is a row of the index under the cell's id and its
    name. `also` - (name, cell) not printed again: another key whose value equals a printed piece."""
    parts = [p for p in pieces if not isinstance(p, str)]
    printed = {c["value"] for _, c in parts}
    for name, c in also:
        if c["value"] not in printed:
            raise RunRefusal(f"{name}: {c['value']} is not among the printed values {sorted(printed)}")
    return {"value": "".join(p if isinstance(p, str) else p[1]["value"] for p in pieces), "parts": parts + list(also)}


def index_rows(name, c):
    if "parts" in c:
        return [{"id": f"{name} / {part}", **x} for part, x in c["parts"]]
    return [{"id": name, **c}]


def pick(items, **where):
    """The items whose fields equal the given values compared as text (the same rule as the key filters)."""
    return [x for x in items if all(str(x.get(k)) == str(v) for k, v in where.items())]


def where_text(**where):
    return ",".join(f"{k}={v}" for k, v in where.items())


def position(items, **where):
    """The place of the one item with the given fields; none or several - a refusal."""
    found = [k for k, x in enumerate(items) if all(str(x.get(f)) == str(v) for f, v in where.items())]
    if len(found) != 1:
        raise RunRefusal(f"{len(found)} entries with {where_text(**where)}: exactly one is expected")
    return found[0]


def same(values, what):
    """The one value all entries share - the note of Table 2 prints it once; different values - a refusal."""
    if len(set(values)) != 1:
        raise RunRefusal(f"{what}: the values differ ({sorted(set(values))[:3]}...) - one value is printed")
    return values[0]


def version(text):
    found = VERSION.search(text)
    if not found:
        raise RunRefusal(f"no version number in {text!r}")
    return found.group(0)


def percentile(values, q):
    return float(np.percentile(np.array(values, dtype=np.float64), q))  # numpy, linear interpolation


def number(x):
    return math.inf if x == "+inf" else x


def verdict_en(text):
    if text is None:
        return "descriptive"
    return f"VVC better than {text.removeprefix('VVC выгоднее ')}" if text.startswith("VVC выгоднее ") else "not established"


def name_en(text):
    return "level 50 by Fidler et al." if text.startswith("уровень 50 по Fidler") else "JPEG PCC 50"


# ---------------------------------------------------------------- tables

def table1(e4, style):
    """The codecs, short: implementation with version numbers, PCC and its range, container - no sha256."""
    head = ("Codec", "Implementation and versions", "PCC", "PCC range (fragments)", "Container")
    versions, plan = e4.data["header"]["codec_versions"], e4.data["header"]["plan"]
    rows = []
    for c in CODECS:
        if c == "vtm":
            implementation = constant(VTM_IMPLEMENTATION)
        else:
            group, fields = VERSION_FIELDS[c]
            pieces = []
            for k, (label, field) in enumerate(fields):
                pieces += [(" / " if k else "") + f"{label} ",
                           (label, cell(version(versions[group][field]), "text", e4,
                                        f"header.codec_versions.{group}.{field}", "version"))]
            implementation = joined(*pieces)
        values = plan["vtm"]["f"] if c == "vtm" else plan[c]
        path = "header.plan.vtm.f" if c == "vtm" else f"header.plan.{c}"
        rows.append([constant(style["codecs"][c]["label"]), implementation, constant(PCC_MEANING[c]),
                     cell(f"{min(values)} to {max(values)}", "text", e4, path, "range"), constant(CONTAINER[c])])
    return head, rows, []


def effect(source, listname, i, c):
    base = f"analysis.{listname}[{i}]"
    return joined(("effect", cell(c["ratio"], ".3f", source, f"{base}.ratio")), " (",
                  ("CI low", cell(c["ratio_low"], ".3f", source, f"{base}.ratio_low")), "–",
                  ("CI high", cell(c["ratio_high"], ".3f", source, f"{base}.ratio_high")), ")")


def table2(e5, style):
    """Pairs by rows; patients and whole images at every threshold by columns, "effect (95% CI)"; p (Holm) and the
    conclusion of the rule of STUDY.md section 6 for the three main rows; a note under the table."""
    a, thresholds = e5.data["analysis"], e5.data["thresholds_db"]
    columns = [(listname, t, f"{label}, {t:.0f} dB") for listname, label in
               (("comparisons", "Patients"), ("whole_comparisons", "Whole images")) for t in thresholds]
    head = ("Pair", *(name for _, _, name in columns), f"p (Holm), {T_MAIN:.0f} dB, patients", "Conclusion")
    rows = []
    for b in (*OTHERS, "jpeg_opt"):
        pair = f"vtm/{b}"
        line = [constant("VVC / JPEG (optimised Huffman)" if b == "jpeg_opt" else f"VVC / {style['codecs'][b]['label']}")]
        for listname, t, _ in columns:
            i = position(a[listname], pair=pair, t_db=t)
            line.append(effect(e5, listname, i, a[listname][i]))
        i = position(a["comparisons"], pair=pair, t_db=T_MAIN)
        c, base = a["comparisons"][i], f"analysis.comparisons[{i}]"
        if c["role"] == "основное":
            line += [cell(c["p_holm"], ".3g", e5, f"{base}.p_holm"),
                     cell(verdict_en(c["verdict"]), "text", e5, f"{base}.verdict", "verdict")]
        else:
            line += [constant("-"), constant("-")]
        rows.append(line)
    whole, main = a["whole_comparisons"], pick(a["comparisons"], role="основное")
    note = joined(
        "Note. Effect - the Hodges-Lehmann estimate of r = CR(VVC) / CR(codec) with its 95% CI; n = ",
        ("n patients", cell(same([c["n"] for c in a["comparisons"]], "n of patients"), "d", e5, "analysis.comparisons.n",
                            "same")),
        " patients and n = ",
        ("n whole images", cell(same([c["n"] for c in whole], "n of whole images"), "d", e5,
                                "analysis.whole_comparisons.n", "same")),
        " whole images in every comparison; exact two-sided Wilcoxon signed-rank test, p = ",
        ("p", cell(same([c["p"] for c in a["comparisons"]], "p of patients"), ".3g", e5, "analysis.comparisons.p", "same")),
        " in every comparison; p (Holm) = ",
        ("p (Holm)", cell(same([c["p_holm"] for c in main], "p (Holm)"), ".3g", e5,
                          "analysis.comparisons[role=основное].p_holm", "same")),
        f" for the three main comparisons ({T_MAIN:.0f} dB, patients).",
        also=[("p, whole images", cell(same([c["p"] for c in whole], "p of whole images"), ".3g", e5,
                                       "analysis.whole_comparisons.p", "same"))])
    return head, rows, [note]


def spread(source, key, values, centre, low, high, fmt):
    """"median (low-high)" of the values under the key: low and high - quartiles or min and max."""
    stat = {"median": lambda: percentile(values, 50), "q25": lambda: percentile(values, 25),
            "q75": lambda: percentile(values, 75), "min": lambda: min(values), "max": lambda: max(values)}
    return joined((centre, cell(stat[centre](), fmt, source, key, centre)), " (",
                  (low, cell(stat[low](), fmt, source, key, low)), "–", (high, cell(stat[high](), fmt, source, key, high)),
                  ")")


def table3(e5, style):
    """At the threshold of 42 dB by codec: CR over the patients, PSNR and HaarPSI over the fragments; and the point
    JPEG PCC 50 with the share of units at or above 42 dB."""
    head = ("Codec or point", "n patients", "CR, median (q25–q75)", "n fragments", "PSNR, dB, median (min–max)",
            "HaarPSI, median (min–max)", f"PSNR-HVS-M ≥ {T_MAIN:.0f} dB: fragments; whole images")
    a = e5.data["analysis"]
    rows = []
    for c in CODECS:
        pw = dict(codec=c, t_db=T_MAIN)
        uw = dict(level="f", codec=c, t_db=T_MAIN, judged=True)
        crs = [p["median_cr"] for p in pick(a["patients"], **pw)]
        units = pick(a["units"], **uw)
        pk, uk = f"analysis.patients[{where_text(**pw)}]", f"analysis.units[{where_text(**uw)}]"
        rows.append([constant(style["codecs"][c]["label"]), cell(len(crs), "d", e5, pk, "count"),
                     spread(e5, f"{pk}.median_cr", crs, "median", "q25", "q75", ".2f"), cell(len(units), "d", e5, uk, "count"),
                     spread(e5, f"{uk}.psnr_db", [number(u["psnr_db"]) for u in units], "median", "min", "max", ".2f"),
                     spread(e5, f"{uk}.haarpsi", [u["haarpsi"] for u in units], "median", "min", "max", ".4f"),
                     constant("-")])
    j = a["jpeg50"]
    name = name_en(j["name"])
    label = cell(name, "text", e5, "analysis.jpeg50.name", "name")
    units = pick(j["units"], level="f")
    shares = [position(j["shares"], level=lv, t_db=T_MAIN) for lv in ("фрагменты", "снимки")]
    reached = []
    for k, (what, i) in enumerate(zip(("fragments", "whole images"), shares)):
        s, base = j["shares"][i], f"analysis.jpeg50.shares[{i}]"
        reached += ["; "] if k else []
        reached += [(f"{what} at or above", cell(s["count"], "d", e5, f"{base}.count")), " of ",
                    (f"{what}", cell(s["n"], "d", e5, f"{base}.n"))]
    rows.append([joined("JPEG PCC 50 (", ("name", label), ")") if name != "JPEG PCC 50" else label,
                 cell(len(j["patients"]), "d", e5, "analysis.jpeg50.patients", "count"),
                 spread(e5, "analysis.jpeg50.patients.median_cr", [p["median_cr"] for p in j["patients"]], "median", "q25",
                        "q75", ".2f"),
                 cell(len(units), "d", e5, "analysis.jpeg50.units[level=f]", "count"),
                 spread(e5, "analysis.jpeg50.units[level=f].psnr_db", [number(u["psnr_db"]) for u in units], "median", "min",
                        "max", ".2f"),
                 spread(e5, "analysis.jpeg50.units[level=f].haarpsi", [u["haarpsi"] for u in units], "median", "min", "max",
                        ".4f"),
                 joined(*reached)])
    return head, rows, []


# ---------------------------------------------------------------- figures

def setup(style):
    """matplotlib with the font of the style, or its fallback when the font is not installed; (pyplot, the font)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    p = style["page"]
    font = p["font"]
    try:
        font_manager.findfont(font_manager.FontProperties(family=font), fallback_to_default=False)
    except ValueError:
        font = p["font_fallback"]
    plt.rcParams.update({"font.family": font, "font.size": p["font_size_pt"], "legend.fontsize": p["font_size_pt"],
                         "lines.linewidth": p["line_width_pt"], "lines.markersize": p["marker_size_pt"],
                         "axes.unicode_minus": False, "pdf.fonttype": 42, "mathtext.default": "regular"})
    return plt, font


def size(style, fig):
    p, f = style["page"], style["figures"]
    widths = {"column": p["column_cm"], "double": p["double_column_cm"]}
    if f[f"{fig}_width"] not in widths:
        raise RunRefusal(f"figures_style.toml: {fig}_width is {f[f'{fig}_width']!r}, 'column' or 'double' is expected")
    return widths[f[f"{fig}_width"]] / 2.54, f[f"{fig}_height_cm"] / 2.54


def line(ax, x, y, s, style, label):
    """A line of the series' colour and style with small markers on it - at most markevery_max, evenly."""
    p = style["page"]
    step = max(1, math.ceil(len(x) / p["markevery_max"]))
    ax.plot(x, y, color=s["color"], linestyle=s["line"], linewidth=p["line_width_pt"], marker=s["marker"],
            markersize=p["marker_size_pt"], markevery=step, label=label)


def plain_log(axis):
    """A logarithmic axis printed with plain numbers 1, 2, 5, 10, 20, 50... - no 9 x 10^0."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def fig1(e4, e5, style, plt):
    """PSNR-HVS-M = f(CR) of one fragment: four lines with small markers, the threshold lines in grey."""
    fragments = sorted(e5.data["analysis"]["complexity"]["fragments"], key=lambda f: (f["mgm"], f["unit"]))
    unit = fragments[FIG1_RANK - 1]["unit"]
    fig, ax = plt.subplots(figsize=size(style, "fig1"), constrained_layout=True)
    data, series = [], []
    for c in CODECS:
        s = style["codecs"][c]
        rows = sorted((r for r in e4.data["rows"] if r["unit"] == unit and r["codec"] == c and r["status"] == "valid"),
                      key=lambda r: r["cr_file"])
        line(ax, [r["cr_file"] for r in rows], [r["psnr_hvs_m_db"] for r in rows], s, style, s["label"])
        data += [{"fragment": unit, "codec": c, "pcc": r["param"], "cr": r["cr_file"], "psnr_hvs_m_db": r["psnr_hvs_m_db"]}
                 for r in rows]
        series.append((f"Fig. 1 / {s['label']}", cell(len(rows), "d", e4, f"rows[{where_text(unit=unit, codec=c, status='valid')}]",
                                                       "count")))
    t = style["thresholds"]
    for level in e5.data["thresholds_db"]:
        ax.axhline(level, color=t["color"], linestyle=t[f"line_{level:.0f}"], linewidth=style["page"]["line_width_pt"],
                   label=f"T = {level:.0f} dB")
    series.append(("Fig. 1 / threshold lines, dB", cell("; ".join(show(x, ".0f") for x in e5.data["thresholds_db"]), "text",
                                                       e5, "thresholds_db", "list")))
    ax.set_xscale("log")
    plain_log(ax.xaxis)
    ax.set_xlabel("CR")
    ax.set_ylabel("PSNR-HVS-M, dB")
    ax.legend(loc=style["figures"]["fig1_legend"], ncols=style["figures"]["fig1_legend_columns"])
    return fig, data, series


def fig2(e5, style, plt):
    """Two panels over the MGM of the 16 patients at 42 dB, points in ascending MGM joined by lines: (a) the median CR
    of every codec; (b) r = CR(VVC) / CR(codec) of the three pairs and the line r = 1."""
    a = e5.data["analysis"]
    mgms = {p["image"]: p["mgm"] for p in a["complexity"]["patients"]}
    by_mgm = lambda p: (mgms[p["image"]], p["image"])  # noqa: E731
    fig, (top, bottom) = plt.subplots(2, 1, figsize=size(style, "fig2"), sharex=True, constrained_layout=True)
    data, series = [], [("Fig. 2 / x: MGM of the patients", cell(len(mgms), "d", e5, "analysis.complexity.patients",
                                                                  "count"))]
    for c in CODECS:
        s = style["codecs"][c]
        points = sorted(pick(a["patients"], codec=c, t_db=T_MAIN), key=by_mgm)
        line(top, [mgms[p["image"]] for p in points], [p["median_cr"] for p in points], s, style, s["label"])
        data += [{"panel": "a", "series": c, "patient": p["image"], "mgm": mgms[p["image"]], "value": p["median_cr"]}
                 for p in points]
        series.append((f"Fig. 2 (a) / {s['label']}", cell(len(points), "d", e5,
                                                           f"analysis.patients[{where_text(codec=c, t_db=T_MAIN)}]", "count")))
    for b in OTHERS:
        s, pair = style["codecs"][b], f"vtm/{b}"
        points = sorted(pick(a["ratios"], pair=pair, t_db=T_MAIN, in_test=True), key=by_mgm)
        line(bottom, [mgms[p["image"]] for p in points], [p["r"] for p in points], s, style, f"VVC / {s['label']}")
        data += [{"panel": "b", "series": pair, "patient": p["image"], "mgm": mgms[p["image"]], "value": p["r"]}
                 for p in points]
        series.append((f"Fig. 2 (b) / VVC / {s['label']}",
                       cell(len(points), "d", e5, f"analysis.ratios[{where_text(pair=pair, t_db=T_MAIN, in_test=True)}]",
                            "count")))
    t = style["thresholds"]
    bottom.axhline(1.0, color=t["color"], linestyle=t["unity_line"], linewidth=style["page"]["line_width_pt"], label="r = 1")
    series.append(("Fig. 2 (b) / reference line", constant("r = 1")))
    top.set_title("(a)", loc="left")
    bottom.set_title("(b)", loc="left")
    top.set_ylabel(f"CR at {T_MAIN:.0f} dB, median")
    bottom.set_ylabel("r = CR(VVC) / CR(codec)")
    bottom.set_xlabel("MGM of the patient (median over fragments)")
    f = style["figures"]
    for ax in (top, bottom):
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + (high - low) * f["fig2_legend_headroom"])
        ax.legend(loc=f["fig2_legend"], ncols=f["fig2_legend_columns"])
    return fig, data, series


def study_module(name):
    spec = importlib.util.spec_from_file_location(name, STUDY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fragment(paths, unit):
    """The original fragment from the copy data/P001 and the grid of cycle 9 by the loaders of E4 (B10, B11)."""
    run_e4 = study_module("run_e4")
    registry, tile_rows, tile = run_e4.load_inputs(paths["data_root"], Path(paths["cache_root"]) / TILES_FILE)
    return np.ascontiguousarray(run_e4.load_units(paths["data_root"], registry, tile_rows, tile)[unit]["array"])


def encode(paths, codec, image, pcc, work):
    """One point with the coder and the arguments of E4 - the function of run_e6.py; (stored file, reconstruction)."""
    return study_module("run_e6").encode(paths, codec, image, pcc, work)


def region_of(original):
    """The block of the BLOCKS x BLOCKS grid with the largest MGM of the original (B13); ties - the smaller row, then
    the smaller column; (row, column, its MGM, the MGM of every block by row and column)."""
    b = original.shape[0] // BLOCKS
    scores = {(r, c): mgm(np.ascontiguousarray(original[r * b:(r + 1) * b, c * b:(c + 1) * b]))
              for r in range(BLOCKS) for c in range(BLOCKS)}
    r, c = min(scores, key=lambda k: (-scores[k], k))
    return r, c, scores[(r, c)], scores


def comparison(e4, e5, e6, paths, work, log):
    """The fragment of the first page of E6 (the largest MGM of E5) and its four reconstructions at the threshold PCC of
    42 dB, every one encoded again and equal to its E4 row by sha256 of the file and of the reconstruction."""
    unit = e6.data["pages"][0]["unit"]
    largest = min(e5.data["analysis"]["complexity"]["fragments"], key=lambda f: (-f["mgm"], f["unit"]))["unit"]
    if unit != largest:
        raise RunRefusal(f"the first page of E6 is {unit}, the fragment with the largest MGM of E5 is {largest}")
    original = load_fragment(paths, unit)
    if original.dtype != np.uint8 or original.shape != (FRAGMENT, FRAGMENT):
        raise RunRefusal(f"{unit}: {original.dtype} {original.shape}, uint8 {FRAGMENT} x {FRAGMENT} is expected")
    a = e5.data["analysis"]
    rows = {(r["unit"], r["codec"], r["param"]): r for r in e4.data["rows"]}
    panels = []
    for c in CODECS:
        i = position(a["units"], unit=unit, codec=c, t_db=T_MAIN)
        u = a["units"][i]
        if not u["judged"]:
            raise RunRefusal(f"{unit}, {c}: not judged at {T_MAIN:.0f} dB ({u['reason']}) - no point to show")
        row = rows[(unit, c, u["pcc"])]
        file, recon = encode(paths, c, original, u["pcc"], work)
        recon = np.ascontiguousarray(recon)
        file_equal = hashlib.sha256(file).hexdigest() == row["file_sha256"]
        recon_equal = hashlib.sha256(recon.tobytes()).hexdigest() == row["recon_sha256"]
        if not (file_equal and recon_equal):
            raise RunRefusal(f"{unit}, {c} PCC {u['pcc']}: encoded again, the {'file' if not file_equal else 'reconstruction'}"
                             " differs from E4 by sha256")
        if recon.dtype != np.uint8 or recon.shape != original.shape:
            raise RunRefusal(f"{unit}, {c}: the reconstruction is {recon.dtype} {recon.shape}, not that of the original")
        panels.append({"codec": c, "i": i, "unit": u, "row": row, "recon": recon})
        log(f"  Fig. 3: {unit}, {c} PCC {u['pcc']} encoded again - file and reconstruction equal to E4 by sha256")
    return unit, original, panels


def fig3(e5, e6, unit, original, panels, style, plt):
    """Upper row - the original and the four codecs at the threshold of 42 dB, the whole fragment; lower row - the same
    region of every panel repeated ZOOM times (pixel repetition); the frame of the region on the upper row. 8-bit
    levels as they are: grey RGB of the very values, no colour map, no interpolation, no stretch of levels."""
    from matplotlib.patches import Rectangle

    r, c, value, scores = region_of(original)
    b = original.shape[0] // BLOCKS
    f = style["figures"]
    fig, axes = plt.subplots(2, 1 + len(panels), figsize=size(style, "fig3"), constrained_layout=True)
    series = [("Fig. 3 / fragment", cell(unit, "text", e6, "pages[0].unit"))]
    data = [{"kind": "block", "fragment": unit, "row": k[0], "column": k[1], "mgm": v} for k, v in sorted(scores.items())]
    data.append({"kind": "region", "fragment": unit, "row": r, "column": c, "mgm": value, "block": b, "zoom": ZOOM})
    titles = ["Original"]
    for p in panels:
        s, u, base = style["codecs"][p["codec"]], p["unit"], f"analysis.units[{p['i']}]"
        pcc, psnr, cr = (cell(u["pcc"], "d", e5, f"{base}.pcc"), cell(u["psnr_hvs_m_db"], ".2f", e5, f"{base}.psnr_hvs_m_db"),
                         cell(u["cr"], ".2f", e5, f"{base}.cr"))
        titles.append(f"{s['label']}, PCC {pcc['value']}\nPSNR-HVS-M {psnr['value']} dB\nCR {cr['value']}")
        series += [(f"Fig. 3 / {s['label']} / PCC", pcc), (f"Fig. 3 / {s['label']} / PSNR-HVS-M", psnr),
                   (f"Fig. 3 / {s['label']} / CR", cr)]
        data.append({"kind": "panel", "fragment": unit, "codec": p["codec"], "pcc": u["pcc"],
                     "psnr_hvs_m_db": u["psnr_hvs_m_db"], "cr": u["cr"], "e4_file_sha256": p["row"]["file_sha256"],
                     "e4_recon_sha256": p["row"]["recon_sha256"], "encoded_again_equal": 1})
    for k, image in enumerate([original] + [p["recon"] for p in panels]):
        part = image[r * b:(r + 1) * b, c * b:(c + 1) * b]
        zoomed = np.repeat(np.repeat(part, ZOOM, axis=0), ZOOM, axis=1)
        for ax, shown in ((axes[0, k], image), (axes[1, k], zoomed)):
            ax.imshow(np.stack([shown] * 3, axis=-1), interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, k].add_patch(Rectangle((c * b - 0.5, r * b - 0.5), b, b, fill=False, edgecolor=f["fig3_frame_color"],
                                       linewidth=f["fig3_frame_width_pt"]))
        axes[0, k].set_title(titles[k])
    axes[0, 0].set_ylabel(f"{FRAGMENT} x {FRAGMENT}")
    axes[1, 0].set_ylabel(f"{b} x {b}, zoom {ZOOM}")
    return fig, data, series


# ---------------------------------------------------------------- writing

def write_csv(path, rows, columns=None):
    with open(path, "w", encoding="utf-8", newline="") as f:
        out = csv.writer(f)
        if columns:
            out.writerow(columns)
        for r in rows:
            out.writerow(r)


def write_table(folder, n, head, rows, notes, index):
    from openpyxl import Workbook

    printed = [list(head)] + [[c["value"] for c in r] for r in rows] + [[c["value"]] for c in notes]
    write_csv(folder / f"table{n}.csv", printed)
    wb = Workbook()
    ws = wb.active
    ws.title = f"Table {n}"
    for r, line_ in enumerate(printed, start=1):
        for c, value in enumerate(line_, start=1):
            ws.cell(row=r, column=c, value=value)
    wb.save(folder / f"table{n}.xlsx")
    for r, line_ in enumerate(rows, start=1):
        for k, c in enumerate(line_):
            index += index_rows(f"Table {n} / row {r} / {head[k]}", c)
    for c in notes:
        index += index_rows(f"Table {n} / note", c)


def build(cache_root, papers_root, paths=None, style_file=STYLE_FILE, force=False, e4_pin=E4_SHA256, log=print):
    """`paths` - the lab paths of run_e4.load_paths (data_root and the coders) for Fig. 3."""
    cache_root, out = Path(cache_root), Path(papers_root) / PAPER
    targets = [out / p for p in OUTPUTS]
    existing = [str(p.relative_to(out)) for p in targets if p.exists()]
    if existing and not force:
        raise RunRefusal(f"{len(existing)} files exist ({existing[:3]}...): --force overwrites them")
    for name in (E4_FILE, E5_FILE, E6_FILE):
        if not (cache_root / name).is_file():
            raise RunRefusal(f"the run file {cache_root / name} does not exist")
    if sha256_file(cache_root / E4_FILE) != e4_pin:
        raise RunRefusal(f"{E4_FILE}: sha256 differs from the pinned {e4_pin}")
    style = tomllib.loads(Path(style_file).read_text(encoding="utf-8"))
    e4, e5, e6 = (Source(name, cache_root / name) for name in (E4_FILE, E5_FILE, E6_FILE))
    work = Path(tempfile.mkdtemp(prefix="p001_figures_", dir=cache_root))
    try:
        unit, original, panels = comparison(e4, e5, e6, paths or {}, work, log)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    plt, font = setup(style)
    index = []
    for n, (fig, data, series) in enumerate((fig1(e4, e5, style, plt), fig2(e5, style, plt),
                                             fig3(e5, e6, unit, original, panels, style, plt)), 1):
        fig.savefig(out / "figures" / f"fig{n}.pdf")
        fig.savefig(out / "figures" / f"fig{n}.png", dpi=style["page"]["dpi"])
        plt.close(fig)
        columns = list(dict.fromkeys(k for row in data for k in row))
        write_csv(out / "figures" / f"fig{n}.csv", [[row.get(k, "") for k in columns] for row in data], columns)
        index += [row for name, c in series for row in index_rows(name, c)]
    for n, (head, rows, notes) in enumerate((table1(e4, style), table2(e5, style), table3(e5, style)), 1):
        write_table(out / "tables", n, head, rows, notes, index)
    write_csv(out / "tables" / "numbers_index.csv", [[i[k] for k in INDEX_COLUMNS] for i in index], INDEX_COLUMNS)
    r, c, value, _ = region_of(original)
    log(f"Figures and tables written into {out}: {len(OUTPUTS)} files; numbers_index.csv - {len(index)} rows")
    log(f"Font: {font}{'' if font == style['page']['font'] else ' (the fallback: ' + style['page']['font'] + ' not found)'}")
    log(f"Fig. 3: {unit}, region - block row {r}, column {c} of {BLOCKS} x {BLOCKS} (MGM {value:.6f}); encoded again and "
        f"equal to E4: {len(panels)} of {len(CODECS)}")
    log(f"Inputs: {E4_FILE} sha256 {e4.sha256}; {E5_FILE} sha256 {e5.sha256}; {E6_FILE} sha256 {e6.sha256}")
    return out, index


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite the figures and tables that exist")
    args = parser.parse_args(argv)
    try:
        paths = study_module("run_e4").load_paths()
        raw = json.loads((ROOT / "lab_paths.json").read_text(encoding="utf-8"))
        if not raw.get("papers_root"):
            raise RunRefusal("lab_paths.json has no key 'papers_root'")
        build(paths["cache_root"], (ROOT / raw["papers_root"]).resolve(), paths=paths, force=args.force)
    except RunRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
