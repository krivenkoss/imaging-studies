"""Figures, tables and captions of the paper P001 from the run files of E4, E5, E6 and the grid of cycle 9 - one command. Run: python studies/P001_vvc-dental/figures.py   (--force: overwrite files that exist)

Reads <cache_root>/cycles/13/e4_main.json (pinned sha256), cycles/17/e5_analysis.json, cycles/16/e6_pages.json and
cycles/9/p001_tiles.json, the copy data/P001 (Fig. 1 and Fig. 7) and the style figures_style.toml beside this script;
writes into <papers_root>/P001_vvc-dental/: figures/fig1..fig7 (.pdf, .png, .csv of the plotted data - no pixel),
figures/captions.txt (the captions of the figures and the titles of the tables), tables/table1..table3 (.xlsx, .csv)
and tables/numbers_index.csv - every number of every table cell, figure series and caption: the value as printed,
the run file (relative to cache_root), its sha256, the key in the file, the operation and the format (the source of
NUMBERS.md). The composition is the one declared in the tasks of cycles 19 and 22; the figures are numbered in the
order of their mention: Fig. 1 - the initial data, Fig. 2 - PSNR-HVS-M and PSNR on CR, Fig. 3 - PSNR-HVS-M on PCC,
Fig. 4 - the threshold PCC on MGM, Fig. 5 - the effect r with its CI, Fig. 6 - CR and r on the MGM of the patient,
Fig. 7 - against the original. All text is English with a decimal point; the codec is "VVC" (VTM only as the
implementation in Table 1); lines of different colour and style carry small markers, Fig. 4 and Fig. 5 - small markers
without lines (STUDY.md section 7, the owner's exception). Images: Fig. 1 - the image D10 of the copy data/P001 with its
grid (B11), a typical fragment and a fragment of the owner's choice; Fig. 7 - the original fragment and the four codecs
at the threshold of 42 dB from reconstructions encoded again with the wrappers of E4 (the function of run_e6.py), each
equal to its E4 row by sha256 of the file and of the reconstruction, otherwise a refusal in words. The PDF carries every
raster as it is; in the PNG every image panel lies on whole pixels and takes an integer multiple of its pixels (pixel
repetition, checked after writing) - the dpi follows from that and is printed. Medical images stay outside Git. A number
is never typed by hand. No file carries the time of the build: the PDF has no CreationDate, the xlsx has FIXED_DATE in
its properties and zip entries - two runs write the same bytes.
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
FRAGMENTS_LEVEL = "фрагменты"  # the level of the fragments in analysis.complexity.rho of E5 (run_e5.py)
FIG1_IMAGE = "D10"  # the image of Fig. 1 - the owner's choice (task of cycle 19)
FIG1_LETTERS = "bc"  # its fragments: (b) - MGM closest to the median MGM of all fragments, (c) - FIG1_C
FIG1_C = "D10_r00_c02"  # Fig. 1 (c) - owner's choice, cycle 21: for the demonstration, not by a rule
FIG1_C_PLACE = "from the upper middle part of the image"  # where (c) lies - the words of the caption (task of cycle 21)
FRAGMENT = 512  # Fig. 7: the whole fragment in the upper row
BLOCKS = 4      # Fig. 7: the region is a block of the BLOCKS x BLOCKS grid of the fragment (128 x 128)
ZOOM = 4        # Fig. 7: the lower row - the region repeated ZOOM times along each axis, no interpolation
MARKS_SHARE = 0.05  # PNG check: at most this share of an image panel may be covered by drawn marks (frames, letters)
JPEG50_PCC = "50"  # the point JPEG PCC 50 of E5 (STUDY.md section 6) - its PCC by definition
PCC_MEANING = {"vtm": "QP", "jpeg": "quality", "heif": "quality", "avif": "quality"}  # STUDY.md section 5
VTM_IMPLEMENTATION = "VTM 23.0 all-intra + GPAC MP4Box 26.08-DEV"  # STUDY.md section 5: E4 holds only the sha256
VERSION_FIELDS = {  # Table 1: the version numbers taken from header.codec_versions of E4 - (group, ((label, field), ...))
    "jpeg": ("jpeg", (("Pillow", "pillow"), ("libjpeg-turbo", "libjpeg_turbo"))),
    "heif": ("heif", (("pillow-heif", "pillow_heif"), ("libheif", "libheif"), ("x265", "x265"))),
    "avif": ("avif_libheif", (("pillow-heif", "pillow_heif"), ("libheif", "libheif"), ("libaom", "aom_encoder"))),
}
VERSION = re.compile(r"\d+(?:\.\d+)+(?:\+\d+)?")  # "1.23.6"; "4.3+1" of "x265 HEVC encoder (4.3+1-e9b8812)"
CONTAINER = {"vtm": "HEIF (vvc1)", "jpeg": "JFIF", "heif": "HEIF (hvc1)", "avif": "AVIF"}
OUTPUTS = [f"figures/fig{k}.{ext}" for k in range(1, 8) for ext in ("pdf", "png", "csv")] + ["figures/captions.txt"] + \
          [f"tables/table{k}.{ext}" for k in (1, 2, 3) for ext in ("xlsx", "csv")] + ["tables/numbers_index.csv"]
INDEX_COLUMNS = ("id", "value", "file", "sha256", "key", "op", "format")
PDF_METADATA = {"CreationDate": None}  # no time of the build in the PDF (owner, cycle 21)
FIXED_DATE = (1980, 1, 1, 0, 0, 0)  # the xlsx: created, modified, every zip entry - the first zip date (owner, cycle 21)


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
    """The one value all entries share - printed once; different values - a refusal."""
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


def threshold(e5, t):
    """The cell of the threshold t as printed, from thresholds_db of E5."""
    return cell(t, ".0f", e5, f"thresholds_db[{e5.data['thresholds_db'].index(t)}]")


# ---------------------------------------------------------------- the choice of fragments

def typical_fragments(fragments, image, n):
    """The n fragments of the image with MGM closest to the median MGM of all fragments, ties - the smaller code;
    (the median, the chosen entries in that order)."""
    median = float(np.median([f["mgm"] for f in fragments]))
    mine = sorted((f for f in fragments if f["image"] == image), key=lambda f: (abs(f["mgm"] - median), f["unit"]))
    if len(mine) < n:
        raise RunRefusal(f"{image} has {len(mine)} fragments in E5, {n} are needed")
    return median, mine[:n]


def fig1_fragments(fragments):
    """Fig. 1: (b) - the fragment of FIG1_IMAGE by the rule of typical_fragments, (c) - FIG1_C; (the median, [(b), (c)]).
    (c) not in E5 once, of another image or the same as (b) - a refusal."""
    median, (typical,) = typical_fragments(fragments, FIG1_IMAGE, 1)
    found = pick(fragments, unit=FIG1_C)
    if len(found) != 1:
        raise RunRefusal(f"Fig. 1 (c): {FIG1_C} is found {len(found)} times in analysis.complexity.fragments of E5, "
                         "once is expected")
    if found[0]["image"] != FIG1_IMAGE:
        raise RunRefusal(f"Fig. 1 (c): {FIG1_C} is a fragment of {found[0]['image']}, not of {FIG1_IMAGE}")
    if found[0]["unit"] == typical["unit"]:
        raise RunRefusal(f"Fig. 1 (c): {FIG1_C} is the fragment (b) by the rule as well")
    return median, [typical, found[0]]


def extreme_fragments(fragments):
    """The fragment of the simplest structure (the smallest MGM) and of the most complex one (the largest MGM); ties -
    the smaller code."""
    return (min(fragments, key=lambda f: (f["mgm"], f["unit"])), min(fragments, key=lambda f: (-f["mgm"], f["unit"])))


def fragment_cells(e5, unit, what):
    """The code and the MGM of a fragment as printed, from analysis.complexity.fragments of E5."""
    fragments = e5.data["analysis"]["complexity"]["fragments"]
    i = position(fragments, unit=unit)
    base = f"analysis.complexity.fragments[{i}]"
    return ((f"{what} fragment", cell(unit, "text", e5, f"{base}.unit")),
            (f"{what} MGM", cell(fragments[i]["mgm"], ".4f", e5, f"{base}.mgm")))


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
    title = joined("Table 1. Codecs: implementation and versions, PCC and its range for the fragments, container.")
    return head, rows, [], title


def effect(source, listname, i, c):
    base = f"analysis.{listname}[{i}]"
    return joined(("effect", cell(c["ratio"], ".3f", source, f"{base}.ratio")), " (",
                  ("CI low", cell(c["ratio_low"], ".3f", source, f"{base}.ratio_low")), "–",
                  ("CI high", cell(c["ratio_high"], ".3f", source, f"{base}.ratio_high")), ")")


def table2(e5, style):
    """Pairs by rows; patients and whole images at every threshold by columns, "effect (95% CI)"; p (Holm) and the
    conclusion of the rule of STUDY.md section 6 for the three main rows; a note under the table."""
    a, thresholds = e5.data["analysis"], e5.data["thresholds_db"]
    vvc = style["codecs"]["vtm"]["label"]
    columns = [(listname, t, f"{label}, {t:.0f} dB") for listname, label in
               (("comparisons", "Patients"), ("whole_comparisons", "Whole images")) for t in thresholds]
    head = ("Pair", *(name for _, _, name in columns), f"p (Holm), {T_MAIN:.0f} dB, patients", "Conclusion")
    rows = []
    for b in (*OTHERS, "jpeg_opt"):
        pair = f"vtm/{b}"
        line_ = [constant(f"{vvc} / JPEG (optimised Huffman)" if b == "jpeg_opt" else f"{vvc} / {style['codecs'][b]['label']}")]
        for listname, t, _ in columns:
            i = position(a[listname], pair=pair, t_db=t)
            line_.append(effect(e5, listname, i, a[listname][i]))
        i = position(a["comparisons"], pair=pair, t_db=T_MAIN)
        c, base = a["comparisons"][i], f"analysis.comparisons[{i}]"
        if c["role"] == "основное":
            line_ += [cell(c["p_holm"], ".3g", e5, f"{base}.p_holm"),
                      cell(verdict_en(c["verdict"]), "text", e5, f"{base}.verdict", "verdict")]
        else:
            line_ += [constant("-"), constant("-")]
        rows.append(line_)
    whole, main = a["whole_comparisons"], pick(a["comparisons"], role="основное")
    note = joined(
        f"Note. Effect - the Hodges-Lehmann estimate of r = CR({vvc}) / CR(codec) with its 95% CI; n = ",
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
    t = [threshold(e5, x) for x in thresholds]
    title = joined(f"Table 2. Comparison of {vvc} with JPEG, HEIF and AVIF: the effect r = CR({vvc}) / CR(codec) "
                   "(Hodges-Lehmann estimate) with its ", ("CI level", constant("95")), "% CI for the patients and the "
                   "whole images at T = ", ("T 1", t[0]), ", ", ("T 2", t[1]), " and ", ("T 3", t[2]),
                   " dB; p (Holm) and the conclusion of the decision rule for the three main comparisons.")
    return head, rows, [note], title


def spread(source, key, values, centre, low, high, fmt):
    """"median (low-high)" of the values under the key: low and high - quartiles or min and max."""
    stat = {"median": lambda: percentile(values, 50), "q25": lambda: percentile(values, 25),
            "q75": lambda: percentile(values, 75), "min": lambda: min(values), "max": lambda: max(values)}
    return joined((centre, cell(stat[centre](), fmt, source, key, centre)), " (",
                  (low, cell(stat[low](), fmt, source, key, low)), "–", (high, cell(stat[high](), fmt, source, key, high)),
                  ")")


def table3(e5, style):
    """At the threshold of 42 dB by codec: CR over the patients; the threshold PCC, PSNR and HaarPSI over the
    fragments; and the point JPEG PCC 50 with the share of units at or above 42 dB."""
    a = e5.data["analysis"]
    rows, counts = [], []
    for c in CODECS:
        pw = dict(codec=c, t_db=T_MAIN)
        uw = dict(level="f", codec=c, t_db=T_MAIN, judged=True)
        crs = [p["median_cr"] for p in pick(a["patients"], **pw)]
        units = pick(a["units"], **uw)
        counts.append(len(units))
        pk, uk = f"analysis.patients[{where_text(**pw)}]", f"analysis.units[{where_text(**uw)}]"
        rows.append([constant(style["codecs"][c]["label"]), cell(len(crs), "d", e5, pk, "count"),
                     spread(e5, f"{pk}.median_cr", crs, "median", "q25", "q75", ".2f"), cell(len(units), "d", e5, uk, "count"),
                     spread(e5, f"{uk}.pcc", [u["pcc"] for u in units], "median", "min", "max", "g"),
                     spread(e5, f"{uk}.psnr_db", [number(u["psnr_db"]) for u in units], "median", "min", "max", ".2f"),
                     spread(e5, f"{uk}.haarpsi", [u["haarpsi"] for u in units], "median", "min", "max", ".4f"),
                     constant("-")])
    head = ("Codec or point", "n patients", "CR, median (q25–q75)", "n fragments",
            f"Threshold PCC for T = {T_MAIN:.0f} dB, median (min–max), {same(counts, 'n fragments')} fragments",
            "PSNR, dB, median (min–max)", "HaarPSI, median (min–max)",
            f"PSNR-HVS-M ≥ {T_MAIN:.0f} dB: fragments; whole images")
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
    rows.append([joined(f"JPEG PCC {JPEG50_PCC} (", ("name", label), ")") if name != "JPEG PCC 50" else label,
                 cell(len(j["patients"]), "d", e5, "analysis.jpeg50.patients", "count"),
                 spread(e5, "analysis.jpeg50.patients.median_cr", [p["median_cr"] for p in j["patients"]], "median", "q25",
                        "q75", ".2f"),
                 cell(len(units), "d", e5, "analysis.jpeg50.units[level=f]", "count"),
                 constant(JPEG50_PCC),
                 spread(e5, "analysis.jpeg50.units[level=f].psnr_db", [number(u["psnr_db"]) for u in units], "median", "min",
                        "max", ".2f"),
                 spread(e5, "analysis.jpeg50.units[level=f].haarpsi", [u["haarpsi"] for u in units], "median", "min", "max",
                        ".4f"),
                 joined(*reached)])
    pw, uw = dict(codec="vtm", t_db=T_MAIN), dict(level="f", codec="vtm", t_db=T_MAIN, judged=True)
    title = joined("Table 3. At the threshold T = ", ("T", threshold(e5, T_MAIN)), " dB by codec and at the point JPEG PCC ",
                   ("JPEG point", constant(JPEG50_PCC)), ": CR over n = ",
                   ("n patients", cell(len(pick(a["patients"], **pw)), "d", e5, f"analysis.patients[{where_text(**pw)}]",
                                       "count")),
                   " patients; the threshold PCC, PSNR and HaarPSI over n = ",
                   ("n fragments", cell(len(pick(a["units"], **uw)), "d", e5, f"analysis.units[{where_text(**uw)}]",
                                        "count")),
                   " fragments; at JPEG PCC ", ("JPEG point again", constant(JPEG50_PCC)),
                   " - the share of the units with PSNR-HVS-M ≥ ", ("T again", threshold(e5, T_MAIN)), " dB.")
    return head, rows, [], title


# ---------------------------------------------------------------- figures: common

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
                         "axes.titlesize": p["font_size_pt"], "axes.labelsize": p["font_size_pt"],
                         "xtick.labelsize": p["font_size_pt"], "ytick.labelsize": p["font_size_pt"],
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


def points(pt, dpi):
    """Points as whole pixels at the dpi, rounded up."""
    return math.ceil(pt * dpi / 72)


def pixel_axes(fig, box, canvas):
    """Axes over the box (x, y, w, h) in whole pixels from the top left of a canvas (W, H) in pixels, no frame."""
    (x, y, w, h), (width, height) = box, canvas
    ax = fig.add_axes((x / width, 1 - (y + h) / height, w / width, h / height))
    ax.set_axis_off()
    return ax


def grey(ax, image):
    """The 8-bit values as they are: grey RGB of the very values, no colour map, no interpolation, no stretch of levels;
    the image fills its axes. Marks drawn over an image are not antialiased: a blended edge would turn image pixels into
    other grey levels."""
    ax.imshow(np.stack([image] * 3, axis=-1), interpolation="none", aspect="auto")


def check_png(path, panels):
    """Every image panel of the PNG: its box holds its raster repeated `scale` times along each axis, pixel for pixel;
    under the drawn marks (coloured frames and letters) only the grey pixels count, the marks cover at most
    MARKS_SHARE of the panel; otherwise a refusal."""
    from PIL import Image

    png = np.asarray(Image.open(path).convert("RGB"))
    for p in panels:
        x, y, w, h = p["box"]
        k = p["scale"]
        if k != int(k) or k < 1 or (w, h) != (p["array"].shape[1] * k, p["array"].shape[0] * k):
            raise RunRefusal(f"{path.name}, panel {p['panel']}: the box {w} x {h} is not a whole multiple of "
                             f"{p['array'].shape[1]} x {p['array'].shape[0]} (scale {k})")
        want = np.repeat(np.repeat(p["array"], int(k), axis=0), int(k), axis=1)
        got = png[y:y + h, x:x + w]
        flat = (got[..., 0] == got[..., 1]) & (got[..., 1] == got[..., 2])
        if p["marks"]:
            equal = np.array_equal(got[..., 0][flat], want[flat]) and float((~flat).mean()) <= MARKS_SHARE
        else:
            equal = bool(flat.all()) and np.array_equal(got[..., 0], want)
        if not equal:
            raise RunRefusal(f"{path.name}, panel {p['panel']}: the PNG is not the raster repeated {k} times pixel for pixel")


def panel_rows(panels, dpi):
    return [{"kind": "png panel", "panel": p["panel"], "unit": p["unit"], "png_x": p["box"][0], "png_y": p["box"][1],
             "png_w": p["box"][2], "png_h": p["box"][3], "scale": p["scale"], "dpi": dpi} for p in panels]


# ---------------------------------------------------------------- Fig. 1: the initial data

def fig1_layout(width, height, side, style):
    """One row over the full width - the image (width x height) and the fragments (side x side in the PNG). The dpi
    is the smallest at which the PNG row with its gaps and margins fits the width; (dpi, canvas, PNG boxes, PDF boxes,
    the top of the row of letters): PNG boxes - whole pixels, the image 1 : 1, the fragments at their scale, centred in
    height; PDF boxes - the three panels of one height with equal gaps from margin to margin of the same canvas."""
    p = style["page"]
    n, letters = 1 + len(FIG1_LETTERS), "a" + FIG1_LETTERS
    width_in = p["double_column_cm"] / 2.54
    spaces_in = ((n - 1) * p["image_gap_pt"] + 2 * p["image_margin_pt"]) / 72
    dpi = math.ceil((width + len(FIG1_LETTERS) * side) / (width_in - spaces_in))
    gap, margin = points(p["image_gap_pt"], dpi), points(p["image_margin_pt"], dpi)
    row = max(height, side)
    canvas = (round(width_in * dpi), margin + row + points(1.6 * p["font_size_pt"], dpi) + margin)
    sizes = [(width, height)] + [(side, side)] * len(FIG1_LETTERS)
    x = (canvas[0] - sum(w for w, _ in sizes) - (n - 1) * gap) // 2
    if x < 0:
        raise RunRefusal(f"Fig. 1: the row of {sum(w for w, _ in sizes)} pixels does not fit {canvas[0]} at {dpi} dpi")
    png = {}
    for letter, (w, h) in zip(letters, sizes):
        png[letter] = (x, margin + (row - h) // 2, w, h)
        x += w + gap
    h = (canvas[0] - 2 * margin - (n - 1) * gap) / (width / height + len(FIG1_LETTERS))
    pdf, x = {}, margin
    for letter, w in zip(letters, [h * width / height] + [h] * len(FIG1_LETTERS)):
        pdf[letter] = (x, margin + (row - h) / 2, w, h)
        x += w + gap
    return dpi, canvas, png, pdf, margin + row


def fig1(e5, tiles, arrays, style, plt, target):
    """One row: (a) the whole image with the grid of its fragments (B11), the frames of (b) and (c) marked and lettered;
    (b), (c) - its fragments of fig1_fragments; the letters of the panels below the row. `target` - "png": the image
    1 : 1 and the fragments repeated fig1_fragment_scale times on whole pixels; "pdf": the three rasters of one height
    (fig1_layout)."""
    from matplotlib.patches import Rectangle

    p, f = style["page"], style["figures"]
    fragments = e5.data["analysis"]["complexity"]["fragments"]
    median, picks = fig1_fragments(fragments)
    tile, grid = tiles.data["parameters"]["tile"], [r for r in tiles.data["rows"] if r["code"] == FIG1_IMAGE]
    whole = arrays[FIG1_IMAGE]
    height, width = whole.shape
    k = f["fig1_fragment_scale"]
    dpi, canvas, png_boxes, pdf_boxes, row_bottom = fig1_layout(width, height, int(tile * k), style)
    boxes = png_boxes if target == "png" else pdf_boxes
    fig = plt.figure(figsize=(canvas[0] / dpi, canvas[1] / dpi), dpi=dpi)
    ax = pixel_axes(fig, boxes["a"], canvas)
    grey(ax, whole)
    letters = {x["unit"]: letter for letter, x in zip(FIG1_LETTERS, picks)}
    for r in grid:
        letter = letters.get(r["tile"])
        ax.add_patch(Rectangle((r["x"] - 0.5, r["y"] - 0.5), tile, tile, fill=False,
                               edgecolor=f["fig1_mark_color"] if letter else f["fig1_grid_color"],
                               linewidth=f["fig1_mark_width_pt"] if letter else f["fig1_grid_width_pt"], antialiased=False,
                               zorder=3 if letter else 2))
        if letter:
            ax.text(r["x"] + tile * 0.04, r["y"] + tile * 0.04, letter, color=f["fig1_mark_color"], ha="left", va="top",
                    fontsize=p["font_size_pt"], antialiased=False)
    panels = [{"panel": "a", "unit": FIG1_IMAGE, "array": whole, "box": png_boxes["a"], "scale": 1, "marks": True}]
    for letter, x in zip(FIG1_LETTERS, picks):
        r = next(r for r in grid if r["tile"] == x["unit"])
        if not np.array_equal(arrays[x["unit"]], whole[r["y"]:r["y"] + tile, r["x"]:r["x"] + tile]):
            raise RunRefusal(f"Fig. 1: {x['unit']} differs from its place in {FIG1_IMAGE} by the grid of cycle 9")
        grey(pixel_axes(fig, boxes[letter], canvas), arrays[x["unit"]])
        panels.append({"panel": letter, "unit": x["unit"], "array": arrays[x["unit"]], "box": png_boxes[letter], "scale": k,
                       "marks": False})
    label = canvas[1] - row_bottom - points(p["image_margin_pt"], dpi)
    for letter, (x, y, w, h) in boxes.items():
        fig.text((x + w / 2) / canvas[0], 1 - (row_bottom + label / 2) / canvas[1], f"({letter})", ha="center", va="center")
    mgms = {x["unit"]: x["mgm"] for x in fragments}
    data = [{"kind": "tile", "unit": r["tile"], "x": r["x"], "y": r["y"], "mgm": mgms[r["tile"]],
             "panel": letters.get(r["tile"], "")} for r in grid]
    data.append({"kind": "median MGM", "mgm": median, "n": len(fragments)})
    data += panel_rows(panels, dpi)
    data += [{"kind": "pdf panel", "panel": letter, "pdf_x_cm": round(x / dpi * 2.54, 4), "pdf_y_cm": round(y / dpi * 2.54, 4),
              "pdf_w_cm": round(w / dpi * 2.54, 4), "pdf_h_cm": round(h / dpi * 2.54, 4)}
             for letter, (x, y, w, h) in pdf_boxes.items()]
    size_cell = cell(tile, "d", tiles, "parameters.tile")
    median_cell = cell(median, ".4f", e5, "analysis.complexity.fragments.mgm", "median")
    count_cell = cell(len(fragments), "d", e5, "analysis.complexity.fragments", "count")
    chosen = [fragment_cells(e5, x["unit"], f"({letter})") for letter, x in zip(FIG1_LETTERS, picks)]
    series = [("Fig. 1 / fragment side, pixels", size_cell),
              ("Fig. 1 / grid frames", cell(len(grid), "d", tiles, f"rows[code={FIG1_IMAGE}]", "count")),
              ("Fig. 1 / median MGM", median_cell), ("Fig. 1 / fragments", count_cell)] + \
             [(f"Fig. 1 / {name}", c) for pair in chosen for name, c in pair]
    caption = joined("Fig. 1. The dental image ", ("image", constant(FIG1_IMAGE)), " with the grid of its ",
                     ("fragment width", size_cell), " × ", ("fragment height", size_cell),
                     " pixel fragments (a); its fragment with MGM closest to the median MGM of the ",
                     ("fragments", count_cell), " fragments (", ("median MGM", median_cell), "), ",
                     chosen[0][0], ", MGM = ", chosen[0][1], " (b); and the fragment ", FIG1_C_PLACE, ", ",
                     chosen[1][0], ", MGM = ", chosen[1][1], " (c).")
    return fig, data, series, dpi, panels, caption


# ---------------------------------------------------------------- Fig. 2 to Fig. 6: the plots

def fig2(e4, e5, style, plt):
    """Dependences of PSNR-HVS-M (a, b) and PSNR (c, d) on CR for the fragment of the simplest structure (a, c) and of
    the most complex one (b, d): four lines with small markers, the threshold lines in grey on (a, b), one legend."""
    simple, complex_ = extreme_fragments(e5.data["analysis"]["complexity"]["fragments"])
    fig, axes = plt.subplots(2, 2, figsize=size(style, "fig2"), constrained_layout=True)
    t = style["thresholds"]
    data, series = [], []
    for col, (frag, what) in enumerate(((simple, "simple structure"), (complex_, "complex structure"))):
        unit = frag["unit"]
        for row, (metric, field) in enumerate((("PSNR-HVS-M", "psnr_hvs_m_db"), ("PSNR", "psnr_db"))):
            ax, letter = axes[row, col], "abcd"[2 * row + col]
            for c in CODECS:
                s = style["codecs"][c]
                rows = sorted((r for r in e4.data["rows"] if r["unit"] == unit and r["codec"] == c and r["status"] == "valid"),
                              key=lambda r: r["cr_file"])
                line(ax, [r["cr_file"] for r in rows], [r[field] for r in rows], s, style, s["label"])
                data += [{"panel": letter, "fragment": unit, "codec": c, "pcc": r["param"], "cr": r["cr_file"],
                          field: r[field]} for r in rows]
                series.append((f"Fig. 2 ({letter}) / {s['label']}",
                               cell(len(rows), "d", e4, f"rows[{where_text(unit=unit, codec=c, status='valid')}]", "count")))
            if row == 0:
                for level in e5.data["thresholds_db"]:
                    ax.axhline(level, color=t["color"], linestyle=t[f"line_{level:.0f}"],
                               linewidth=style["page"]["line_width_pt"], label=f"T = {level:.0f} dB")
            ax.set_xscale("log")
            plain_log(ax.xaxis)
            ax.set_xlabel("CR")
            ax.set_ylabel(f"{metric}, dB")
            ax.set_title(f"({letter}) {what}", loc="left")
    series.append(("Fig. 2 / threshold lines, dB", cell("; ".join(show(x, ".0f") for x in e5.data["thresholds_db"]), "text",
                                                       e5, "thresholds_db", "list")))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=style["figures"]["fig2_legend_columns"])
    cells = [fragment_cells(e5, x["unit"], what) for x, what in ((simple, "simple"), (complex_, "complex"))]
    series += [(f"Fig. 2 / {name}", c) for pair in cells for name, c in pair]
    th = [threshold(e5, x) for x in e5.data["thresholds_db"]]
    caption = joined("Fig. 2. Dependences of PSNR-HVS-M (a,b) and PSNR (c,d) on compression ratio (CR) for the fragments "
                     "of simple structure ", cells[0][0], ", MGM = ", cells[0][1], " (a,c) and complex structure ",
                     cells[1][0], ", MGM = ", cells[1][1], " (b,d); horizontal lines - the thresholds T = ", ("T 1", th[0]),
                     ", ", ("T 2", th[1]), " and ", ("T 3", th[2]), " dB.")
    return fig, data, series, style["page"]["dpi"], [], caption


def fig3(e4, e5, style, plt):
    """Dependences of PSNR-HVS-M on PCC, a panel per codec, for the fragments of Fig. 2: the simple one with the line
    fig3_simple_line, the complex one with fig3_complex_line, both in the colour and with the small markers of the codec;
    the valid rows of E4 only (as Fig. 2), the x axis over the whole PCC range of the plan of E4; the threshold lines in
    grey and their legend above the panels."""
    simple, complex_ = extreme_fragments(e5.data["analysis"]["complexity"]["fragments"])
    f, t = style["figures"], style["thresholds"]
    fig, axes = plt.subplots(2, 2, figsize=size(style, "fig3"), constrained_layout=True)
    plan = e4.data["header"]["plan"]
    cells = [fragment_cells(e5, x["unit"], what)[0] for x, what in ((simple, "simple"), (complex_, "complex"))]
    data, series = [], [(f"Fig. 3 / {name}", c) for name, c in cells]
    for k, (ax, c) in enumerate(zip(axes.flat, CODECS)):
        s, letter = style["codecs"][c], "abcd"[k]
        lines = []
        for frag, what in ((simple, "simple"), (complex_, "complex")):
            unit = frag["unit"]
            rows = sorted((r for r in e4.data["rows"] if r["unit"] == unit and r["codec"] == c and r["status"] == "valid"),
                          key=lambda r: r["param"])
            line(ax, [r["param"] for r in rows], [r["psnr_hvs_m_db"] for r in rows], {**s, "line": f[f"fig3_{what}_line"]},
                 style, f"{what}, {unit}")
            lines.append(ax.get_lines()[-1])
            data += [{"panel": letter, "codec": c, "fragment": unit, "structure": what, "pcc": r["param"],
                      "psnr_hvs_m_db": r["psnr_hvs_m_db"]} for r in rows]
            series.append((f"Fig. 3 ({letter}) / {s['label']} / {what}",
                           cell(len(rows), "d", e4, f"rows[{where_text(unit=unit, codec=c, status='valid')}]", "count")))
        for level in e5.data["thresholds_db"]:
            ax.axhline(level, color=t["color"], linestyle=t[f"line_{level:.0f}"], linewidth=style["page"]["line_width_pt"],
                       label=f"T = {level:.0f} dB")
        values, path = (plan["vtm"]["f"], "header.plan.vtm.f") if c == "vtm" else (plan[c], f"header.plan.{c}")
        ax.set_xlim(min(values), max(values))
        series.append((f"Fig. 3 ({letter}) / x: PCC range", cell(f"{min(values)} to {max(values)}", "text", e4, path, "range")))
        ax.legend(handles=lines, loc=f["fig3_legend"])
        ax.set_xlabel(f"PCC ({PCC_MEANING[c]})")
        ax.set_ylabel("PSNR-HVS-M, dB")
        ax.set_title(f"({letter}) {s['label']}", loc="left")
    series.append(("Fig. 3 / threshold lines, dB", cell("; ".join(show(x, ".0f") for x in e5.data["thresholds_db"]), "text",
                                                       e5, "thresholds_db", "list")))
    handles = [x for x in axes[0, 0].get_lines() if x.get_label().startswith("T = ")]
    fig.legend(handles, [x.get_label() for x in handles], loc="outside upper center", ncols=len(handles))
    labels = [style["codecs"][c]["label"] for c in CODECS]
    th = [threshold(e5, x) for x in e5.data["thresholds_db"]]
    caption = joined(f"Fig. 3. Dependences of PSNR-HVS-M on PCC for {labels[0]} (a), {labels[1]} (b), {labels[2]} (c) and "
                     f"{labels[3]} (d) for the fragments of simple structure ", cells[0], " (solid lines) and complex "
                     "structure ", cells[1], " (dashed lines); horizontal lines - the thresholds T = ", ("T 1", th[0]), ", ",
                     ("T 2", th[1]), " and ", ("T 3", th[2]), " dB.")
    return fig, data, series, style["page"]["dpi"], [], caption


def fig4(e5, style, plt):
    """The threshold PCC at 42 dB of every fragment against its MGM, a panel per codec: small markers of the codec
    without lines (the owner's exception, STUDY.md section 7); Spearman's rho of E5 in the title of the panel; the PCC
    axis in whole numbers."""
    from matplotlib.ticker import MaxNLocator

    a = e5.data["analysis"]["complexity"]
    fragments = sorted(a["fragments"], key=lambda x: (x["mgm"], x["unit"]))
    p = style["page"]
    fig, axes = plt.subplots(1, len(CODECS), figsize=size(style, "fig4"), constrained_layout=True)
    count = cell(len(fragments), "d", e5, "analysis.complexity.fragments", "count")
    data, series = [], []
    for k, (ax, c) in enumerate(zip(axes, CODECS)):
        s, letter, field = style["codecs"][c], "abcd"[k], f"pcc_{c}"
        ax.plot([x["mgm"] for x in fragments], [x[field] for x in fragments], linestyle="none", marker=s["marker"],
                markersize=p["marker_size_pt"], color=s["color"], label=s["label"])
        i = position(a["rho"], level=FRAGMENTS_LEVEL, key=field)
        rho = cell(a["rho"][i]["rho"], ".3f", e5, f"analysis.complexity.rho[{i}].rho")
        ax.set_title(f"({letter}) {s['label']}, ρ = {rho['value']}", loc="left")
        ax.set_xlabel("MGM")
        ax.set_ylabel(f"Threshold PCC ({PCC_MEANING[c]}), {T_MAIN:.0f} dB")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        data += [{"panel": letter, "codec": c, "fragment": x["unit"], "mgm": x["mgm"], "threshold_pcc": x[field]}
                 for x in fragments]
        series += [(f"Fig. 4 ({letter}) / {s['label']} / points",
                    cell(len(fragments), "d", e5, f"analysis.complexity.fragments.{field}", "count")),
                   (f"Fig. 4 ({letter}) / {s['label']} / rho", rho)]
    rho_n = cell(same([x["n"] for x in pick(a["rho"], level=FRAGMENTS_LEVEL)], "n of rho"), "d", e5,
                 f"analysis.complexity.rho[level={FRAGMENTS_LEVEL}].n", "same")
    if rho_n["value"] != count["value"]:
        raise RunRefusal(f"Fig. 4: rho of E5 is over {rho_n['value']} fragments, the panels show {count['value']}")
    series.append(("Fig. 4 / n of rho", rho_n))
    labels = [style["codecs"][c]["label"] for c in CODECS]
    caption = joined("Fig. 4. Threshold PCC at T = ", ("T", threshold(e5, T_MAIN)), " dB versus MGM for the ",
                     ("fragments", count), f" fragments: {labels[0]} (a), {labels[1]} (b), {labels[2]} (c) and {labels[3]} "
                     "(d); ρ - Spearman rank correlation.")
    return fig, data, series, p["dpi"], [], caption


def fig5(e5, style, plt):
    """The effect r = CR(VVC) / CR(codec) with its 95% CI (the same numbers as Table 2) - (a) the patients, (b) the
    whole images; a row per pair and threshold; the estimate - a small marker of the second codec without a line, the CI
    - a horizontal segment in its colour; the vertical line r = 1."""
    a, thresholds = e5.data["analysis"], e5.data["thresholds_db"]
    p, t = style["page"], style["thresholds"]
    vvc = style["codecs"]["vtm"]["label"]
    fig, axes = plt.subplots(1, 2, figsize=size(style, "fig5"), sharex=True, sharey=True, constrained_layout=True)
    rows = [(b, level) for b in OTHERS for level in thresholds]
    one = constant("1")
    data, series = [], []
    for k, (ax, (listname, what)) in enumerate(zip(axes, (("comparisons", "patients"), ("whole_comparisons",
                                                                                         "whole images")))):
        letter = "ab"[k]
        for b in OTHERS:
            s, pair = style["codecs"][b], f"vtm/{b}"
            for level in thresholds:
                y = len(rows) - 1 - rows.index((b, level))
                x = a[listname][position(a[listname], pair=pair, t_db=level)]
                ax.plot([x["ratio_low"], x["ratio_high"]], [y, y], color=s["color"], linestyle="solid",
                        linewidth=p["line_width_pt"])
                ax.plot([x["ratio"]], [y], linestyle="none", marker=s["marker"], markersize=p["marker_size_pt"],
                        color=s["color"])
                data.append({"panel": letter, "level": what, "pair": pair, "t_db": level, "n": x["n"], "ratio": x["ratio"],
                             "ratio_low": x["ratio_low"], "ratio_high": x["ratio_high"]})
            series.append((f"Fig. 5 ({letter}) / {vvc} / {s['label']}",
                           cell(len([x for x in a[listname] if x["pair"] == pair]), "d", e5,
                                f"analysis.{listname}[pair={pair}]", "count")))
        ax.axvline(float(one["value"]), color=t["color"], linestyle=t["unity_line"], linewidth=p["line_width_pt"],
                   label=f"r = {one['value']}")
        ax.set_title(f"({letter}) {what}", loc="left")
        ax.set_xlabel(f"r = CR({vvc}) / CR(codec)")
    axes[0].set_yticks(range(len(rows)), [f"{vvc} / {style['codecs'][b]['label']}, {level:.0f} dB"
                                          for b, level in reversed(rows)])
    series.append(("Fig. 5 / reference line r", one))
    n, n_whole = (cell(same([x["n"] for x in a[name]], f"n of {name}"), "d", e5, f"analysis.{name}.n", "same")
                  for name in ("comparisons", "whole_comparisons"))
    if n["value"] != n_whole["value"]:
        raise RunRefusal(f"Fig. 5: n = {n['value']} patients and n = {n_whole['value']} whole images - one n is printed")
    series.append(("Fig. 5 (b) / n whole images", n_whole))
    th = [threshold(e5, x) for x in thresholds]
    caption = joined(f"Fig. 5. The effect r = CR({vvc}) / CR(codec) (Hodges-Lehmann estimate) with its ",
                     ("CI level", constant("95")), "% CI for the patients (a) and the whole images (b) at T = ",
                     ("T 1", th[0]), ", ", ("T 2", th[1]), " and ", ("T 3", th[2]), " dB; n = ", ("n", n),
                     "; dashed line - r = ", ("reference", one), ".")
    return fig, data, series, p["dpi"], [], caption


def fig6(e5, style, plt):
    """Two panels over the MGM of the 16 patients at 42 dB, points in ascending MGM joined by lines: (a) the median CR
    of every codec; (b) r = CR(VVC) / CR(codec) of the three pairs and the line r = 1."""
    a = e5.data["analysis"]
    vvc = style["codecs"]["vtm"]["label"]
    mgms = {p["image"]: p["mgm"] for p in a["complexity"]["patients"]}
    by_mgm = lambda p: (mgms[p["image"]], p["image"])  # noqa: E731
    fig, (top, bottom) = plt.subplots(2, 1, figsize=size(style, "fig6"), sharex=True, constrained_layout=True)
    patients = cell(len(mgms), "d", e5, "analysis.complexity.patients", "count")
    data, series = [], [("Fig. 6 / x: MGM of the patients", patients)]
    for c in CODECS:
        s = style["codecs"][c]
        pts = sorted(pick(a["patients"], codec=c, t_db=T_MAIN), key=by_mgm)
        line(top, [mgms[p["image"]] for p in pts], [p["median_cr"] for p in pts], s, style, s["label"])
        data += [{"panel": "a", "series": c, "patient": p["image"], "mgm": mgms[p["image"]], "value": p["median_cr"]}
                 for p in pts]
        series.append((f"Fig. 6 (a) / {s['label']}", cell(len(pts), "d", e5,
                                                           f"analysis.patients[{where_text(codec=c, t_db=T_MAIN)}]", "count")))
    for b in OTHERS:
        s, pair = style["codecs"][b], f"vtm/{b}"
        pts = sorted(pick(a["ratios"], pair=pair, t_db=T_MAIN, in_test=True), key=by_mgm)
        line(bottom, [mgms[p["image"]] for p in pts], [p["r"] for p in pts], s, style, f"{vvc} / {s['label']}")
        data += [{"panel": "b", "series": pair, "patient": p["image"], "mgm": mgms[p["image"]], "value": p["r"]}
                 for p in pts]
        series.append((f"Fig. 6 (b) / {vvc} / {s['label']}",
                       cell(len(pts), "d", e5, f"analysis.ratios[{where_text(pair=pair, t_db=T_MAIN, in_test=True)}]",
                            "count")))
    t = style["thresholds"]
    one = constant("1")
    bottom.axhline(1.0, color=t["color"], linestyle=t["unity_line"], linewidth=style["page"]["line_width_pt"],
                   label=f"r = {one['value']}")
    series.append(("Fig. 6 (b) / reference line r", one))
    top.set_title("(a)", loc="left")
    bottom.set_title("(b)", loc="left")
    top.set_ylabel(f"CR at {T_MAIN:.0f} dB, median")
    bottom.set_ylabel(f"r = CR({vvc}) / CR(codec)")
    bottom.set_xlabel("MGM of the patient (median over fragments)")
    f = style["figures"]
    for ax in (top, bottom):
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + (high - low) * f["fig6_legend_headroom"])
        ax.legend(loc=f["fig6_legend"], ncols=f["fig6_legend_columns"])
    caption = joined("Fig. 6. Dependences of the median CR at T = ", ("T", threshold(e5, T_MAIN)), " dB (a) and of r = "
                     f"CR({vvc}) / CR(codec) (b) on the MGM of the patient (the median over the fragments of the image) "
                     "for n = ", ("n patients", patients), " patients; dashed line - r = ", ("reference", one), ".")
    return fig, data, series, style["page"]["dpi"], [], caption


# ---------------------------------------------------------------- Fig. 7: against the original

def study_module(name):
    spec = importlib.util.spec_from_file_location(name, STUDY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_units(paths, codes):
    """The original units (whole images and fragments) of the copy data/P001 and the grid of cycle 9, by the loaders
    of E4 (B10, B11); {code: uint8 array}."""
    run_e4 = study_module("run_e4")
    registry, tile_rows, tile = run_e4.load_inputs(paths["data_root"], Path(paths["cache_root"]) / TILES_FILE)
    units = run_e4.load_units(paths["data_root"], registry, tile_rows, tile)
    return {code: np.ascontiguousarray(units[code]["array"]) for code in codes}


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


def comparison(e4, e5, e6, arrays, paths, work, log):
    """The fragment of the first page of E6 (the largest MGM of E5) and its four reconstructions at the threshold PCC of
    42 dB, every one encoded again and equal to its E4 row by sha256 of the file and of the reconstruction."""
    unit = e6.data["pages"][0]["unit"]
    largest = extreme_fragments(e5.data["analysis"]["complexity"]["fragments"])[1]["unit"]
    if unit != largest:
        raise RunRefusal(f"the first page of E6 is {unit}, the fragment with the largest MGM of E5 is {largest}")
    original = arrays[unit]
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
        log(f"  Fig. 7: {unit}, {c} PCC {u['pcc']} encoded again - file and reconstruction equal to E4 by sha256")
    return unit, original, panels


def fig7(e5, e6, unit, original, recons, style, plt):
    """Upper row - the original and the four codecs at the threshold of 42 dB, the whole fragment; lower row - the same
    region of every panel repeated ZOOM times (pixel repetition); the frame of the region on the upper row. In the PNG
    every panel is repeated fig7_scale times; the dpi follows."""
    from matplotlib.patches import Rectangle

    p, f = style["page"], style["figures"]
    r, c, value, scores = region_of(original)
    b = original.shape[0] // BLOCKS
    k = f["fig7_scale"]
    side, n = int(FRAGMENT * k), 1 + len(recons)
    width_in = p["double_column_cm"] / 2.54
    dpi = math.floor(n * side / (width_in - ((n - 1) * p["image_gap_pt"] + 2 * p["image_margin_pt"]) / 72))
    canvas_w, gap, margin = round(width_in * dpi), points(p["image_gap_pt"], dpi), points(p["image_margin_pt"], dpi)
    left = (canvas_w - n * side - (n - 1) * gap) // 2
    top = margin + points(2 * 1.3 * p["font_size_pt"] + 4, dpi)  # two lines of the title and its pad
    canvas = (canvas_w, top + 2 * side + gap + margin)
    fig = plt.figure(figsize=(canvas[0] / dpi, canvas[1] / dpi), dpi=dpi)
    series = [("Fig. 7 / fragment", cell(unit, "text", e6, "pages[0].unit"))]
    data = [{"kind": "block", "fragment": unit, "row": q[0], "column": q[1], "mgm": v} for q, v in sorted(scores.items())]
    data.append({"kind": "region", "fragment": unit, "row": r, "column": c, "mgm": value, "block": b, "zoom": ZOOM})
    titles = ["Original\n "]
    for x in recons:
        s, u, base = style["codecs"][x["codec"]], x["unit"], f"analysis.units[{x['i']}]"
        pcc, psnr, cr = (cell(u["pcc"], "d", e5, f"{base}.pcc"), cell(u["psnr_hvs_m_db"], ".2f", e5, f"{base}.psnr_hvs_m_db"),
                         cell(u["cr"], ".2f", e5, f"{base}.cr"))
        titles.append(f"{s['label']}\nPCC {pcc['value']}, {psnr['value']} dB, CR {cr['value']}")
        series += [(f"Fig. 7 / {s['label']} / PCC", pcc), (f"Fig. 7 / {s['label']} / PSNR-HVS-M", psnr),
                   (f"Fig. 7 / {s['label']} / CR", cr)]
        data.append({"kind": "panel", "fragment": unit, "codec": x["codec"], "pcc": u["pcc"],
                     "psnr_hvs_m_db": u["psnr_hvs_m_db"], "cr": u["cr"], "e4_file_sha256": x["row"]["file_sha256"],
                     "e4_recon_sha256": x["row"]["recon_sha256"], "encoded_again_equal": 1})
    panels = []
    for j, (name, image) in enumerate(zip(["original", *(x["codec"] for x in recons)], [original, *(x["recon"] for x in recons)])):
        part = image[r * b:(r + 1) * b, c * b:(c + 1) * b]
        zoomed = np.repeat(np.repeat(part, ZOOM, axis=0), ZOOM, axis=1)
        x0 = left + j * (side + gap)
        for row, (shown, y0) in enumerate(((image, top), (zoomed, top + side + gap))):
            box = (x0, y0, side, side)
            ax = pixel_axes(fig, box, canvas)
            grey(ax, shown)
            panels.append({"panel": f"{'upper' if row == 0 else 'lower'} {name}", "unit": unit, "array": shown, "box": box,
                           "scale": k, "marks": row == 0})
            if row == 0:
                ax.add_patch(Rectangle((c * b - 0.5, r * b - 0.5), b, b, fill=False, edgecolor=f["fig7_frame_color"],
                                       linewidth=f["fig7_frame_width_pt"], antialiased=False))
                ax.set_title(titles[j], pad=3)
    data += panel_rows(panels, dpi)
    block, zoom = constant(str(b)), constant(str(ZOOM))
    caption = joined("Fig. 7. The fragment ", ("fragment", cell(unit, "text", e6, "pages[0].unit")), " with the largest MGM (",
                     fragment_cells(e5, unit, "largest")[1], ") before (Original) and after compression by "
                     f"{', '.join(style['codecs'][x]['label'] for x in CODECS[:-1])} and {style['codecs'][CODECS[-1]]['label']}"
                     " at the threshold PCC for T = ", ("T", threshold(e5, T_MAIN)), " dB (upper row) and its ",
                     ("region width", block), " × ", ("region height", block), " pixel region with the largest MGM "
                     "magnified ", ("zoom", zoom), " times by pixel repetition (lower row); above the panels - PCC, "
                     "PSNR-HVS-M in dB and CR.")
    return fig, data, series, dpi, panels, caption


# ---------------------------------------------------------------- writing

def write_csv(path, rows, columns=None):
    with open(path, "w", encoding="utf-8", newline="") as f:
        out = csv.writer(f)
        if columns:
            out.writerow(columns)
        for r in rows:
            out.writerow(r)


def write_table(folder, n, head, rows, notes, index):
    """The CSV and the xlsx of a table and its rows of the index. The xlsx carries no time of the build: openpyxl 3.1.5
    cannot leave created and modified out (None fails to write) and wb.save sets modified to the save time, so the book
    is written by ExcelWriter - the writer wb.save calls - with both at FIXED_DATE, and its zip is written again with
    every entry at FIXED_DATE (owner's decision, cycle 21)."""
    import datetime
    import io
    import zipfile

    from openpyxl import Workbook
    from openpyxl.writer.excel import ExcelWriter

    printed = [list(head)] + [[c["value"] for c in r] for r in rows] + [[c["value"]] for c in notes]
    write_csv(folder / f"table{n}.csv", printed)
    wb = Workbook()
    ws = wb.active
    ws.title = f"Table {n}"
    for r, line_ in enumerate(printed, start=1):
        for c, value in enumerate(line_, start=1):
            ws.cell(row=r, column=c, value=value)
    wb.properties.created = wb.properties.modified = datetime.datetime(*FIXED_DATE)
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        ExcelWriter(wb, archive).save()
    with zipfile.ZipFile(raw) as book, zipfile.ZipFile(folder / f"table{n}.xlsx", "w", zipfile.ZIP_DEFLATED) as out:
        for entry in book.infolist():
            out.writestr(zipfile.ZipInfo(entry.filename, FIXED_DATE), book.read(entry.filename), zipfile.ZIP_DEFLATED)
    for r, line_ in enumerate(rows, start=1):
        for k, c in enumerate(line_):
            index += index_rows(f"Table {n} / row {r} / {head[k]}", c)
    for c in notes:
        index += index_rows(f"Table {n} / note", c)


def build(cache_root, papers_root, paths=None, style_file=STYLE_FILE, force=False, e4_pin=E4_SHA256, log=print):
    """`paths` - the lab paths of run_e4.load_paths (data_root and the coders) for Fig. 1 and Fig. 7."""
    cache_root, out = Path(cache_root), Path(papers_root) / PAPER
    targets = [out / p for p in OUTPUTS]
    existing = [str(p.relative_to(out)) for p in targets if p.exists()]
    if existing and not force:
        raise RunRefusal(f"{len(existing)} files exist ({existing[:3]}...): --force overwrites them")
    for name in (E4_FILE, E5_FILE, E6_FILE, TILES_FILE):
        if not (cache_root / name).is_file():
            raise RunRefusal(f"the run file {cache_root / name} does not exist")
    if sha256_file(cache_root / E4_FILE) != e4_pin:
        raise RunRefusal(f"{E4_FILE}: sha256 differs from the pinned {e4_pin}")
    style = tomllib.loads(Path(style_file).read_text(encoding="utf-8"))
    e4, e5, e6, tiles = (Source(name, cache_root / name) for name in (E4_FILE, E5_FILE, E6_FILE, TILES_FILE))
    paths = paths or {}
    _, picks = fig1_fragments(e5.data["analysis"]["complexity"]["fragments"])
    arrays = load_units(paths, list(dict.fromkeys([FIG1_IMAGE, *(x["unit"] for x in picks), e6.data["pages"][0]["unit"]])))
    work = Path(tempfile.mkdtemp(prefix="p001_figures_", dir=cache_root))
    try:
        unit, original, recons = comparison(e4, e5, e6, arrays, paths, work, log)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    plt, font = setup(style)
    index, captions = [], []
    for n, (fig, data, series, dpi, panels, caption) in enumerate((
            fig1(e5, tiles, arrays, style, plt, "png"), fig2(e4, e5, style, plt), fig3(e4, e5, style, plt),
            fig4(e5, style, plt), fig5(e5, style, plt), fig6(e5, style, plt),
            fig7(e5, e6, unit, original, recons, style, plt)), 1):
        if n == 1:  # Fig. 1 has a layout of its own for the PDF: the three rasters of one height
            pdf = fig1(e5, tiles, arrays, style, plt, "pdf")[0]
            pdf.savefig(out / "figures" / "fig1.pdf", metadata=PDF_METADATA)
            plt.close(pdf)
        else:
            fig.savefig(out / "figures" / f"fig{n}.pdf", metadata=PDF_METADATA)
        fig.savefig(out / "figures" / f"fig{n}.png", dpi=dpi)
        width, height = (round(v * dpi) for v in fig.get_size_inches())
        plt.close(fig)
        check_png(out / "figures" / f"fig{n}.png", panels)
        columns = list(dict.fromkeys(k for row in data for k in row))
        write_csv(out / "figures" / f"fig{n}.csv", [[row.get(k, "") for k in columns] for row in data], columns)
        index += [row for name, c in series for row in index_rows(name, c)]
        captions.append((f"Fig. {n}", caption))
        log(f"  fig{n}.png: {dpi} dpi, {width} x {height} pixels ({width / dpi * 2.54:.2f} x {height / dpi * 2.54:.2f} cm)"
            + (f"; image panels pixel for pixel: {len(panels)} - " + ", ".join(f"{q['panel']} x{q['scale']}" for q in panels)
               if panels else ""))
    for n, (head, rows, notes, title) in enumerate((table1(e4, style), table2(e5, style), table3(e5, style)), 1):
        write_table(out / "tables", n, head, rows, notes, index)
        captions.append((f"Table {n}", title))
    for name, c in captions:
        index += index_rows(f"Caption {name}", c)
    (out / "figures" / "captions.txt").write_text("\n\n".join(c["value"] for _, c in captions) + "\n", encoding="utf-8")
    write_csv(out / "tables" / "numbers_index.csv", [[i[k] for k in INDEX_COLUMNS] for i in index], INDEX_COLUMNS)
    r, c, value, _ = region_of(original)
    log(f"Figures, tables and captions written into {out}: {len(OUTPUTS)} files; numbers_index.csv - {len(index)} rows")
    log(f"Font: {font}{'' if font == style['page']['font'] else ' (the fallback: ' + style['page']['font'] + ' not found)'}")
    log(f"Fig. 1: {FIG1_IMAGE}, " + ", ".join(f"({letter}) {x['unit']} MGM {x['mgm']:.4f}" for letter, x in zip(FIG1_LETTERS, picks)))
    log(f"Fig. 7: {unit}, region - block row {r}, column {c} of {BLOCKS} x {BLOCKS} (MGM {value:.6f}); encoded again and "
        f"equal to E4: {len(recons)} of {len(CODECS)}")
    log(f"Inputs: {E4_FILE} sha256 {e4.sha256}; {E5_FILE} sha256 {e5.sha256}; {E6_FILE} sha256 {e6.sha256}; "
        f"{TILES_FILE} sha256 {tiles.sha256}")
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
