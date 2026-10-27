"""Figures and tables of the paper P001 from the run files of E4 and E5 - one command. Run: python studies/P001_vvc-dental/figures.py   (--force: overwrite files that exist)

Reads <cache_root>/cycles/13/e4_main.json (pinned sha256) and <cache_root>/cycles/17/e5_analysis.json, and the style
figures_style.toml beside this script; writes into <papers_root>/P001_vvc-dental/: figures/fig1..fig3 (.pdf, .png
at the dpi of the style, .csv of the plotted data) and tables/table1..table4 (.xlsx, .csv) and tables/numbers_index.csv
- every table cell and every figure series: the value as printed, the run file (relative to cache_root), its sha256,
the key in the file, the operation and the format (the source of NUMBERS.md). The first composition is the
assistant's proposal of cycle 17; all text is English with a decimal point; PCC names the parameter of compression;
series differ without colour (STUDY.md section 7). A number is never typed by hand: it is read from a run file.
"""

import argparse
import csv
import importlib.util
import json
import math
import sys
import tomllib
from pathlib import Path

import numpy as np

from imstudies.lab.runs import RunRefusal, sha256_file

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
STYLE_FILE = STUDY / "figures_style.toml"
PAPER = "P001_vvc-dental"
E4_FILE, E5_FILE = "cycles/13/e4_main.json", "cycles/17/e5_analysis.json"
E4_SHA256 = "44a1733512fdfbb9d02d210ff81d492b9a5d929ec8d547d84676d47350875f2a"  # A12
SCRIPT = "core/studies/P001_vvc-dental/figures.py"
CODECS = ("vtm", "jpeg", "heif", "avif")
OTHERS = ("jpeg", "heif", "avif")
T_MAIN = 42.0
FIG1_RANK = 77  # the fragment of Fig. 1: the 77th of 154 by ascending MGM, ties - the smaller code
LEVEL_EN = {"пациенты": "patients", "снимки": "whole images", "фрагменты": "fragments"}
PCC_MEANING = {"vtm": "QP", "jpeg": "quality", "heif": "quality", "avif": "quality"}  # STUDY.md section 5
IMPLEMENTATION = {  # STUDY.md section 5, REFERENCE B05-B09, B12: the names; the run file holds versions and sha256
    "vtm": "VTM-23.0, all intra (B05); packed by GPAC MP4Box 26.08-DEV (B09)",
    "jpeg": "Pillow with libjpeg-turbo, baseline (B06)",
    "heif": "pillow-heif with libheif and x265 (B07)",
    "avif": "pillow-heif with libheif and libaom (B12)",
}
CONTAINER = {"vtm": "HEIF (vvc1)", "jpeg": "JFIF", "heif": "HEIF (hvc1)", "avif": "AVIF"}
VERSION_KEY = {"vtm": "vtm_gpac_sha256", "jpeg": "jpeg", "heif": "heif", "avif": "avif_libheif"}
OUTPUTS = [f"figures/fig{k}.{ext}" for k in (1, 2, 3) for ext in ("pdf", "png", "csv")] + \
          [f"tables/table{k}.{ext}" for k in (1, 2, 3, 4) for ext in ("xlsx", "csv")] + ["tables/numbers_index.csv"]
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


def pick(items, **where):
    """The items whose fields equal the given values compared as text (the same rule as the key filters)."""
    return [x for x in items if all(str(x.get(k)) == str(v) for k, v in where.items())]


def where_text(**where):
    return ",".join(f"{k}={v}" for k, v in where.items())


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
    head = ("Codec", "Implementation", "Versions or sha256 of the executables", "PCC", "PCC range (fragments)",
            "Container")
    versions = e4.data["header"]["codec_versions"]
    plan = e4.data["header"]["plan"]
    rows = []
    for c in CODECS:
        v = versions[VERSION_KEY[c]]
        joined = "; ".join(f"{k} {str(x)[:8]}" for k, x in v.items()) if c == "vtm" else \
                 "; ".join(f"{k} {x}" for k, x in v.items())
        values = plan["vtm"]["f"] if c == "vtm" else plan[c]
        path = "header.plan.vtm.f" if c == "vtm" else f"header.plan.{c}"
        rows.append([constant(style["codecs"][c]["label"]), constant(IMPLEMENTATION[c]),
                     cell(joined, "text", e4, f"header.codec_versions.{VERSION_KEY[c]}", "join8" if c == "vtm" else "join"),
                     constant(PCC_MEANING[c]), cell(f"{min(values)} to {max(values)}", "text", e4, path, "range"),
                     constant(CONTAINER[c])])
    return head, rows


def table2(e5, style):
    head = ("Level", "T, dB", "Pair", "n", "Effect r (HL)", "95% CI low", "95% CI high", "p", "p (Holm)",
            "Conclusion (rule of section 6)")
    a = e5.data["analysis"]
    rows = []
    for listname, pairs in (("comparisons", OTHERS), ("whole_comparisons", OTHERS), ("comparisons", ("jpeg_opt",))):
        for t in (40.0, 41.0, 42.0):
            for b in pairs:
                i = next(k for k, c in enumerate(a[listname]) if c["pair"] == f"vtm/{b}" and c["t_db"] == t)
                c, base = a[listname][i], f"analysis.{listname}[{i}]"
                main = c["role"] == "основное"
                pair = "VVC/JPEG (optimised Huffman)" if b == "jpeg_opt" else f"VVC/{style['codecs'][b]['label']}"
                rows.append([cell(LEVEL_EN[c["level"]], "text", e5, f"{base}.level", "en"),
                             cell(c["t_db"], ".0f", e5, f"{base}.t_db"), constant(pair),
                             cell(c["n"], "d", e5, f"{base}.n"), cell(c["ratio"], ".3f", e5, f"{base}.ratio"),
                             cell(c["ratio_low"], ".3f", e5, f"{base}.ratio_low"),
                             cell(c["ratio_high"], ".3f", e5, f"{base}.ratio_high"), cell(c["p"], ".3g", e5, f"{base}.p"),
                             cell(c["p_holm"], ".3g", e5, f"{base}.p_holm") if main else constant("-"),
                             cell(verdict_en(c["verdict"]), "text", e5, f"{base}.verdict", "verdict")])
    return head, rows


def table3(e5, style):
    head = ("Codec", "n patients", "CR median", "CR q25", "CR q75", "n fragments", "PSNR median, dB", "PSNR min, dB",
            "PSNR max, dB", "HaarPSI median", "HaarPSI min", "HaarPSI max")
    a = e5.data["analysis"]
    rows = []
    for c in CODECS:
        pw = dict(codec=c, t_db=T_MAIN)
        uw = dict(level="f", codec=c, t_db=T_MAIN, judged=True)
        crs = [p["median_cr"] for p in pick(a["patients"], **pw)]
        units = pick(a["units"], **uw)
        psnr = [number(u["psnr_db"]) for u in units]
        haar = [u["haarpsi"] for u in units]
        pk, uk = f"analysis.patients[{where_text(**pw)}]", f"analysis.units[{where_text(**uw)}]"
        rows.append([constant(style["codecs"][c]["label"]), cell(len(crs), "d", e5, pk, "count"),
                     cell(percentile(crs, 50), ".2f", e5, f"{pk}.median_cr", "median"),
                     cell(percentile(crs, 25), ".2f", e5, f"{pk}.median_cr", "q25"),
                     cell(percentile(crs, 75), ".2f", e5, f"{pk}.median_cr", "q75"),
                     cell(len(units), "d", e5, uk, "count"),
                     cell(percentile(psnr, 50), ".2f", e5, f"{uk}.psnr_db", "median"),
                     cell(min(psnr), ".2f", e5, f"{uk}.psnr_db", "min"), cell(max(psnr), ".2f", e5, f"{uk}.psnr_db", "max"),
                     cell(percentile(haar, 50), ".4f", e5, f"{uk}.haarpsi", "median"),
                     cell(min(haar), ".4f", e5, f"{uk}.haarpsi", "min"), cell(max(haar), ".4f", e5, f"{uk}.haarpsi", "max")])
    return head, rows


def table4(e5):
    """Two blocks under the name of the point: shares of units at or above T, then per patient the median CR."""
    j = e5.data["analysis"]["jpeg50"]
    rows = [[constant("Point"), cell(name_en(j["name"]), "text", e5, "analysis.jpeg50.name", "name")],
            [constant(x) for x in ("Level", "T, dB", "n", "PSNR-HVS-M >= T", "Share, %")]]
    for i, s in enumerate(j["shares"]):
        base = f"analysis.jpeg50.shares[{i}]"
        rows.append([cell(LEVEL_EN[s["level"]], "text", e5, f"{base}.level", "en"), cell(s["t_db"], ".0f", e5, f"{base}.t_db"),
                     cell(s["n"], "d", e5, f"{base}.n"), cell(s["count"], "d", e5, f"{base}.count"),
                     cell(s["share_pct"], ".1f", e5, f"{base}.share_pct")])
    rows.append([constant(x) for x in ("Patient", "n fragments", "Median CR", "Median CR, optimised Huffman (as IrfanView)")])
    for i, p in enumerate(j["patients"]):
        base = f"analysis.jpeg50.patients[{i}]"
        rows.append([cell(p["image"], "text", e5, f"{base}.image"), cell(p["n"], "d", e5, f"{base}.n"),
                     cell(p["median_cr"], ".2f", e5, f"{base}.median_cr"),
                     cell(p["median_cr_opt"], ".2f", e5, f"{base}.median_cr_opt")])
    crs, opts = [p["median_cr"] for p in j["patients"]], [p["median_cr_opt"] for p in j["patients"]]
    rows.append([constant("Median over patients"), cell(len(crs), "d", e5, "analysis.jpeg50.patients", "count"),
                 cell(percentile(crs, 50), ".2f", e5, "analysis.jpeg50.patients.median_cr", "median"),
                 cell(percentile(opts, 50), ".2f", e5, "analysis.jpeg50.patients.median_cr_opt", "median")])
    return None, rows


# ---------------------------------------------------------------- figures

def setup(style):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = style["page"]
    plt.rcParams.update({"font.family": p["font"], "font.size": p["font_size_pt"], "lines.linewidth": p["line_width_pt"],
                         "lines.markersize": p["marker_size_pt"], "axes.unicode_minus": False})
    return plt


def size(style, key):
    p = style["page"]
    width = p["column_cm"] if style["figures"][key] == "column" else p["double_column_cm"]
    return width / 2.54, p["height_cm"] / 2.54


def fig1(e4, e5, style, plt):
    """PSNR-HVS-M = f(CR) of one fragment, four codecs, the threshold lines and the threshold points at 42 dB."""
    fragments = sorted(e5.data["analysis"]["complexity"]["fragments"], key=lambda f: (f["mgm"], f["unit"]))
    unit = fragments[FIG1_RANK - 1]["unit"]
    fig, ax = plt.subplots(figsize=size(style, "fig1_width"), constrained_layout=True)
    data, series = [], []
    for c in CODECS:
        s = style["codecs"][c]
        rows = sorted((r for r in e4.data["rows"] if r["unit"] == unit and r["codec"] == c and r["status"] == "valid"),
                      key=lambda r: r["cr_file"])
        ax.plot([r["cr_file"] for r in rows], [r["psnr_hvs_m_db"] for r in rows], color=s["color"], linestyle=s["line"],
                marker=s["marker"], markevery=max(1, len(rows) // 12), label=s["label"])
        data += [{"fragment": unit, "codec": c, "pcc": r["param"], "cr": r["cr_file"], "psnr_hvs_m_db": r["psnr_hvs_m_db"]}
                 for r in rows]
        series.append((f"Fig. 1 / {s['label']}", cell(len(rows), "d", e4, f"rows[{where_text(unit=unit, codec=c, status='valid')}]",
                                                       "count")))
    t = style["thresholds"]
    for level in (40, 41, 42):
        ax.axhline(level, color=t["color"], linestyle=t[f"line_{level}"], linewidth=0.8)
    marks = []
    for c in CODECS:
        u = pick(e5.data["analysis"]["units"], unit=unit, codec=c, t_db=T_MAIN, judged=True)[0]
        marks.append((c, u["cr"], u["psnr_hvs_m_db"]))
        s = style["codecs"][c]
        ax.plot([u["cr"]], [u["psnr_hvs_m_db"]], linestyle="none", marker=s["marker"], markerfacecolor="none",
                markeredgecolor=s["color"], markersize=style["figures"]["threshold_marker_size_pt"])
    for row in data:
        row["threshold_42"] = int(any(m[0] == row["codec"] and m[1] == row["cr"] for m in marks))
    series.append(("Fig. 1 / threshold points at 42 dB",
                   cell(len(marks), "d", e5, f"analysis.units[{where_text(unit=unit, t_db=T_MAIN, judged=True)}]", "count")))
    series.append(("Fig. 1 / threshold lines, dB", cell("; ".join(show(x, ".0f") for x in e5.data["thresholds_db"]), "text",
                                                       e5, "thresholds_db", "list")))
    ax.set_xscale("log")
    ax.set_xlabel("CR")
    ax.set_ylabel("PSNR-HVS-M, dB")
    ax.legend(loc=style["page"]["legend"])
    return fig, data, series


def fig2(e5, style, plt):
    """r per patient at 42 dB for the three pairs, the Hodges-Lehmann estimate and its 95 % interval, r = 1."""
    a = e5.data["analysis"]
    fig, ax = plt.subplots(figsize=size(style, "fig2_width"), constrained_layout=True)
    data, series, labels = [], [], []
    for k, b in enumerate(OTHERS, start=1):
        s, pair = style["codecs"][b], f"vtm/{b}"
        points = pick(a["ratios"], pair=pair, t_db=T_MAIN, in_test=True)
        offsets = np.linspace(-0.2, 0.2, len(points)) if len(points) > 1 else np.zeros(1)
        ax.plot(k + offsets, [p["r"] for p in points], linestyle="none", marker=s["marker"], color=s["color"],
                label=f"VVC/{s['label']}")
        i = next(k for k, x in enumerate(a["comparisons"]) if x["pair"] == pair and x["t_db"] == T_MAIN)
        c = a["comparisons"][i]
        ax.errorbar([k + 0.32], [c["ratio"]], yerr=[[c["ratio"] - c["ratio_low"]], [c["ratio_high"] - c["ratio"]]],
                    fmt=s["marker"], color=s["color"], markerfacecolor="none", capsize=3)
        data += [{"pair": pair, "patient": p["image"], "r": p["r"]} for p in points]
        data.append({"pair": pair, "patient": "estimate", "r": c["ratio"], "ci_low": c["ratio_low"],
                     "ci_high": c["ratio_high"]})
        key = f"analysis.comparisons[{i}]"
        series += [(f"Fig. 2 / VVC/{s['label']} / patients",
                    cell(len(points), "d", e5, f"analysis.ratios[{where_text(pair=pair, t_db=T_MAIN, in_test=True)}]", "count")),
                   (f"Fig. 2 / VVC/{s['label']} / estimate", cell(c["ratio"], ".3f", e5, f"{key}.ratio")),
                   (f"Fig. 2 / VVC/{s['label']} / 95% CI low", cell(c["ratio_low"], ".3f", e5, f"{key}.ratio_low")),
                   (f"Fig. 2 / VVC/{s['label']} / 95% CI high", cell(c["ratio_high"], ".3f", e5, f"{key}.ratio_high"))]
        labels.append(f"VVC/{s['label']}")
    ax.axhline(1.0, color=style["thresholds"]["color"], linestyle="dashed", linewidth=0.8)
    ax.set_xticks([1, 2, 3], labels)
    ax.set_ylabel("r = CR(VVC) / CR(codec) at 42 dB")
    return fig, data, series


def fig3(e5, style, plt):
    """MGM against CR at the threshold of 42 dB, 154 fragments, four codecs."""
    fig, ax = plt.subplots(figsize=size(style, "fig3_width"), constrained_layout=True)
    data, series = [], []
    for c in CODECS:
        s = style["codecs"][c]
        units = pick(e5.data["analysis"]["units"], level="f", codec=c, t_db=T_MAIN, judged=True)
        ax.plot([u["mgm"] for u in units], [u["cr"] for u in units], linestyle="none", marker=s["marker"],
                color=s["color"], markerfacecolor="none", label=s["label"])
        data += [{"fragment": u["unit"], "codec": c, "mgm": u["mgm"], "cr": u["cr"]} for u in units]
        series.append((f"Fig. 3 / {s['label']}", cell(len(units), "d", e5, f"analysis.units[{where_text(level='f', codec=c, t_db=T_MAIN, judged=True)}]",
                                                       "count")))
    ax.set_yscale("log")
    ax.set_xlabel("MGM")
    ax.set_ylabel("CR at 42 dB")
    ax.legend(loc=style["page"]["legend"])
    return fig, data, series


# ---------------------------------------------------------------- writing

def write_csv(path, rows, columns=None):
    with open(path, "w", encoding="utf-8", newline="") as f:
        out = csv.writer(f)
        if columns:
            out.writerow(columns)
        for r in rows:
            out.writerow(r)


def write_table(folder, n, head, rows, index):
    from openpyxl import Workbook

    printed = ([list(head)] if head else []) + [[c["value"] for c in r] for r in rows]
    write_csv(folder / f"table{n}.csv", printed)
    wb = Workbook()
    ws = wb.active
    ws.title = f"Table {n}"
    for r, line in enumerate(printed, start=1):
        for c, value in enumerate(line, start=1):
            ws.cell(row=r, column=c, value=value)
    wb.save(folder / f"table{n}.xlsx")
    for r, line in enumerate(rows, start=1):
        for k, c in enumerate(line):
            column = head[k] if head else f"column {k + 1}"
            index.append({"id": f"Table {n} / row {r} / {column}", **c})


def build(cache_root, papers_root, style_file=STYLE_FILE, force=False, e4_pin=E4_SHA256, log=print):
    cache_root, out = Path(cache_root), Path(papers_root) / PAPER
    targets = [out / p for p in OUTPUTS]
    existing = [str(p.relative_to(out)) for p in targets if p.exists()]
    if existing and not force:
        raise RunRefusal(f"{len(existing)} files exist ({existing[:3]}...): --force overwrites them")
    e4_path, e5_path = cache_root / E4_FILE, cache_root / E5_FILE
    for path in (e4_path, e5_path):
        if not path.is_file():
            raise RunRefusal(f"the run file {path} does not exist")
    if sha256_file(e4_path) != e4_pin:
        raise RunRefusal(f"{E4_FILE}: sha256 differs from the pinned {e4_pin}")
    style = tomllib.loads(Path(style_file).read_text(encoding="utf-8"))
    e4, e5 = Source(E4_FILE, e4_path), Source(E5_FILE, e5_path)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    plt = setup(style)
    index = []
    for n, (fig, data, series) in enumerate((fig1(e4, e5, style, plt), fig2(e5, style, plt), fig3(e5, style, plt)), 1):
        fig.savefig(out / "figures" / f"fig{n}.pdf")
        fig.savefig(out / "figures" / f"fig{n}.png", dpi=style["page"]["dpi"])
        plt.close(fig)
        columns = list(dict.fromkeys(k for row in data for k in row))
        write_csv(out / "figures" / f"fig{n}.csv", [[row.get(k, "") for k in columns] for row in data], columns)
        index += [{"id": name, **c} for name, c in series]
    for n, (head, rows) in enumerate((table1(e4, style), table2(e5, style), table3(e5, style), table4(e5)), 1):
        write_table(out / "tables", n, head, rows, index)
    write_csv(out / "tables" / "numbers_index.csv", [[i[k] for k in INDEX_COLUMNS] for i in index], INDEX_COLUMNS)
    log(f"Figures and tables written into {out}: {len(OUTPUTS)} files; numbers_index.csv - {len(index)} rows")
    log(f"Inputs: {E4_FILE} sha256 {e4.sha256}; {E5_FILE} sha256 {e5.sha256}")
    return out, index


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite the figures and tables that exist")
    args = parser.parse_args(argv)
    try:
        spec = importlib.util.spec_from_file_location("run_e4", STUDY / "run_e4.py")
        run_e4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_e4)
        paths = run_e4.load_paths()
        raw = json.loads((ROOT / "lab_paths.json").read_text(encoding="utf-8"))
        if not raw.get("papers_root"):
            raise RunRefusal("lab_paths.json has no key 'papers_root'")
        build(paths["cache_root"], (ROOT / raw["papers_root"]).resolve(), force=args.force)
    except RunRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
