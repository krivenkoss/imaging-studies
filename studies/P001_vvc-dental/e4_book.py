"""E4 book of study P001 and the resume journal: shared by the dry run (cycle 11) and the main run E4.

Contract: STUDY.md section 7 ("Книга E4") and ../../REFERENCE.md#lab-experiment (chain, point 3; sheets
«Просмотр» and «Обзор»). The book holds the rows of the run file («Строки»), overview data per level, codec
and parameter («Обзор-данные»: median and quartiles over the units of the level, n), lists for the drop-down
choices («Списки»), and two sheets of native Excel charts driven by INDEX/MATCH formulas: «Просмотр» (one unit
chosen from a list: metric = f(parameter), metric = f(CR), CR = f(parameter), metric against metric) and «Обзор»
(level and metric chosen from lists: median and quartiles). No dynamic arrays - the book opens in Excel 2016;
full recalculation on load. Every static cell is checked against the run file; formula cells are checked as
text. Lossless rows (MSE = 0) keep their values in «Строки» but give no chart point (NA) and are counted under
the charts. Shared lab code lives here until a second study needs it (then src/, by its own contract).
"""

import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path

import numpy as np

CODECS = ("vtm", "jpeg", "heif", "avif")
CODEC_NAMES = {"vtm": "VVC (VTM)", "jpeg": "JPEG", "heif": "HEIF", "avif": "AVIF"}
PARAM_NAMES = {codec: "PCC" for codec in CODECS}  # parameter controlled compression (STUDY.md section 5)
PCC_LINE = ("PCC", "VVC (VTM) — QP; JPEG, HEIF, AVIF — quality")
# The one style table (STUDY.md section 7, "Вид рядов"): colour (RGB hex), line style (DrawingML preset), marker.
# AVIF #CC79A7 - owner decision 2026-10-21 (cycle 12): the declared #E69F00 was 67 from JPEG in RGB.
CODEC_STYLE = {"vtm": ("0072B2", "solid", "circle"), "jpeg": ("D55E00", "dash", "square"),
               "heif": ("000000", "sysDot", "triangle"), "avif": ("CC79A7", "dashDot", "diamond")}
METRIC_STYLE = {"psnr_db": ("0072B2", "solid", "circle"), "psnr_hvs_m_db": ("D55E00", "dash", "square"),
                "haarpsi": ("000000", "sysDot", "triangle")}
QUARTILE_DASH = {"медиана": "solid", "q25": "dash", "q75": "sysDot"}  # colour and marker of the codec
THRESHOLD_STYLE = {40: ("7F7F7F", "sysDot", "none"), 41: ("7F7F7F", "dash", "none"), 42: ("7F7F7F", "solid", "none")}


class StyleRefusal(Exception):
    """Two series of one chart coincide in colour and line style (owner decision 2026-10-21: colour AND line)."""
LEVELS = {"f": "фрагменты", "m": "снимки"}
METRICS = (("psnr_db", "PSNR", "PSNR, дБ"), ("psnr_hvs_m_db", "PSNR-HVS-M", "PSNR-HVS-M, дБ"),
           ("haarpsi", "HaarPSI", "HaarPSI, безразм."))
QUANTITIES = (*((key, name) for key, name, _ in METRICS), ("cr_file", "CR"))  # overview columns
STATS = (("median", "медиана"), ("q25", "q25"), ("q75", "q75"))
THRESHOLDS_DB = (40, 41, 42)
ROWS = (("Ключ «единица|кодек|PCC»", "key"), ("Единица", "unit"), ("Уровень", "level"), ("Кодек", "codec"),
        ("PCC", "param"), ("Статус", "status"), ("Файл, байт", "file_bytes"), ("CR, безразм.", "cr_file"),
        ("bpp, бит/пикс.", "bpp"), ("PSNR, дБ", "psnr_db"), ("PSNR-HVS-M, дБ", "psnr_hvs_m_db"),
        ("HaarPSI, безразм.", "haarpsi"), ("Кодер, с", "seconds"), ("Метрики, с", "seconds_metrics"),
        ("sha256 файла", "file_sha256"), ("Причина сбоя", "detail"))
ROW_COL = {key: chr(ord("A") + i) for i, (_, key) in enumerate(ROWS)}


# ---------------------------------------------------------------- resume journal (LAB-EXPERIMENT, chain, 3)

class JournalRefusal(Exception):
    """The journal cannot be continued; the message says why in words."""


def _canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


class Journal:
    """Append-only journal: the first line is the header (declared, plan, commit, versions, images); then one line
    per finished task {"task": id, "rows": [...]}. A run started again with an equal header skips the finished
    tasks; another header is refused in words naming what differs. A task whose rows are all `error` (a failed
    call, also one cut by a stop - Ctrl+C reaches the coder too) is not finished: it runs again, and its last
    entry is the one that counts (`last`). A torn last line (a hard stop while writing) is ignored and counted."""

    def __init__(self, path, header):
        self.path, self.header, self.lock = Path(path), header, threading.Lock()
        self.done, self.last, self.entries, self.torn_lines = {}, {}, [], 0
        if self.path.is_file():
            old, entries, torn = read_journal(self.path)
            if _canonical(old) != _canonical(header):
                differ = sorted(k for k in set(old) | set(header) if _canonical(old.get(k)) != _canonical(header.get(k)))
                inner = {k: sorted(s for s in set(old.get(k) or {}) | set(header.get(k) or {})
                                   if _canonical((old.get(k) or {}).get(s)) != _canonical((header.get(k) or {}).get(s)))
                         for k in differ if isinstance(old.get(k), dict) and isinstance(header.get(k), dict)}
                raise JournalRefusal(f"the journal {self.path.name} was started with another header: differs in "
                                     f"{differ} {inner} - finish or remove it by hand, or restore the same setup")
            self.entries, self.torn_lines = entries, torn
            for entry in entries:
                self.last[entry["task"]] = entry["rows"]
            if self.torn_lines:  # rewrite without the torn line, so the next appends start on a fresh line
                lines = self.path.read_text(encoding="utf-8").split("\n")
                good = [lines[0]] + [ln for ln in lines[1:] if ln.strip() and _loads_ok(ln)]
                self.path.write_text("\n".join(good) + "\n", encoding="utf-8")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "x", encoding="utf-8", newline="\n") as f:
                f.write(_canonical(header) + "\n")
        self.done = {task: rows for task, rows in self.last.items() if not failed(rows)}
        self.resumed, self.to_retry = len(self.done), len(self.last) - len(self.done)

    def append(self, task, rows, **extra):
        """One finished task: its rows (and extra fields, e.g. the wall time of a finished pool)."""
        line = json.dumps({"task": task, "rows": rows, **extra}, ensure_ascii=False, allow_nan=False)
        with self.lock:
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.last[task] = rows
            if failed(rows):
                self.done.pop(task, None)
            else:
                self.done[task] = rows
            self.entries.append({"task": task, "rows": rows, **extra})


def read_journal(path):
    """Header, entries and the number of torn lines of a journal, read only: also a look from outside (--status)
    while a session may be writing its last line."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    entries, torn = [], 0
    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            torn += 1
    return json.loads(lines[0]), entries, torn


def failed(rows):
    """A task is not finished when it has rows and every one of them is an error."""
    return bool(rows) and all(r.get("status") == "error" for r in rows)


def _loads_ok(line):
    try:
        json.loads(line)
        return True
    except json.JSONDecodeError:
        return False


# ---------------------------------------------------------------- overview data

def overview(rows):
    """Per (level, codec, param): n valid units, n lossless, and median, q25, q75 of each quantity over the valid
    units of that level only (numpy.percentile, linear)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["level"], r["codec"], r["param"]), []).append(r)
    out = []
    for (level, codec, param), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], CODECS.index(kv[0][1]), kv[0][2])):
        valid = [r for r in rs if r["status"] == "valid"]
        entry = {"key": f"{LEVELS[level]}|{codec}|{param}", "level": LEVELS[level], "codec": codec, "param": param,
                 "n": len(valid), "n_lossless": sum(r["status"] == "lossless" for r in rs)}
        for key, name in QUANTITIES:
            values = np.array([r[key] for r in valid], dtype=np.float64)
            for stat, label in STATS:
                q = {"median": 50, "q25": 25, "q75": 75}[stat]
                entry[f"{name}|{label}"] = float(np.percentile(values, q)) if values.size else None
        out.append(entry)
    return out


OVERVIEW_COLUMNS = (("Ключ «уровень|кодек|PCC»", "key"), ("Уровень", "level"), ("Кодек", "codec"),
                    ("PCC", "param"), ("n единиц, шт", "n"), ("Без потерь, шт", "n_lossless"),
                    *((f"{name} {label}", f"{name}|{label}") for _, name in QUANTITIES for _, label in STATS))


# ---------------------------------------------------------------- the book

def _col(n):
    """1-based column number -> letters."""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _q(sheet):
    return "'" + sheet + "'"


def params_of(rows, codec):
    return sorted({r["param"] for r in rows if r["codec"] == codec})


def view_formula(codec, param_cell, quantity, last):
    """«Просмотр»: the value of `quantity` of the unit in $B$1 for (codec, param), NA unless the row is valid."""
    rng = lambda key: f"{_q('Строки')}!${ROW_COL[key]}$3:${ROW_COL[key]}${last}"
    match = f'MATCH($B$1&"|{codec}|"&{param_cell},{rng("key")},0)'
    return f'=IFERROR(IF(INDEX({rng("status")},{match})="valid",INDEX({rng(quantity)},{match}),NA()),NA())'


def overview_formula(codec, param_cell, column_key, last, n_cols):
    """«Обзор»: a column of «Обзор-данные» for (level in $B$1, codec, param); column_key may name the metric of $B$2."""
    first, end = _col(5), _col(n_cols)
    table = f"{_q('Обзор-данные')}!${first}$3:${end}${last}"
    header = f"{_q('Обзор-данные')}!${first}$2:${end}$2"
    keys = f"{_q('Обзор-данные')}!$A$3:$A${last}"
    return (f'=IFERROR(INDEX({table},MATCH($B$1&"|{codec}|"&{param_cell},{keys},0),'
            f'MATCH({column_key},{header},0)),NA())')


def _table_cells(cells, sheet, columns, records):
    for j, (head, key) in enumerate(columns, start=1):
        cells[sheet, 1, j], cells[sheet, 2, j] = head, key
    for i, rec in enumerate(records):
        for j, (_, key) in enumerate(columns, start=1):
            cells[sheet, 3 + i, j] = rec.get(key)


VIEW_QUANTITIES = (("psnr_db", "PSNR, дБ"), ("psnr_hvs_m_db", "PSNR-HVS-M, дБ"), ("haarpsi", "HaarPSI, безразм."),
                   ("cr_file", "CR, безразм."))
VIEW_BLOCK = 6  # param + 4 quantities + gap
OVER_BLOCK = 9  # param, median, q25, q75, n, CR median, CR q25, CR q75, gap
TABLE_ROW = 8   # first data row of the tables on «Просмотр» and «Обзор» (row 7 - headers)


def view_layout(rows):
    """Columns of the codec blocks and the data rows of «Просмотр»: {codec: (first column, [params])}."""
    return {codec: (1 + k * VIEW_BLOCK, params_of(rows, codec)) for k, codec in enumerate(CODECS)}


def over_layout(rows):
    return {codec: (1 + k * OVER_BLOCK, params_of(rows, codec)) for k, codec in enumerate(CODECS)}


def book_cells(record, run_sha256, describe):
    """All cells of the book: static values and formula texts, {(sheet, row, column): value}."""
    cells = {}
    rows = record["rows"]
    lines = [("Команда", record["command"]), ("Вопрос", record["question"]),
             ("Файл прогона", f"{record['run']} (cycles/{record['cycle']}/ под cache_root из lab_paths.json)"),
             ("sha256 файла прогона", run_sha256), PCC_LINE, *describe(record)]
    for i, (k, v) in enumerate(lines, start=1):
        cells["Описание", i, 1], cells["Описание", i, 2] = k, v
    for i, (k, v) in enumerate(record["summary_lines"], start=1):
        cells["Сводка", i, 1], cells["Сводка", i, 2] = k, v
    _table_cells(cells, "Строки", ROWS, rows)
    over = record["overview"]
    _table_cells(cells, "Обзор-данные", OVERVIEW_COLUMNS, over)
    units = list(record["units"])
    cells["Списки", 1, 1], cells["Списки", 1, 2], cells["Списки", 1, 3] = "Единица", "Уровень", "Метрика"
    for i, u in enumerate(units, start=2):
        cells["Списки", i, 1] = u
    for i, level in enumerate(LEVELS.values(), start=2):
        cells["Списки", i, 2] = level
    for i, (_, name, _) in enumerate(METRICS, start=2):
        cells["Списки", i, 3] = name
    last_row, last_over = 2 + len(rows), 2 + len(over)
    # «Просмотр»
    cells["Просмотр", 1, 1], cells["Просмотр", 1, 2] = "Единица (выбор из списка):", units[0]
    cells["Просмотр", 2, 1] = "Одна единица анализа; точки без потерь (MSE = 0) на графиках не показываются."
    cells["Просмотр", 3, 1] = "Без потерь, точек нет:"
    for k, codec in enumerate(CODECS):
        cells["Просмотр", 3, 2 + k] = (f'="{CODEC_NAMES[codec]}: "&COUNTIFS({_q("Строки")}!$B$3:$B${last_row},$B$1,'
                                       f'{_q("Строки")}!$D$3:$D${last_row},"{codec}",'
                                       f'{_q("Строки")}!$F$3:$F${last_row},"lossless")')
    for codec, (c0, params) in view_layout(rows).items():
        cells["Просмотр", TABLE_ROW - 2, c0] = CODEC_NAMES[codec]
        cells["Просмотр", TABLE_ROW - 1, c0] = PARAM_NAMES[codec]
        for j, (_, head) in enumerate(VIEW_QUANTITIES, start=1):
            cells["Просмотр", TABLE_ROW - 1, c0 + j] = head
        for i, p in enumerate(params):
            r = TABLE_ROW + i
            cells["Просмотр", r, c0] = p
            for j, (quantity, _) in enumerate(VIEW_QUANTITIES, start=1):
                cells["Просмотр", r, c0 + j] = view_formula(codec, f"${_col(c0)}{r}", quantity, last_row)
    # threshold lines of PSNR-HVS-M = f(CR): x from the smallest to the largest CR of the unit
    cr_ranges = [f"${_col(c0 + 4)}${TABLE_ROW}:${_col(c0 + 4)}${TABLE_ROW + len(params) - 1}"
                 for c0, params in view_layout(rows).values()]
    cells["Просмотр", 4, 1], cells["Просмотр", 5, 1] = "Пороги PSNR-HVS-M: CR от", "CR до"
    cells["Просмотр", 4, 2] = f"=MIN({','.join(cr_ranges)})"
    cells["Просмотр", 5, 2] = f"=MAX({','.join(cr_ranges)})"
    for k, t in enumerate(THRESHOLDS_DB):
        cells["Просмотр", 4, 3 + k], cells["Просмотр", 5, 3 + k] = t, t
    # «Обзор»
    cells["Обзор", 1, 1], cells["Обзор", 1, 2] = "Уровень (выбор из списка):", LEVELS["f"]
    cells["Обзор", 2, 1], cells["Обзор", 2, 2] = "Метрика (выбор из списка):", METRICS[1][1]
    cells["Обзор", 3, 1] = "Обзор, не результат: медиана и квартили по единицам выбранного уровня при каждом значении параметра; n — рядом."
    n_cols = len(OVERVIEW_COLUMNS)
    heads = ("медиана", "q25", "q75", "n единиц, шт", "CR медиана", "CR q25", "CR q75")
    keys = ('$B$2&"|медиана"', '$B$2&"|q25"', '$B$2&"|q75"', '"n"', '"CR|медиана"', '"CR|q25"', '"CR|q75"')
    for codec, (c0, params) in over_layout(rows).items():
        cells["Обзор", TABLE_ROW - 2, c0] = CODEC_NAMES[codec]
        cells["Обзор", TABLE_ROW - 1, c0] = PARAM_NAMES[codec]
        for j, head in enumerate(heads, start=1):
            cells["Обзор", TABLE_ROW - 1, c0 + j] = head
        for i, p in enumerate(params):
            r = TABLE_ROW + i
            cells["Обзор", r, c0] = p
            for j, key in enumerate(keys, start=1):
                cells["Обзор", r, c0 + j] = overview_formula(codec, f"${_col(c0)}{r}", key, last_over, n_cols)
    return cells


def _scatter(title, x_title, y_title, log_x=False, log_y=False):
    from openpyxl.chart import ScatterChart

    chart = ScatterChart()
    chart.title, chart.style = title, 13
    chart.x_axis.title, chart.y_axis.title = x_title, y_title
    chart.x_axis.delete = chart.y_axis.delete = False
    if log_x:
        chart.x_axis.scaling.logBase = 10
    if log_y:
        chart.y_axis.scaling.logBase = 10
    chart.width, chart.height = 16, 8
    return chart


def _series(ws, x_col, y_col, first, last, title, style):
    """A scatter series with its explicit colour, line style and marker from the style table."""
    from openpyxl.chart import Reference, Series
    from openpyxl.chart.marker import Marker
    from openpyxl.chart.shapes import GraphicalProperties
    from openpyxl.drawing.line import LineProperties

    color, dash, marker = style
    s = Series(Reference(ws, min_col=y_col, min_row=first, max_row=last),
               Reference(ws, min_col=x_col, min_row=first, max_row=last), title=title)
    s.graphicalProperties = GraphicalProperties(ln=LineProperties(solidFill=color, prstDash=dash, w=19050))
    s.marker = Marker(symbol=marker, size=5)
    if marker != "none":
        s.marker.graphicalProperties = GraphicalProperties(solidFill=color, ln=LineProperties(solidFill=color))
    s.smooth = False
    return s


def series_style(series):
    """(colour, line style, marker) written on a series."""
    ln = series.graphicalProperties.line
    color = getattr(ln.solidFill, "srgbClr", ln.solidFill)
    color = getattr(color, "val", color)  # a string when built, an RGB object when read back
    return color, ln.prstDash, series.marker.symbol or "none"  # the reader turns symbol "none" into None


def check_styles(chart):
    """Refuse a chart where two series coincide in colour and line style; returns the chart."""
    seen = {}
    for part in getattr(chart, "_charts", None) or [chart]:  # _charts of a combined chart holds the chart itself
        for s in part.series:
            color, dash, _ = series_style(s)
            title = s.tx.strRef.f if s.tx and s.tx.strRef else (s.tx.v if s.tx else "?")
            if (color, dash) in seen:
                name = chart.title.tx.rich.p[0].r[0].t if chart.title else "?"
                raise StyleRefusal(f"chart '{name}': series '{seen[color, dash]}' and '{title}' coincide in colour "
                                   f"#{color} and line style {dash}")
            seen[color, dash] = title
    return chart


def add_view_charts(ws, rows):
    """13 charts of «Просмотр», anchored right of the tables in a grid of two columns."""
    layout = view_layout(rows)
    anchor_col = 1 + len(CODECS) * VIEW_BLOCK + 1
    charts = []
    for codec, (c0, params) in layout.items():  # metric = f(parameter): PSNR, PSNR-HVS-M left, HaarPSI right
        first, last = TABLE_ROW, TABLE_ROW + len(params) - 1
        chart = _scatter(f"{CODEC_NAMES[codec]}: метрики = f({PARAM_NAMES[codec]})", PARAM_NAMES[codec], "PSNR, PSNR-HVS-M, дБ")
        chart.series.append(_series(ws, c0, c0 + 1, first, last, "PSNR", METRIC_STYLE["psnr_db"]))
        chart.series.append(_series(ws, c0, c0 + 2, first, last, "PSNR-HVS-M", METRIC_STYLE["psnr_hvs_m_db"]))
        right = _scatter("", PARAM_NAMES[codec], "HaarPSI, безразм.")
        right.series.append(_series(ws, c0, c0 + 3, first, last, "HaarPSI", METRIC_STYLE["haarpsi"]))
        right.y_axis.axId, right.y_axis.crosses = 200, "max"
        chart += right
        charts.append(chart)
    for m, (key, name, unit_title) in enumerate(METRICS):  # metric = f(CR), four codecs
        chart = _scatter(f"{name} = f(CR), четыре кодека", "CR, безразм. (лог. шкала)", unit_title, log_x=True)
        for codec, (c0, params) in layout.items():
            chart.series.append(_series(ws, c0 + 4, c0 + 1 + m, TABLE_ROW, TABLE_ROW + len(params) - 1, CODEC_NAMES[codec],
                                        CODEC_STYLE[codec]))
        if key == "psnr_hvs_m_db":
            for k, t in enumerate(THRESHOLDS_DB):
                chart.series.append(_series(ws, 2, 3 + k, 4, 5, f"порог {t} дБ", THRESHOLD_STYLE[t]))
        charts.append(chart)
    for codec, (c0, params) in layout.items():  # CR = f(parameter)
        chart = _scatter(f"{CODEC_NAMES[codec]}: CR = f({PARAM_NAMES[codec]})", PARAM_NAMES[codec],
                         "CR, безразм. (лог. шкала)", log_y=True)
        chart.series.append(_series(ws, c0, c0 + 4, TABLE_ROW, TABLE_ROW + len(params) - 1, "CR", CODEC_STYLE[codec]))
        charts.append(chart)
    for x_m, y_m in ((0, 1), (1, 2)):  # metric against metric
        chart = _scatter(f"{METRICS[y_m][1]} против {METRICS[x_m][1]}", METRICS[x_m][2], METRICS[y_m][2])
        for codec, (c0, params) in layout.items():
            chart.series.append(_series(ws, c0 + 1 + x_m, c0 + 1 + y_m, TABLE_ROW, TABLE_ROW + len(params) - 1,
                                        CODEC_NAMES[codec], CODEC_STYLE[codec]))
        charts.append(chart)
    for k, chart in enumerate(charts):
        ws.add_chart(check_styles(chart), f"{_col(anchor_col + (k % 2) * 11)}{2 + (k // 2) * 18}")
    return charts


def add_overview_charts(ws, rows):
    """9 charts of «Обзор»."""
    layout = over_layout(rows)
    anchor_col = 1 + len(CODECS) * OVER_BLOCK + 1
    charts = []
    for codec, (c0, params) in layout.items():  # selected metric = f(parameter): median and quartiles
        first, last = TABLE_ROW, TABLE_ROW + len(params) - 1
        chart = _scatter(f"{CODEC_NAMES[codec]}: выбранная метрика = f({PARAM_NAMES[codec]}), медиана и квартили",
                         PARAM_NAMES[codec], "выбранная метрика (дБ или безразм.)")
        color, _, marker = CODEC_STYLE[codec]
        for j, title in ((1, "медиана"), (2, "q25"), (3, "q75")):
            chart.series.append(_series(ws, c0, c0 + j, first, last, title, (color, QUARTILE_DASH[title], marker)))
        charts.append(chart)
    chart = _scatter("Медиана метрики против медианы CR, четыре кодека", "медиана CR, безразм. (лог. шкала)",
                     "медиана выбранной метрики", log_x=True)
    for codec, (c0, params) in layout.items():
        chart.series.append(_series(ws, c0 + 5, c0 + 1, TABLE_ROW, TABLE_ROW + len(params) - 1, CODEC_NAMES[codec],
                                    CODEC_STYLE[codec]))
    charts.append(chart)
    for codec, (c0, params) in layout.items():  # CR = f(parameter): median and quartiles
        first, last = TABLE_ROW, TABLE_ROW + len(params) - 1
        chart = _scatter(f"{CODEC_NAMES[codec]}: CR = f({PARAM_NAMES[codec]}), медиана и квартили", PARAM_NAMES[codec],
                         "CR, безразм. (лог. шкала)", log_y=True)
        color, _, marker = CODEC_STYLE[codec]
        for j, title in ((5, "медиана"), (6, "q25"), (7, "q75")):
            chart.series.append(_series(ws, c0, c0 + j, first, last, title, (color, QUARTILE_DASH[title], marker)))
        charts.append(chart)
    for k, chart in enumerate(charts):
        ws.add_chart(check_styles(chart), f"{_col(anchor_col + (k % 2) * 11)}{2 + (k // 2) * 18}")
    return charts


SHEETS = ("Описание", "Сводка", "Просмотр", "Обзор", "Строки", "Обзор-данные", "Списки")


def build_book(run_path, book_path, describe, check):
    """Build the book from the run file and check it (`check` - a function (run_path, book_path))."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    run_bytes = Path(run_path).read_bytes()
    record = json.loads(run_bytes.decode("utf-8"))
    cells = book_cells(record, hashlib.sha256(run_bytes).hexdigest(), describe)
    wb = Workbook()
    wb.remove(wb.active)
    for name in SHEETS:
        wb.create_sheet(name)
    for (sheet, r, c), v in cells.items():
        wb[sheet].cell(row=r, column=c, value=v)
    n_units = len(record["units"])
    for sheet, cell, rng in (("Просмотр", "B1", f"$A$2:$A${1 + n_units}"), ("Обзор", "B1", f"$B$2:$B${1 + len(LEVELS)}"),
                             ("Обзор", "B2", f"$C$2:$C${1 + len(METRICS)}")):
        dv = DataValidation(type="list", formula1=f"{_q('Списки')}!{rng}", allow_blank=False)
        wb[sheet].add_data_validation(dv)
        dv.add(cell)
        wb[sheet][cell].font = Font(bold=True)
    add_view_charts(wb["Просмотр"], record["rows"])
    add_overview_charts(wb["Обзор"], record["rows"])
    for name in ("Описание", "Сводка"):
        wb[name].column_dimensions["A"].width = 44
        wb[name].column_dimensions["B"].width = 130
        for row in wb[name].iter_rows(min_col=2, max_col=2):
            row[0].alignment = Alignment(wrap_text=True, vertical="top")
    for name in ("Строки", "Обзор-данные"):
        for c in wb[name][1]:
            c.font = Font(bold=True)
        wb[name].freeze_panes = "B3"
    wb.calculation.fullCalcOnLoad = True
    wb.save(book_path)
    check(run_path, book_path)


def check_book(run_path, book_path, describe, refusal):
    """Every cell (values and formula texts) equals what the run file gives; no other cell is filled."""
    from openpyxl import load_workbook

    run_bytes = Path(run_path).read_bytes()
    cells = book_cells(json.loads(run_bytes.decode("utf-8")), hashlib.sha256(run_bytes).hexdigest(), describe)
    book = load_workbook(book_path)
    for (sheet, r, c), v in cells.items():
        got = book[sheet].cell(row=r, column=c).value
        expected = float(f"{v:.16g}") if isinstance(v, float) else v
        if got != expected:
            raise refusal(f"book cell {sheet}!R{r}C{c} = {got!r} differs from the run file value {v!r}")
    for sheet in book.sheetnames:
        extra = {(sheet, x.row, x.column) for row in book[sheet].iter_rows() for x in row if x.value is not None}
        if extra - set(cells):
            raise refusal(f"book sheet {sheet} has cells not taken from the run file: {sorted(extra - set(cells))[:3]}")


# ---------------------------------------------------------------- the engine shared by E4 (moved from run_e4_dry.py;
# run_e4_dry.py keeps its own copies as the record of cycle 11)

LEVEL_VTM, FRAME_RATE_VTM = "5", 1  # B05


def jsonable(obj):
    """Non-finite floats as the strings '+inf', '-inf', 'nan' (as json_number), recursively."""
    import math

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else ("+inf" if obj > 0 else ("-inf" if obj < 0 else "nan"))
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def measure(original, recon):
    """PSNR, PSNR-HVS-M, HaarPSI (B02-B04) against the original 8-bit image and their seconds."""
    import time

    from imstudies.core import metrics

    t0 = time.perf_counter()
    out = {"psnr_db": metrics.psnr(original, recon, max_pixels=original.size).value,
           "psnr_hvs_m_db": metrics.psnr_hvs_m(original, recon, max_pixels=original.size).value,
           "haarpsi": metrics.haarpsi(original, recon, max_pixels=original.size).value}
    out["seconds_metrics"] = time.perf_counter() - t0
    return out


def make_row(unit, level, codec, param, image, file, recon, seconds, **extra):
    """One file: size, CR by file, bpp, metrics, sha256 of the file and of the decoding; lossless if recon = image."""
    h, w = image.shape
    row = {"key": f"{unit}|{codec}|{param}", "unit": unit, "level": level, "codec": codec, "param": param,
           "file_bytes": len(file), "cr_file": h * w / len(file), "bpp": 8 * len(file) / (h * w), "seconds": seconds,
           "file_sha256": hashlib.sha256(file).hexdigest(),
           "recon_sha256": hashlib.sha256(np.ascontiguousarray(recon).tobytes()).hexdigest(), "detail": None, **extra}
    row |= measure(image, recon)
    row["status"] = "lossless" if np.array_equal(image, recon) else "valid"
    return jsonable(row)


def failure_detail(exc):
    """The cause of a failure: the exception and, for a program that failed, the tails of its stdout and stderr kept
    by run_tool (2000 characters each) - the message of a CalledProcessError alone names only the command and code."""
    text = f"{type(exc).__name__}: {str(exc)[:400]}"
    if isinstance(exc, subprocess.CalledProcessError):
        for name, tail in (("stdout", exc.output), ("stderr", exc.stderr)):
            if isinstance(tail, bytes):
                tail = tail.decode("utf-8", errors="replace")
            if tail:
                text += f"\n{name}: {tail}"
    return text


def error_row(unit, level, codec, param, exc, **extra):
    return {"key": f"{unit}|{codec}|{param}", "unit": unit, "level": level, "codec": codec, "param": param,
            "status": "error", "detail": failure_detail(exc), "file_bytes": None, "cr_file": None,
            "bpp": None, "psnr_db": None, "psnr_hvs_m_db": None, "haarpsi": None, "seconds": None,
            "seconds_metrics": None, "file_sha256": None, "recon_sha256": None, **extra}


def vtm_task(paths, unit, level, image, qp, work, timeout_s):
    """B05 then B09 for one QP; a failure is an error row."""
    import time

    from imstudies.lab.coders import vtm, vvc_heif

    try:
        t0 = time.perf_counter()
        coded = vtm.encode_decode(image, qp, max_pixels=image.size, level=LEVEL_VTM, frame_rate=FRAME_RATE_VTM,
                                  timeout_s=timeout_s, encoder=paths["encoder"], decoder=paths["decoder"],
                                  cfg=paths["cfg"], work_root=work)
        t1 = time.perf_counter()
        packed = vvc_heif.pack(coded.bitstream, coded.recon, timeout_s=timeout_s, mp4box=paths["mp4box"],
                               gpac=paths["gpac"], decoder=paths["decoder"], work_root=work)
        t2 = time.perf_counter()
        return make_row(unit, level, "vtm", qp, image, packed.file, coded.recon, t2 - t0, seconds_b05=t1 - t0,
                        seconds_b09=t2 - t1, bitstream_sha256=hashlib.sha256(coded.bitstream).hexdigest())
    except Exception as exc:  # noqa: BLE001 - status `error`, the run goes on
        return error_row(unit, level, "vtm", qp, exc)


def still_rows(paths, unit, level, image, codec, values, work, timeout_s):
    """JPEG, HEIF (one call per value) or AVIF (B12, one batch call) at every value; failures are error rows."""
    import time

    from imstudies.lab.coders import avif_libheif, heif, jpeg

    if codec == "avif":
        try:
            t0 = time.perf_counter()
            coded = avif_libheif.encode_decode(image, list(values), max_pixels=image.size, timeout_s=timeout_s,
                                               python=paths["avif_libheif_python"], work_root=work)
            each = (time.perf_counter() - t0) / len(values)
            return [make_row(unit, level, "avif", q, image, c.file, c.recon, each) for q, c in zip(values, coded)]
        except Exception as exc:  # noqa: BLE001
            return [error_row(unit, level, "avif", q, exc) for q in values]
    module = {"jpeg": jpeg, "heif": heif}[codec]
    rows = []
    for q in values:
        try:
            t0 = time.perf_counter()
            c = module.encode_decode(image, q, max_pixels=image.size)
            rows.append(make_row(unit, level, codec, q, image, c.file, c.recon, time.perf_counter() - t0))
        except Exception as exc:  # noqa: BLE001
            rows.append(error_row(unit, level, codec, q, exc))
    return rows


class MemoryProbe:
    """RSS of this process and all children every 0.5 s: peak total and peak of one EncoderApp, bytes."""

    def __init__(self):
        self.peak_total = self.peak_encoder = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        try:
            import psutil
        except ImportError:
            return
        me = psutil.Process(os.getpid())
        while not self._stop.is_set():
            total = 0
            try:
                procs = [me, *me.children(recursive=True)]
            except psutil.Error:
                procs = [me]
            for p in procs:
                try:
                    rss = p.memory_info().rss
                    total += rss
                    if p.name().lower().startswith("encoderapp"):
                        self.peak_encoder = max(self.peak_encoder, rss)
                except psutil.Error:
                    continue
            self.peak_total = max(self.peak_total, total)
            self._stop.wait(0.5)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def interp(points, x):
    """Linear interpolation of sorted (x, y) points; outside them the nearest end value."""
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if x1 <= x <= x2:
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    raise ValueError(x)


def refinement(grid, threshold=42.0, step=4):
    """Refinement QPs of a whole image after its grid (STUDY.md sections 5-7): the first crossing of PSNR-HVS-M =
    threshold from the high-quality side (low QP) between nodes a and a + step gives QP a + 1 ... a + step - 1.
    `grid` - (QP, PSNR-HVS-M) of the nodes, '+inf' for lossless. Returns (qps, None) or ([], reason)."""
    import math

    pts = sorted((q, math.inf if v == "+inf" else float(v)) for q, v in grid)
    for (a, va), (b, vb) in zip(pts, pts[1:]):
        if va >= threshold > vb:
            if b - a != step:
                raise ValueError(f"grid nodes {a} and {b} are not {step} apart")
            return list(range(a + 1, b)), None
    if all(v >= threshold for _, v in pts):
        return [], f"no_crossing: every node is at or above {threshold:g} dB"
    if all(v < threshold for _, v in pts):
        return [], f"no_crossing: every node is below {threshold:g} dB"
    return [], f"no_crossing: no fall below {threshold:g} dB from the high-quality side"


def machine():
    import platform

    out = {"processor": platform.processor(), "logical_cpus": os.cpu_count(), "platform": platform.platform()}
    try:
        import psutil
        out |= {"physical_cpus": psutil.cpu_count(logical=False),
                "memory_total_gib": round(psutil.virtual_memory().total / 2 ** 30, 1)}
    except ImportError:
        pass
    return out
