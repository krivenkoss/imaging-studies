"""Addendum E5 of P001: how much CR does each codec lose when one PCC is taken for every unit instead of the threshold PCC of the unit, how do the codecs compare at these values, and how many different files does the PCC scale give? Run: python studies/P001_vvc-dental/run_e5_fixed.py   (--plan: the fixed PCC, the missing VVC points and the control, no coding, no pixels)

Study P001, STUDY.md section 7, «Дополнение E5» (owner's decision 2026-10-30; cycle 26): declared after the numbers of
E5, descriptive, no test of hypotheses; the conclusion of E5 does not change. Inputs, each by its pinned sha256: the run
file of E4 <cache_root>/cycles/13/e4_main.json and the run file of E5 cycles/17/e5_analysis.json, whose header names
the same E4. The fixed PCC of a codec is taken from E5 - the value at which every fragment is not below 42 dB: for VVC
(a larger QP compresses more) the smallest threshold PCC of the fragments, for JPEG, HEIF and AVIF (a larger quality
compresses less) the largest; it must equal the expected value (VVC 25, JPEG 57, HEIF 43, AVIF 66) or the run refuses
in words. The points at the fixed PCC are the E4 rows; a point missing in E4 may only be a VVC point of a whole image
(the grid of step 4 and the refinement at the threshold) - it is encoded with the wrapper and the arguments of E4
(B05 + B09, the timeout of whole images) and its row has the sha256 of the file and of the reconstruction as in E4; any
other missing point is a refusal. Control before the long part: one VVC point of a whole image that E4 has at the fixed
PCC is encoded again and must equal E4 by both sha256. Points go into the journal e5_fixed.points.jsonl as they finish;
the same command does not encode a finished point again. Writes <cache_root>/cycles/26/e5_fixed.json once and the book
e5_fixed.xlsx, every cell checked against the run file. The time of coding is neither kept nor printed (owner,
2026-10-30). Only codes leave the run.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np

import imstudies
from imstudies.lab.runs import RunRefusal, git_state, sha256_file

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
_spec = importlib.util.spec_from_file_location("run_e4", STUDY / "run_e4.py")
run_e4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_e4)
e4_book = run_e4.e4_book

COMMAND = "python studies/P001_vvc-dental/run_e5_fixed.py"
QUESTION = ("Дополнение E5, P001: сколько CR теряет каждый кодек, если вместо порогового PCC своей единицы взять одно "
            "значение PCC для всех единиц, как кодеки соотносятся при этих значениях и сколько различных файлов даёт "
            "шкала PCC?")
CYCLE = 26
RUN_FILE, BOOK_FILE, JOURNAL_FILE, WORK_DIR = "e5_fixed.json", "e5_fixed.xlsx", "e5_fixed.points.jsonl", "e5_fixed_work"
E4_FILE, E5_FILE = "cycles/13/e4_main.json", "cycles/17/e5_analysis.json"
E4_SHA256 = "44a1733512fdfbb9d02d210ff81d492b9a5d929ec8d547d84676d47350875f2a"  # A12
E5_SHA256 = "82b8306c699ec25fab3acdb97b6e058bc39674f2a499163fd50922e52c61d54d"  # A15
T_DB = 42.0
CODECS = ("vtm", "jpeg", "heif", "avif")
OTHERS = ("jpeg", "heif", "avif")
EXPECTED_PCC = {"vtm": 25, "jpeg": 57, "heif": 43, "avif": 66}  # STUDY.md section 7: a check of the value taken from E5
CODERS = ("encoder", "decoder", "cfg", "mp4box", "gpac")  # VTM and GPAC files, sha256 as in the header of E4
TIME_FIELDS = ("seconds", "seconds_b05", "seconds_b09", "seconds_metrics")  # the time of coding is not kept
LEVELS = e4_book.LEVELS
SUMMARY_LEVELS = {"f": "пациенты", "m": "снимки"}  # a patient - the medians over its fragments; a whole image - itself
NAMES = e4_book.CODEC_NAMES
NEW, CONTROL = "новая", "контроль"
DECLARED = {
    "status": ("Описательно, объявлено после чисел E5 и после диагностического расчёта ассистента по E4 (STUDY.md §7, "
               "«Дополнение E5», решение владельца 2026-10-30); проверки гипотез нет; вывод E5 «VVC выгоднее всех» не "
               "меняется. Роль данных — check."),
    "input": ("Файл прогона E4 cycles/13/e4_main.json и файл прогона E5 cycles/17/e5_analysis.json — по закреплённым "
              "sha256; E5 называет в шапке тот же E4. Пиксели копии data/P001 (B10, B11, sha256 сверяются) — только "
              "для кодирования недостающих точек VVC и контроля."),
    "fixed": ("Фиксированный PCC кодека — из записей E5 analysis.units при T = 42 дБ на уровне фрагментов: значение, при "
              "котором все фрагменты не ниже 42 дБ; у VVC (больший QP сжимает сильнее) — наименьший пороговый PCC "
              "фрагментов, у JPEG, HEIF и AVIF (большее quality сжимает слабее) — наибольший. Фрагмент вне E5 или не "
              "судимый при 42 дБ — отказ словами. Ожидаемое (VVC 25, JPEG 57, HEIF 43, AVIF 66) — проверка: "
              "расхождение — отказ словами."),
    "points": ("Точка единицы и кодека при фиксированном PCC — строка E4 (статус valid или lossless). Недостающей может "
               "быть только точка VVC целого снимка (сетка с шагом 4 и уточнение у порога): она кодируется той же "
               "обёрткой и с теми же аргументами, что в E4 (B05 + B09, предел времени целых снимков), исполняемые "
               "файлы VTM и GPAC и cfg — с sha256 из шапки E4, иначе отказ; строка — с sha256 файла и восстановления, "
               "как в E4. Недостающая точка другого кодека или фрагмента — отказ словами."),
    "control": ("Добавлено исполнителем (контроль, способный только ослабить): до новых точек кодируется снова точка VVC "
                "при фиксированном PCC первого по коду целого снимка, у которого она есть в E4; sha256 файла и "
                "восстановления обязаны равняться строке E4, иначе отказ словами."),
    "journal": ("Журнал e5_fixed.points.jsonl: шапка — объявленное, коммит, входы, фиксированные PCC, точки, sha256 "
                "кодеров; строка на законченную точку; та же команда не кодирует готовую точку снова; иная шапка — "
                "отказ словами; точка со сбоем — отказ, та же команда повторит её."),
    "unit": ("На единицу и кодек: PCC и CR при фиксированном PCC, PSNR-HVS-M, PSNR, HaarPSI, bpp той же точки; пороговый "
             "PCC и CR в собственном пороге при 42 дБ (E5); потеря — CR при фиксированном / CR в собственном пороге; "
             "ниже 42 дБ — точка valid с PSNR-HVS-M < 42 дБ (lossless — не ниже)."),
    "patients": ("Пациент — снимок, n = 16: для кодека — медиана (numpy.median) по фрагментам снимка CR при "
                 "фиксированном PCC и CR в собственном пороге, рядом — число фрагментов; потеря пациента — отношение "
                 "этих медиан; r_fix = медиана CR VVC / медиана CR кодека X, X = JPEG, HEIF, AVIF. Медиана CR в "
                 "собственном пороге обязана равняться E5 (patients, 42 дБ), иначе отказ. Целый снимок — то же по "
                 "одной единице."),
    "summary": ("По 16 пациентам и по 16 целым снимкам: потеря по кодекам и r_fix по парам — медиана, квартили "
                "(numpy.percentile, линейная), min, max, n; у r_fix — число единиц с r_fix > 1 из n."),
    "scale": ("Шкала PCC — по всему перебору E4 для JPEG, HEIF и AVIF, фрагменты и снимки: на единицу — число значений "
              "PCC и число различных файлов (sha256 файла); значения PCC, при которых файл равен файлу при PCC − 1 "
              "у всех единиц уровня и у всех единиц."),
    "time": "Время кодирования не хранится и не печатается (решение владельца 2026-10-30: в статье не нужно).",
}


# ---------------------------------------------------------------- the fixed PCC and the points

def fixed_pcc(units, fragments, expected):
    """{codec: fixed PCC} from the E5 records at T_DB on the fragments: VVC - the smallest threshold PCC, JPEG, HEIF,
    AVIF - the largest. A fragment missing or not judged, or a value other than `expected` - a refusal in words."""
    out = {}
    for codec in CODECS:
        mine = {u["unit"]: u for u in units if u["level"] == "f" and u["codec"] == codec and u["t_db"] == T_DB}
        if set(mine) != set(fragments):
            raise RunRefusal(f"{NAMES[codec]}: E5 has {len(mine)} fragments at {T_DB:g} dB, E4 has {len(fragments)} - "
                             "the fixed PCC needs every fragment")
        unjudged = sorted(u for u, x in mine.items() if not x["judged"])
        if unjudged:
            raise RunRefusal(f"{NAMES[codec]}: {len(unjudged)} fragments are not judged at {T_DB:g} dB "
                             f"({unjudged[:3]}...) - no PCC keeps every fragment at the threshold")
        pick = min if codec == "vtm" else max  # VVC: a larger QP compresses more; the others: a larger quality less
        out[codec] = pick(x["pcc"] for x in mine.values())
        if out[codec] != expected[codec]:
            raise RunRefusal(f"{NAMES[codec]}: the fixed PCC by E5 is {out[codec]}, {expected[codec]} is expected "
                             "(STUDY.md section 7) - the run stops")
    return out


def usable(row):
    return row is not None and row["status"] in ("valid", "lossless")


def missing_points(levels, rows, fixed):
    """(unit, codec, PCC) without a usable E4 row at the fixed PCC; only VVC of whole images may be missing, any other
    is a refusal in words."""
    missing = [(u, c, fixed[c]) for u in levels for c in CODECS if not usable(rows.get((u, c, fixed[c])))]
    other = [m for m in missing if not (m[1] == "vtm" and levels[m[0]] == "m")]
    if other:
        raise RunRefusal(f"{len(other)} points at the fixed PCC are missing in E4 outside VVC of whole images: "
                         + ", ".join(f"{u} {NAMES[c]} PCC {p}" for u, c, p in other[:5]))
    return missing


def control_point(levels, rows, fixed):
    """The VVC point at the fixed PCC of the first whole image (by code) that has it in E4."""
    for unit in sorted(u for u, lv in levels.items() if lv == "m"):
        if usable(rows.get((unit, "vtm", fixed["vtm"]))):
            return unit, "vtm", fixed["vtm"]
    raise RunRefusal(f"no whole image has VVC at PCC {fixed['vtm']} in E4 - the control point is missing")


def check_coders(paths, pinned):
    """sha256 of the VTM and GPAC files equal to the header of E4; a missing file or another sha256 - a refusal."""
    out = {}
    for key in CODERS:
        path = Path(paths[key])
        if not path.is_file():
            raise RunRefusal(f"the coder file '{key}' of lab_paths.json does not exist")
        out[key] = sha256_file(path)
        if out[key] != pinned[key]:
            raise RunRefusal(f"the coder file '{key}' has sha256 {out[key]}, E4 was encoded with {pinned[key]}")
    return out


def encode_point(paths, unit, image, qp, work):
    """One VVC point of a whole image with the wrapper and the arguments of E4 (B05 + B09); the row of E4 without the
    time of coding. A module name so that a test can put a coder of its own in its place."""
    row = e4_book.vtm_task(paths, unit, "m", image, qp, work, run_e4.TIMEOUT_S["m"])
    return {k: v for k, v in row.items() if k not in TIME_FIELDS}


# ---------------------------------------------------------------- the tables

def points_table(levels, image_of, at, own, fixed):
    """Unit x codec: the point at the fixed PCC and the own threshold of E5 at T_DB; `at` - {(unit, codec): row}."""
    out = []
    for unit, level in levels.items():
        for codec in CODECS:
            row, mine = at[(unit, codec)], own[(unit, codec)]
            below = row["status"] == "valid" and row["psnr_hvs_m_db"] < T_DB
            cr_own = mine["cr"] if mine["judged"] else None
            out.append({"unit": unit, "level": level, "image": image_of[unit], "codec": codec, "pcc_fix": fixed[codec],
                        "source": row.get("source", "E4"), "status": row["status"], "cr_fix": row["cr_file"],
                        "psnr_hvs_m_db": row["psnr_hvs_m_db"], "psnr_db": row["psnr_db"], "haarpsi": row["haarpsi"],
                        "bpp": row["bpp"], "below_t": below, "pcc_own": mine["pcc"], "cr_own": cr_own,
                        "own_reason": mine["reason"], "loss": row["cr_file"] / cr_own if cr_own else None,
                        "file_sha256": row["file_sha256"], "recon_sha256": row["recon_sha256"]})
    return out


def per_image(points, level, images):
    """Per image (a patient - its fragments; a whole image - itself) and codec: medians of CR at the fixed PCC and at
    the own threshold, the loss as their ratio, n beside."""
    out = []
    for image in images:
        for codec in CODECS:
            mine = [p for p in points if p["level"] == level and p["image"] == image and p["codec"] == codec]
            if not mine:
                raise RunRefusal(f"{image}: no unit of the level {LEVELS[level]} for {NAMES[codec]}")
            own = [p["cr_own"] for p in mine if p["cr_own"] is not None]
            fix = float(np.median([p["cr_fix"] for p in mine]))
            own_median = float(np.median(own)) if own else None
            out.append({"level": SUMMARY_LEVELS[level], "image": image, "codec": codec, "n": len(mine), "own_n": len(own),
                        "cr_fix": fix, "cr_own": own_median, "loss": fix / own_median if own_median else None})
    return out


def ratios(table):
    """r_fix = CR(VVC) / CR(X) at the fixed PCC per image of the table, X = JPEG, HEIF, AVIF."""
    by = {(x["image"], x["codec"]): x for x in table}
    return [{"level": x["level"], "image": x["image"], "pair": f"vtm/{x['codec']}",
             "r_fix": by[(x["image"], "vtm")]["cr_fix"] / x["cr_fix"]} for x in table if x["codec"] != "vtm"]


def spread(values):
    """n, median, quartiles (numpy, linear), min and max; no value - n = 0 and no number."""
    v = np.array(values, dtype=np.float64)
    if not v.size:
        return {"n": 0, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    return {"n": int(v.size), "median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75)), "min": float(v.min()), "max": float(v.max())}


def summary(table, ratio_rows):
    """Per level: the loss by codec and r_fix by pair over the images - median, quartiles, min, max, n; r_fix > 1."""
    out = []
    for level in dict.fromkeys(x["level"] for x in table):
        for codec in CODECS:
            losses = [x["loss"] for x in table if x["level"] == level and x["codec"] == codec and x["loss"] is not None]
            out.append({"level": level, "what": "потеря «фиксированный / свой»", "codec": codec, **spread(losses),
                        "above_1": None})
        for codec in OTHERS:
            rs = [x["r_fix"] for x in ratio_rows if x["level"] == level and x["pair"] == f"vtm/{codec}"]
            out.append({"level": level, "what": "r_fix", "codec": f"vtm/{codec}", **spread(rs),
                        "above_1": sum(r > 1 for r in rs)})
    return out


def below_table(points):
    """Per level and codec: the units below T_DB at the fixed PCC, of how many, and their codes."""
    out = []
    for level in ("f", "m"):
        for codec in CODECS:
            mine = [p for p in points if p["level"] == level and p["codec"] == codec]
            low = sorted(p["unit"] for p in mine if p["below_t"])
            out.append({"level": LEVELS[level], "codec": codec, "n": len(mine), "below": len(low), "units": low,
                        "images": sorted({p["image"] for p in mine if p["below_t"]})})
    return out


def scale(rows, levels):
    """The PCC scale of JPEG, HEIF, AVIF over the whole sweep of E4: per unit the number of PCC values and of different
    files (sha256 of the file) and the PCC at which the file equals the file at PCC - 1; per codec and level (and all
    units) the counts of units by those numbers and the PCC equal to PCC - 1 at every unit."""
    files = {}
    for r in rows:
        if r["codec"] in OTHERS and r["file_sha256"] is not None:
            files.setdefault((r["unit"], r["codec"]), {})[r["param"]] = r["file_sha256"]
    units = []
    for unit in levels:
        for codec in OTHERS:
            f = files.get((unit, codec), {})
            equal = [q for q in sorted(f) if q - 1 in f and f[q] == f[q - 1]]
            units.append({"unit": unit, "level": LEVELS[levels[unit]], "codec": codec, "values": len(f),
                          "files": len(set(f.values())), "equal_previous": equal})
    out = []
    for codec in OTHERS:
        for level in (*LEVELS.values(), "все"):
            mine = [u for u in units if u["codec"] == codec and level in (u["level"], "все")]
            count = lambda key: {str(k): sum(u[key] == k for u in mine) for k in sorted({u[key] for u in mine})}  # noqa: E731
            common = set.intersection(*(set(u["equal_previous"]) for u in mine)) if mine else set()
            out.append({"codec": codec, "level": level, "n": len(mine), "values_per_unit": count("values"),
                        "files_per_unit": count("files"), "equal_previous_all": sorted(common)})
    return units, out


# ---------------------------------------------------------------- the run

def load_pinned(path, pin, what):
    if not Path(path).is_file():
        raise RunRefusal(f"the run file of {what} {path} does not exist")
    sha = sha256_file(path)
    if sha != pin:
        raise RunRefusal(f"the run file of {what} has sha256 {sha}, the addendum is declared for {pin}")
    return json.loads(Path(path).read_text(encoding="utf-8")), sha


def prepare(e4_path, e5_path, pins, expected):
    """The inputs by their sha256, the fixed PCC, the E4 rows by key, the missing points and the control point."""
    e4, e4_sha = load_pinned(e4_path, pins[0], "E4")
    e5, e5_sha = load_pinned(e5_path, pins[1], "E5")
    if e5["inputs"]["e4_run_sha256"] != e4_sha:
        raise RunRefusal(f"the run file of E5 names E4 {e5['inputs']['e4_run_sha256']}, the E4 read has {e4_sha}")
    levels = e4["units"]
    fixed = fixed_pcc(e5["analysis"]["units"], [u for u, lv in levels.items() if lv == "f"], expected)
    rows = {(r["unit"], r["codec"], r["param"]): r for r in e4["rows"]}
    return e4, e4_sha, e5, e5_sha, levels, fixed, rows, missing_points(levels, rows, fixed), control_point(levels, rows, fixed)


def task_of(point, role):
    unit, codec, pcc = point
    return f"{'control|' if role == CONTROL else ''}{unit}|{codec}|{pcc}"


def plan_lines(fixed, expected, missing, control, coders, done):
    out = ["Fixed PCC by E5 (expected): " + ", ".join(f"{NAMES[c]} {fixed[c]} ({expected[c]})" for c in CODECS)]
    out.append(f"Missing VVC points of whole images at PCC {fixed['vtm']}: {len(missing)} - "
               + (", ".join(u for u, _, _ in missing) or "none"))
    out.append(f"Control: {control[0]} VVC PCC {control[2]} encoded again, equal to E4 by sha256 of the file and of the "
               "reconstruction")
    out.append("Coder files VTM and GPAC: sha256 equal to the header of E4 - " + ", ".join(sorted(coders)))
    out.append(f"Journal: {len(done)} of {len(missing) + 1} points finished")
    return out


def encoded_points(paths, arrays_of, journal, missing, control, rows, work, log):
    """The control first, then the missing points; a finished point of the journal is not encoded again. A failed
    point or a control that differs from E4 - a refusal in words (the journal keeps the finished points)."""
    todo = [(control, CONTROL)] + [(m, NEW) for m in missing]
    left = [(p, role) for p, role in todo if task_of(p, role) not in journal.done]
    arrays = arrays_of(sorted({p[0] for p, _ in left})) if left else {}
    for point, role in todo:
        task = task_of(point, role)
        if task in journal.done:
            log(f"  {task}: finished in the journal")
        else:
            unit, _, qp = point
            row = encode_point(paths, unit, np.ascontiguousarray(arrays[unit]), qp, work) | {"role": role}
            journal.append(task, [row])
            if row["status"] == "error":
                raise RunRefusal(f"{unit} VVC PCC {qp}: the coder failed ({row['detail'][:300]}) - the same command "
                                 "encodes it again")
            log(f"  {task}: encoded, {row['status']}")
        (row,) = journal.done[task]
        if role == CONTROL:
            e4 = rows[point]
            equal = (row["file_sha256"] == e4["file_sha256"], row["recon_sha256"] == e4["recon_sha256"])
            if not all(equal):
                raise RunRefusal(f"control {point[0]} VVC PCC {point[2]}: encoded again, the "
                                 f"{'file' if not equal[0] else 'reconstruction'} differs from E4 by sha256 - the "
                                 "new points would not be those of E4")
    out = []
    for point, role in todo:
        (row,) = journal.done[task_of(point, role)]
        extra = {}
        if role == CONTROL:
            e4 = rows[point]
            extra = {"e4_file_sha256": e4["file_sha256"], "e4_recon_sha256": e4["recon_sha256"],
                     "file_equal_e4": row["file_sha256"] == e4["file_sha256"],
                     "recon_equal_e4": row["recon_sha256"] == e4["recon_sha256"]}
        out.append({k: v for k, v in row.items() if k not in TIME_FIELDS} | extra)
    return out


def analyse(e4, e5, levels, fixed, rows, new_rows):
    """Every table of the addendum from the E4 rows, the new points and E5."""
    at = {}
    for (unit, codec, pcc), row in rows.items():
        if pcc == fixed[codec] and usable(row):
            at[(unit, codec)] = row
    for row in new_rows:
        if row["role"] == NEW:
            at[(row["unit"], row["codec"])] = row | {"source": f"цикл {CYCLE}"}
    own = {(u["unit"], u["codec"]): u for u in e5["analysis"]["units"] if u["t_db"] == T_DB}
    image_of = {u: (own[(u, "vtm")]["image"] if lv == "f" else u) for u, lv in levels.items()}
    images = sorted(u for u, lv in levels.items() if lv == "m")
    points = points_table(levels, image_of, at, own, fixed)
    patients = per_image(points, "f", images)
    e5_medians = {(p["image"], p["codec"]): p["median_cr"] for p in e5["analysis"]["patients"] if p["t_db"] == T_DB}
    for p in patients:
        if p["cr_own"] != e5_medians[(p["image"], p["codec"])]:
            raise RunRefusal(f"{p['image']} {NAMES[p['codec']]}: the median CR at the own threshold {p['cr_own']} "
                             f"differs from E5 {e5_medians[(p['image'], p['codec'])]}")
    whole = per_image(points, "m", images)
    ratio_rows = ratios(patients) + ratios(whole)
    scale_units, scale_summary = scale(e4["rows"], levels)
    return {"fixed_pcc": fixed, "points": points, "patients": patients, "whole": whole, "ratios": ratio_rows,
            "summary": summary(patients + whole, ratio_rows), "below": below_table(points), "new_points": new_rows,
            "scale": {"units": scale_units, "summary": scale_summary}}


def run(e4_path, e5_path, out_dir, state, paths, arrays_of, pins=(E4_SHA256, E5_SHA256), expected=EXPECTED_PCC,
        plan_only=False, log=print):
    """--plan: the inputs, the fixed PCC, the missing points, the control and the coders - nothing encoded, no pixel
    read. The run: the points, the run file and the book. `arrays_of(codes)` - {code: uint8 array} of whole images."""
    out_dir = Path(out_dir)
    run_path, book_path, journal_path = out_dir / RUN_FILE, out_dir / BOOK_FILE, out_dir / JOURNAL_FILE
    if not plan_only:
        for path in (run_path, book_path):
            if path.exists():
                raise RunRefusal(f"{path} already exists: the run file and the book of cycle {CYCLE} are never "
                                 "overwritten")
    e4, e4_sha, e5, e5_sha, levels, fixed, rows, missing, control = prepare(e4_path, e5_path, pins, expected)
    coders = check_coders(paths, e4["header"]["codec_versions"]["vtm_gpac_sha256"])
    log(f"{COMMAND}\n{QUESTION}\nDeclared (after the numbers of E5):")
    for key, text in DECLARED.items():
        log(f"  [{key}] {text}")
    inputs = {"e4_run": E4_FILE, "e4_run_sha256": e4_sha, "e5_run": E5_FILE, "e5_run_sha256": e5_sha,
              "registry_sha256": run_e4.REGISTRY_SHA256, "tiles_sha256": run_e4.TILES_SHA256}
    points = [task_of(control, CONTROL)] + [task_of(m, NEW) for m in missing]
    if plan_only:
        done = []
        if journal_path.is_file():
            _, entries, _ = e4_book.read_journal(journal_path)
            done = sorted({e["task"] for e in entries if not e4_book.failed(e["rows"])})
        for line in plan_lines(fixed, expected, missing, control, coders, done):
            log(line)
        return None
    header = {"declared": DECLARED, "core_commit": state["core_commit"], "core_tree": state["core_tree"],
              "inputs": inputs, "fixed_pcc": fixed, "points": points, "coders_sha256": coders,
              "timeout_s": run_e4.TIMEOUT_S["m"]}
    try:
        journal = e4_book.Journal(journal_path, header)
    except e4_book.JournalRefusal as exc:
        raise RunRefusal(str(exc)) from exc
    started = datetime.datetime.now(datetime.timezone.utc)
    for line in plan_lines(fixed, expected, missing, control, coders, sorted(journal.done)):
        log(line)
    work = out_dir / WORK_DIR
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        new_rows = encoded_points(paths, arrays_of, journal, missing, control, rows, work, log)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    analysis = analyse(e4, e5, levels, fixed, rows, new_rows)
    record = {"run": RUN_FILE, "study": "P001", "cycle": CYCLE, "question": QUESTION, "command": COMMAND,
              "started_utc": started.isoformat(timespec="seconds"),
              "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), **state,
              "declared_after_numbers": DECLARED, "t_db": T_DB, "expected_pcc": expected, "inputs": inputs,
              "coders_sha256": coders, "journal": JOURNAL_FILE,
              "environment": {"python": platform.python_version(), "imstudies": imstudies.__version__,
                              "numpy": np.__version__},
              "analysis": analysis}
    record["description"] = description(record)
    record["summary_lines"] = summary_lines(record)
    with open(run_path, "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(e4_book.jsonable(record), ensure_ascii=False, indent=1, allow_nan=False))
    build_book(run_path, book_path)
    for line in record["summary_lines"]:
        log(line)
    log(f"Run file: {run_path} (sha256 {sha256_file(run_path)})\nBook: {book_path} (sha256 {sha256_file(book_path)}), "
        "checked cell by cell")
    return run_path, book_path


def description(record):
    """The lines of «Описание», all from the run file."""
    a = record["analysis"]
    new = [p for p in a["new_points"] if p["role"] == NEW]
    (control,) = [p for p in a["new_points"] if p["role"] == CONTROL]
    lines = [("Вопрос", QUESTION), ("Команда", COMMAND),
             ("Файл прогона", f"{RUN_FILE} (cycles/{CYCLE}/ под cache_root из lab_paths.json)"),
             ("Начат, UTC", record["started_utc"]), ("Закончен, UTC", record["finished_utc"]),
             ("Коммит core; дерево", f"{record['core_commit']}; {record['core_tree']}"),
             ("Вход: E4, sha256", f"{record['inputs']['e4_run']}; {record['inputs']['e4_run_sha256']}"),
             ("Вход: E5, sha256", f"{record['inputs']['e5_run']}; {record['inputs']['e5_run_sha256']}"),
             ("Кодеры VTM и GPAC, sha256 (= шапке E4)", json.dumps(record["coders_sha256"], ensure_ascii=False)),
             ("Версии", json.dumps(record["environment"], ensure_ascii=False)),
             *((f"{k} (объявлено после чисел E5)", v) for k, v in record["declared_after_numbers"].items()),
             ("Фиксированный PCC (ожидалось)", "; ".join(f"{NAMES[c]} {a['fixed_pcc'][c]} ({record['expected_pcc'][c]})"
                                                         for c in CODECS)),
             ("Новые точки VVC, шт; снимки", f"{len(new)}; {', '.join(p['unit'] for p in new) or '—'}"),
             ("Контроль", f"{control['unit']} VVC PCC {control['param']}: файл равен E4 — "
                          f"{'да' if control['file_equal_e4'] else 'нет'}, восстановление — "
                          f"{'да' if control['recon_equal_e4'] else 'нет'}"),
             e4_book.PCC_LINE,
             ("Где что", "«Таблица» — потеря и r_fix по пациентам и по снимкам (медиана, квартили, min, max, n), единицы "
                         "ниже 42 дБ, строки пациентов и снимков; «Единицы» — точка каждой единицы и кодека; «Шкала PCC» "
                         "— различные файлы на единицу; «Новые точки VVC» — закодированные в цикле 26 и контроль"),
             ("На пальцах", "Для каждого кодека выбрано одно значение PCC, при котором все фрагменты не хуже порога, и "
                            "все снимки сжаты при нём; сравнено, насколько это сжатие слабее, чем у подобранного для "
                            "каждой единицы своего значения, и во сколько раз VVC сжимает сильнее других кодеков при "
                            "этих значениях. Это описание, а не проверка гипотезы.")]
    return lines


def summary_lines(record):
    """What the console prints after the numbers, n beside each."""
    a = record["analysis"]
    out = ["Fixed PCC: " + ", ".join(f"{NAMES[c]} {a['fixed_pcc'][c]}" for c in CODECS)]
    for x in a["below"]:
        out.append(f"  below {T_DB:g} dB at the fixed PCC, {x['level']}, {NAMES[x['codec']]}: {x['below']} of {x['n']}")
    four = lambda x: "-" if x is None else f"{x:.4f}"  # noqa: E731
    for s in a["summary"]:
        name = NAMES[s["codec"]] if s["codec"] in NAMES else "VVC / " + NAMES[s["codec"].split("/")[1]]
        out.append(f"  {s['level']}, {s['what']}, {name}: median {four(s['median'])} (min {four(s['min'])}, max "
                   f"{four(s['max'])}), n = {s['n']}" + (f"; r_fix > 1: {s['above_1']} of {s['n']}"
                                                         if s["above_1"] is not None else ""))
    for s in a["scale"]["summary"]:
        out.append(f"  scale {NAMES[s['codec']]}, {s['level']}: different files per unit {s['files_per_unit']} "
                   f"(n = {s['n']}); PCC equal to PCC - 1 at every unit: {len(s['equal_previous_all'])}")
    return out


# ---------------------------------------------------------------- the book

SHEETS = ("Описание", "Таблица", "Единицы", "Шкала PCC", "Новые точки VVC")
SUMMARY_COLUMNS = (("Уровень", "level"), ("Величина", "what"), ("Кодек или пара", "codec"), ("n, шт", "n"),
                   ("Медиана, раз", "median"), ("q25, раз", "q25"), ("q75, раз", "q75"), ("Мин., раз", "min"),
                   ("Макс., раз", "max"), ("r_fix > 1, шт", "above_1"))
BELOW_COLUMNS = (("Уровень", "level"), ("Кодек", "codec"), ("n, шт", "n"), ("Ниже 42 дБ, шт", "below"),
                 ("Единицы ниже", "units"), ("Снимки с единицами ниже", "images"))
IMAGE_COLUMNS = (("Уровень", "level"), ("Снимок", "image"), ("Кодек", "codec"), ("Единиц, шт", "n"),
                 ("Собственный порог судим, шт", "own_n"), ("CR при фиксированном PCC, безразм.", "cr_fix"),
                 ("CR в собственном пороге, безразм.", "cr_own"), ("Потеря, раз", "loss"))
RATIO_COLUMNS = (("Уровень", "level"), ("Снимок", "image"), ("Пара", "pair"), ("r_fix, раз", "r_fix"))
POINT_COLUMNS = (("Единица", "unit"), ("Уровень", "level"), ("Снимок", "image"), ("Кодек", "codec"),
                 ("Фиксированный PCC", "pcc_fix"), ("Источник точки", "source"), ("Статус", "status"),
                 ("CR при фиксированном PCC, безразм.", "cr_fix"), ("PSNR-HVS-M, дБ", "psnr_hvs_m_db"),
                 ("PSNR, дБ", "psnr_db"), ("HaarPSI, безразм.", "haarpsi"), ("bpp, бит/пикс.", "bpp"),
                 ("Ниже 42 дБ", "below_t"), ("Пороговый PCC (E5, 42 дБ)", "pcc_own"),
                 ("CR в собственном пороге, безразм.", "cr_own"), ("Почему порог не судится", "own_reason"),
                 ("Потеря, раз", "loss"), ("sha256 файла", "file_sha256"), ("sha256 восстановления", "recon_sha256"))
SCALE_COLUMNS = (("Кодек", "codec"), ("Уровень", "level"), ("Единиц, шт", "n"),
                 ("Значений PCC на единицу: единиц, шт", "values_per_unit"),
                 ("Различных файлов на единицу: единиц, шт", "files_per_unit"),
                 ("PCC, при которых файл = файлу при PCC − 1 у всех единиц", "equal_previous_all"))
SCALE_UNIT_COLUMNS = (("Единица", "unit"), ("Уровень", "level"), ("Кодек", "codec"), ("Значений PCC, шт", "values"),
                      ("Различных файлов, шт", "files"), ("PCC, при которых файл = файлу при PCC − 1", "equal_previous"))
NEW_COLUMNS = (("Роль", "role"), ("Единица", "unit"), ("Кодек", "codec"), ("PCC", "param"), ("Статус", "status"),
               ("Файл, байт", "file_bytes"), ("CR, безразм.", "cr_file"), ("bpp, бит/пикс.", "bpp"),
               ("PSNR, дБ", "psnr_db"), ("PSNR-HVS-M, дБ", "psnr_hvs_m_db"), ("HaarPSI, безразм.", "haarpsi"),
               ("sha256 файла", "file_sha256"), ("sha256 восстановления", "recon_sha256"),
               ("sha256 потока", "bitstream_sha256"), ("sha256 файла E4", "e4_file_sha256"),
               ("sha256 восстановления E4", "e4_recon_sha256"), ("Файл = E4", "file_equal_e4"),
               ("Восстановление = E4", "recon_equal_e4"))


def shown(value):
    """A value of the run file as a cell: a list or a dict as text, the rest as it is."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "—"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value


def put_table(cells, sheet, top, columns, records):
    """A table from column A: the header row, the key row of the run file under it, then the records; the next free
    row after a gap."""
    for j, (head, key) in enumerate(columns, start=1):
        cells[sheet, top, j], cells[sheet, top + 1, j] = head, key
    for i, rec in enumerate(records):
        for j, (_, key) in enumerate(columns, start=1):
            value = shown(rec.get(key))
            if value is not None:
                cells[sheet, top + 2 + i, j] = value
    return top + 2 + len(records) + 1


def book_cells(record, run_sha256):
    """All cells of the book, {(sheet, row, column): value}."""
    a, cells = record["analysis"], {}
    for i, (k, v) in enumerate([("sha256 файла прогона", run_sha256), *record["description"]], start=1):
        cells["Описание", i, 1], cells["Описание", i, 2] = k, v
    top = 1
    for title, columns, records in (("Потеря и r_fix по пациентам и по целым снимкам", SUMMARY_COLUMNS, a["summary"]),
                                    ("Единицы ниже 42 дБ при фиксированном PCC", BELOW_COLUMNS, a["below"]),
                                    ("Пациенты: медианы по фрагментам", IMAGE_COLUMNS, a["patients"]),
                                    ("Целые снимки", IMAGE_COLUMNS, a["whole"]),
                                    ("r_fix по пациентам и по целым снимкам", RATIO_COLUMNS, a["ratios"])):
        cells["Таблица", top, 1] = title
        top = put_table(cells, "Таблица", top + 1, columns, records)
    put_table(cells, "Единицы", 1, POINT_COLUMNS, a["points"])
    top = put_table(cells, "Шкала PCC", 1, SCALE_COLUMNS, a["scale"]["summary"])
    put_table(cells, "Шкала PCC", top, SCALE_UNIT_COLUMNS, a["scale"]["units"])
    put_table(cells, "Новые точки VVC", 1, NEW_COLUMNS, a["new_points"])
    return cells


def build_book(run_path, book_path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    run_bytes = Path(run_path).read_bytes()
    cells = book_cells(json.loads(run_bytes.decode("utf-8")), hashlib.sha256(run_bytes).hexdigest())
    wb = Workbook()
    wb.remove(wb.active)
    for name in SHEETS:
        wb.create_sheet(name)
    for (sheet, r, c), v in cells.items():
        wb[sheet].cell(row=r, column=c, value=v)
    wb["Описание"].column_dimensions["A"].width = 44
    wb["Описание"].column_dimensions["B"].width = 130
    for row in wb["Описание"].iter_rows(min_col=2, max_col=2):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    for name in SHEETS[1:]:
        for c in wb[name][1]:
            c.font = Font(bold=True)
    wb.save(book_path)
    check_book(run_path, book_path)


def check_book(run_path, book_path):
    """Every cell equals what the run file gives; no other cell is filled."""
    from openpyxl import load_workbook

    run_bytes = Path(run_path).read_bytes()
    cells = book_cells(json.loads(run_bytes.decode("utf-8")), hashlib.sha256(run_bytes).hexdigest())
    book = load_workbook(book_path)
    for (sheet, r, c), v in cells.items():
        got = book[sheet].cell(row=r, column=c).value
        expected = float(f"{v:.16g}") if isinstance(v, float) else v
        if got != expected:
            raise RunRefusal(f"book cell {sheet}!R{r}C{c} = {got!r} differs from the run file value {v!r}")
    for sheet in book.sheetnames:
        extra = {(sheet, x.row, x.column) for row in book[sheet].iter_rows() for x in row if x.value is not None}
        if extra - set(cells):
            raise RunRefusal(f"book sheet {sheet} has cells not taken from the run file: {sorted(extra - set(cells))[:3]}")


def whole_arrays(paths):
    """{code: uint8 array} of whole images of the copy data/P001 (B10, B11 by their pinned sha256)."""
    def arrays_of(codes):
        registry, tile_rows, tile = run_e4.load_inputs(paths["data_root"], Path(paths["cache_root"]) / "cycles" / "9" /
                                                       "p001_tiles.json")
        units = run_e4.load_units(paths["data_root"], registry, tile_rows, tile)
        return {code: units[code]["array"] for code in codes}
    return arrays_of


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", action="store_true", help="the fixed PCC, the missing points and the control only")
    args = parser.parse_args(argv)
    try:
        paths = run_e4.load_paths()
        cycles = paths["cache_root"] / "cycles"
        state = None if args.plan else git_state(ROOT)
        run(paths["cache_root"] / E4_FILE, paths["cache_root"] / E5_FILE, cycles / str(CYCLE), state, paths,
            whole_arrays(paths), plan_only=args.plan)
    except RunRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
