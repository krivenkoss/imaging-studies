"""E4 of P001: at which PCC do VVC, JPEG, HEIF and AVIF reach the PSNR-HVS-M threshold on 154 fragments and 16 whole radiographs, and with what CR - the main run? Run: python studies/P001_vvc-dental/run_e4.py --hours 10   (one session; the same command continues; --status: progress from the journal; --plan: the plan from metadata, no pixels)

Study P001, contract E4 of STUDY.md section 7, search of section 5. Units: the 16 whole images of the copy
data/P001 (B10) and the 154 fragments of the centred grid (B11 0.2.0, cycles/9/p001_tiles.json). Whole images:
VTM grid QP -12, -8, ..., 60 and, after the grid of an image, 3 QP between the nodes around the first crossing of
PSNR-HVS-M = 42 dB from the high-quality side (no crossing - no_crossing with the reason); fragments: VTM QP
-12..63; JPEG, HEIF, AVIF - every value on both levels. A pool of 4 coder processes, slow tasks first by the
times of cycle 11, whole images before fragments; rows go to the journal e4_main.partial.jsonl as tasks finish;
the same command continues it (another commit or header is refused). A failed task runs again once in the same
session, right after the failure, and then at the next session. Sessions (LAB-EXPERIMENT, chain, 3): `--hours H`
or the empty file STOP beside the journal stop a session by plan - no new task is given, the started ones are
finished into the journal, the journal is copied to e4_main.partial.session-<k>.jsonl, exit code 0; Ctrl+C or a
killed process loses only the started tasks. The output of a session is copied to e4_main.session-<k>.log, with a
progress line every 10 min. The session that leaves no task unfinished writes <cache_root>/cycles/<CYCLE>/e4_main.json
once and the book e4_main.xlsx (e4_book.py). Only codes - no file names, no pixels - leave the run. `--status` and
`--plan` read the journal, the registry and the grid only. `--assemble` (STUDY.md section 7, «Сборка из журнала»;
cycle 14) writes the run file and the book from the journal of cycle 13 (pinned sha256) when a task failed at every
attempt: no task is run, no coder called, no pixel read; such a task gives its error rows with the number of attempts.
"""

import argparse
import collections
import concurrent.futures
import datetime
import importlib.util
import json
import os
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import imstudies
from imstudies.lab import p001_tiles
from imstudies.lab.coders import avif_libheif, heif, jpeg
from imstudies.lab.runs import RunRefusal, git_state, sha256_file

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
_spec = importlib.util.spec_from_file_location("e4_book", STUDY / "e4_book.py")
e4_book = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e4_book)

COMMAND = "python studies/P001_vvc-dental/run_e4.py"
QUESTION = ("E4, P001: при каком PCC кодеки VVC, JPEG, HEIF и AVIF дают на 154 фрагментах и 16 целых снимках "
            "PSNR-HVS-M порога и какой при этом CR — основной прогон?")
CYCLE = 13  # the cycle that runs E4 (confirmed by the task of cycle 13)
RUN_FILE, BOOK_FILE, JOURNAL_FILE, WORK_DIR = "e4_main.json", "e4_main.xlsx", "e4_main.partial.jsonl", "e4_main_work"
STOP_FLAG = "STOP"  # an empty file beside the journal: the session stops by plan
SESSION_LOG, SESSION_COPY = "e4_main.session-{k}.log", "e4_main.partial.session-{k}.jsonl"
STOPPED = "сессия остановлена по плану — та же команда продолжит"
POLL_S = 5.0  # the deadline and the flag are looked at between tasks and at least this often
PROGRESS_S = 600  # a progress line every 10 min
FRESH_S = 900  # --status: a session runs while its journal changed within 15 min (a progress line goes there every 10)
INCOMPLETE = "grid incomplete: a node is missing or failed - waits for the next session"
REGISTRY_SHA256 = "03affc6c150cb3f5beaf2a6c99c1f55861aa877666d77a9e20cc382e30d8f949"  # data/registry/P001.json, cycle 8
TILES_SHA256 = "aad8e80ccb5b972ef3ce7d9f22225f34c15e56ce08dc0247efbdf0af8fcc10a1"  # cycles/9/p001_tiles.json
CYCLE11_SHA256 = "a0365d9766cae3813d56b7a0c2f822cb9fce4c8bb11a4e11a37c0d3bf16979d9"  # cycles/11/e4_dry.json
JOURNAL13_SHA256 = "4af635e950df1040171920edeb6eb07bad4e76a2a1991fa332e3bf911ad1ece5"  # cycles/13 journal after session 7
ASSEMBLED = "собран из журнала: незаконченных задач {n}"
SESSION_END_RU = {"stopped by plan (deadline)": "остановлена по сроку", "stopped by plan (flag)": "остановлена флагом",
                  "ended with tasks left": "закончена, остались задачи", "finished the run": "закончила прогон",
                  "cut off": "оборвана — записи конца нет"}
TIMEOUT_S = {"f": 600, "m": 3600}
POOL = 4
FULL_PLAN = {
    "vtm": {"f": list(range(-12, 64)), "m_grid": list(range(-12, 61, 4)), "refine_step": 4, "threshold_db": 42.0},
    "jpeg": list(range(1, 101)), "heif": list(range(0, 101)), "avif": list(range(0, 101)),
    "tile": 512, "pool": POOL, "controls": {"fragment_qps": [22, 37], "whole_qp": 40},
}
DECLARED = {
    "units": ("16 целых снимков D01…D16 копии data/P001 (B10, реестр sha256 сверяется) — уровень «снимки»; 154 фрагмента "
              "сетки по центру (B11 0.2.0, cycles/9/p001_tiles.json, sha256 сверяется) — уровень «фрагменты»; роль check; "
              "исключений нет. Прогон читает только data/P001, реестр и сетку; в выходах — только коды."),
    "search": ("PCC (STUDY.md §5): фрагменты — VTM QP −12…63; снимки — сетка QP −12, −8, …, 60 и после сетки снимка 3 QP "
               "a + 1…a + 3 между узлами a и a + 4 первого пересечения PSNR-HVS-M = 42 дБ со стороны высокого качества "
               "(§6; без потерь — выше порога); нет пересечения — уточнения нет, no_crossing с причиной; сетка с "
               "незаконченной или со сбойной задачей — уточнение ждёт следующей сессии. JPEG 1…100, HEIF 0…100, AVIF "
               "0…100 (B12, пакет) — оба уровня полностью."),
    "per_file": ("Размер, CR по файлу, bpp, PSNR, PSNR-HVS-M, HaarPSI, секунды; sha256 файла и восстановления; статусы "
                 "valid, lossless, error."),
    "order": ("Пул из 4 процессов кодера. Этап 1 — все задачи снимков (сетка VTM, JPEG, HEIF, AVIF), этап 2 — уточнение "
              "снимков и все задачи фрагментов; внутри этапа — снимки раньше фрагментов, медленные первыми: ожидаемые "
              "секунды по циклу 11 (фрагменты — медиана шести изображений по значению PCC, снимки — мозаика, VTM — "
              "интерполяция по QP, вне — ближайшая; плюс медиана секунд метрик уровня), при равенстве — по кодеку, PCC, "
              "единице. JPEG и HEIF единицы — одна задача на все значения."),
    "resume": ("Журнал e4_main.partial.jsonl: шапка — объявленное, план, коммит, версии кодеков, sha256 реестра и сетки; "
               "строка на законченную задачу; та же команда продолжает; иной коммит или иная шапка — отказ словами; "
               "задача со всеми строками error выполняется снова — в той же сессии один раз, сразу после сбоя, затем "
               "при следующей сессии; в файл прогона идёт её последняя запись. Файл прогона и книгу пишет сессия, после "
               "которой не осталось ни незаконченной задачи, ни снимка без законченной сетки."),
    "sessions": ("Сессии по ≈ 10 ч (LAB-EXPERIMENT, цепочка, п. 3): --hours H — через H ч от начала сессии новые задачи "
                 "не выдаются; файл-флаг STOP рядом с журналом — то же в любой момент (проверяется между задачами и не "
                 "реже раза в 5 с, затем переименовывается в STOP.done-<k>); начатые задачи доводятся до журнала, "
                 "журнал копируется в e4_main.partial.session-<k>.jsonl, выход с кодом 0. Ctrl+C и снятие процесса "
                 "теряют только начатые задачи. Вывод сессии — и в e4_main.session-<k>.log; строка прогресса — каждые "
                 "10 мин и в конце сессии; --status — та же строка по журналу, без запуска задач."),
    "control": "Детерминизм: первый фрагмент QP 22 и 37 и первый снимок QP 40 повторяются последовательно — sha256 = пулу.",
    "memory": "Опрос каждые 0,5 с (psutil): сумма RSS прогона и дочерних, RSS одного EncoderApp — по этапам, ГиБ.",
    "book": "Книга по STUDY.md §7 («Книга E4», «Вид рядов»): PCC, таблица стиля, «Просмотр», «Обзор»; страниц нет (E6).",
}


class RunStopped(Exception):
    """The run was stopped (Ctrl+C or the test stop); the journal holds every finished task."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def load_paths():
    path = ROOT / "lab_paths.json"
    if not path.is_file():
        raise RunRefusal(f"{path} not found: copy lab_paths.example.json to lab_paths.json and set the paths")
    raw = json.loads(path.read_text(encoding="utf-8"))
    paths = {}
    for key in ("data_root", "cache_root", "avif_libheif_python"):
        if not raw.get(key) or raw[key].startswith("<"):
            raise RunRefusal(f"lab_paths.json has no key '{key}' or it is the example placeholder")
        paths[key] = (ROOT / raw[key]).resolve()
    for codec, keys in (("vtm", ("encoder", "decoder", "cfg")), ("gpac", ("mp4box", "gpac"))):
        for key in keys:
            value = (raw.get("coders", {}).get(codec) or {}).get(key)
            if not value:
                raise RunRefusal(f"lab_paths.json has no key coders.{codec}.{key}")
            paths[key] = (ROOT / value).resolve()
    return paths


def load_inputs(data_root, tiles_path, pins=(REGISTRY_SHA256, TILES_SHA256), copies=True):
    """Registry and grid (metadata only): pinned sha256, every copy present (unless `copies` is false - the assembly
    does not touch data/P001), tiles inside their image and as many as the registry says. No pixel is read here."""
    data_root, tiles_path = Path(data_root), Path(tiles_path)
    registry_path = data_root / "registry" / "P001.json"
    for what, path, pin in (("the registry data/registry/P001.json", registry_path, pins[0]),
                            ("the grid cycles/9/p001_tiles.json", tiles_path, pins[1])):
        if not path.is_file():
            raise RunRefusal(f"{what} does not exist: E4 needs the copy of B10 and the grid of B11")
        if pin is not None and sha256_file(path) != pin:
            raise RunRefusal(f"{what}: sha256 differs from the pinned {pin}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    tiles = json.loads(tiles_path.read_text(encoding="utf-8"))
    rows = tiles["rows"]
    tile = tiles.get("parameters", {}).get("tile", 512)
    by_code = {e["code"]: e for e in registry["images"]}
    for code, entry in by_code.items():
        if copies and not (data_root / "P001" / f"{code}.png").is_file():
            raise RunRefusal(f"{code}: the copy data/P001/{code}.png does not exist")
        mine = [r for r in rows if r["code"] == code]
        if len(mine) != entry["full_tiles"]:
            raise RunRefusal(f"{code}: {len(mine)} tiles in the grid, the registry says {entry['full_tiles']}")
        for r in mine:
            if not (0 <= r["x"] and r["x"] + tile <= entry["W"] and 0 <= r["y"] and r["y"] + tile <= entry["H"]):
                raise RunRefusal(f"{r['tile']}: the tile lies outside its image")
    return registry, rows, tile


def unit_levels(registry, tile_rows):
    """Units and their levels from metadata, in the order of load_units: whole images by code, then fragments."""
    levels = {e["code"]: "m" for e in sorted(registry["images"], key=lambda e: e["code"])}
    return levels | {r["tile"]: "f" for r in sorted(tile_rows, key=lambda r: r["tile"])}


def load_units(data_root, registry, tile_rows, tile):
    """Pixels of the copy (E4 run only): whole images first, then the fragments as views."""
    _, images = p001_tiles.load_copy(data_root)
    arrays = {img["code"]: img["values"] for img in images}
    units = {code: {"level": "m", "array": arrays[code]} for code in sorted(arrays)}
    for r in sorted(tile_rows, key=lambda r: r["tile"]):
        units[r["tile"]] = {"level": "f", "array": arrays[r["code"]][r["y"]:r["y"] + tile, r["x"]:r["x"] + tile],
                            "image": r["code"]}
    return units


def timing_from(cycle11):
    """Expected seconds from the dry run: per codec and PCC on fragments (median of six images) and on the mosaic."""
    rows = [r for r in cycle11["rows"] if r["status"] != "error" and r["seconds"] is not None]
    out = {"f": {}, "m": {}, "metrics": {}}
    for level in ("f", "m"):
        mine = [r for r in rows if r["level"] == level]
        for codec in e4_book.CODECS:
            per = {}
            for r in mine:
                if r["codec"] == codec:
                    per.setdefault(r["param"], []).append(r["seconds"])
            out[level][codec] = sorted((p, statistics.median(v)) for p, v in per.items())
        out["metrics"][level] = statistics.median(r["seconds_metrics"] for r in mine)
    crossing = cycle11["summary"]["estimate"]["variants"]["B"]["refine_qps"]
    return out | {"k": cycle11["summary"]["k"], "refine_qps": crossing}


def expected_seconds(timing, level, codec, params):
    """Expected seconds of a task: codec (interpolated by PCC, nearest outside) plus metrics, summed over params."""
    points = timing[level][codec]
    return sum(e4_book.interp(points, p) + timing["metrics"][level] for p in params)


def vtm_task(paths, unit, level, image, qp, work):
    """One VTM task (B05 + B09); a module name so that tests can put a coder of their own in its place."""
    return e4_book.vtm_task(paths, unit, level, image, qp, work, TIMEOUT_S[level])


def still_task(paths, unit, level, image, codec, values, work):
    return e4_book.still_rows(paths, unit, level, image, codec, values, work, TIMEOUT_S[level])


def task_id(unit, codec, param):
    return f"{unit}|{codec}|{param}"


def entry_kind(task):
    """What a journal entry is: a task of the plan, a control task, or a note of the run (pool, session, progress)."""
    head = task.split("|", 1)[0]
    return head if head in ("pool", "session", "progress", "control") else "task"


def order_key(task):
    """Whole images before fragments; slow first; then codec, PCC, unit."""
    return (task["level"] != "m", -task["expected"], e4_book.CODECS.index(task["codec"]), str(task["param"]), task["unit"])


def make_task(level, unit, codec, param, phase, plan, timing):
    params = [param] if codec == "vtm" else plan[codec]
    return {"id": task_id(unit, codec, param), "unit": unit, "level": level, "codec": codec, "param": param,
            "kind": phase, "expected": expected_seconds(timing, level, codec, params)}


def stage_tasks(levels, level, plan, timing):
    """Tasks of whole images (level m: the VTM grid, JPEG, HEIF, AVIF) or of fragments (f: VTM, JPEG, HEIF, AVIF);
    JPEG, HEIF and AVIF of a unit are one task over all their values."""
    qps, phase = (plan["vtm"]["m_grid"], "grid") if level == "m" else (plan["vtm"]["f"], "fragment")
    tasks = []
    for unit, lv in levels.items():
        if lv == level:
            tasks += [make_task(level, unit, "vtm", qp, phase, plan, timing) for qp in qps]
            tasks += [make_task(level, unit, c, "batch", "still", plan, timing) for c in ("jpeg", "heif", "avif")]
    return tasks


def refinements(levels, plan, last):
    """Per whole image: refinement QPs after a complete grid, or the reason why there are none (yet)."""
    out, v = {}, plan["vtm"]
    for unit, level in levels.items():
        if level != "m":
            continue
        grid = [last.get(task_id(unit, "vtm", qp)) for qp in v["m_grid"]]
        if any(g is None or g[0]["status"] == "error" for g in grid):
            out[unit] = {"qps": [], "reason": INCOMPLETE}
            continue
        qps, reason = e4_book.refinement([(g[0]["param"], g[0]["psnr_hvs_m_db"]) for g in grid], v["threshold_db"],
                                         v["refine_step"])
        out[unit] = {"qps": qps, "reason": reason}
    return out


def refine_tasks(refine, plan, timing):
    return [make_task("m", unit, "vtm", qp, "refine", plan, timing) for unit, ref in refine.items() for qp in ref["qps"]]


def control_ids(levels, plan):
    """Determinism control: the first fragment at two QPs and the first whole image at one, one after another."""
    c = plan["controls"]
    fragment = min((u for u, lv in levels.items() if lv == "f"), default=None)
    whole = min((u for u, lv in levels.items() if lv == "m"), default=None)
    pairs = ([(fragment, q) for q in c["fragment_qps"]] if fragment else []) + ([(whole, c["whole_qp"])] if whole else [])
    return [(f"control|{unit}|vtm|{qp}", unit, qp) for unit, qp in pairs]


def planned(levels, plan, timing, last):
    """Every task of the run as far as it is known now; an image whose grid is not finished counts the
    refine_step - 1 QPs of the dry run (cycle 11) as its refinement."""
    refine = refinements(levels, plan, last)
    guess = timing["refine_qps"][:plan["vtm"]["refine_step"] - 1]
    known = {u: {"qps": guess if r["reason"] == INCOMPLETE else r["qps"]} for u, r in refine.items()}
    tasks = stage_tasks(levels, "m", plan, timing) + refine_tasks(known, plan, timing) + stage_tasks(levels, "f", plan, timing)
    return tasks, refine


def unfinished(levels, plan, timing, last):
    """What keeps the run from its run file: tasks of the plan and controls not finished (never run or failed) and
    images whose grid is not finished."""
    done = {t for t, rows in last.items() if not e4_book.failed(rows)}
    tasks, refine = planned(levels, plan, timing, last)
    ids = [t["id"] for t in tasks] + [c[0] for c in control_ids(levels, plan)]
    return [i for i in ids if i not in done], sorted(u for u, r in refine.items() if r["reason"] == INCOMPLETE)


def progress_line(levels, plan, timing, last, session, wall_s, peak_bytes):
    """One line: finished / all tasks by level and codec, controls, error rows and failed tasks, the wall of the
    session, the expected rest (times and k of cycle 11) and the peak memory."""
    done = {t for t, rows in last.items() if not e4_book.failed(rows)}
    tasks, _ = planned(levels, plan, timing, last)
    count = {}
    for t in tasks:
        c = count.setdefault((t["level"], t["codec"]), [0, 0])
        c[0] += t["id"] in done
        c[1] += 1
    parts = [f"{name} " + ", ".join(f"{e4_book.CODEC_NAMES[c]} {count[(level, c)][0]}/{count[(level, c)][1]}"
                                    for c in e4_book.CODECS if (level, c) in count)
             for level, name in (("m", "whole images"), ("f", "fragments"))]
    controls = control_ids(levels, plan)
    rest_s = sum(t["expected"] for t in tasks if t["id"] not in done)
    errors = sum(r["status"] == "error" for t, rows in last.items() if entry_kind(t) == "task" for r in rows)
    to_retry = sum(e4_book.failed(rows) for t, rows in last.items() if entry_kind(t) in ("task", "control"))
    when = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (f"progress {when} | session {session}, wall {wall_s / 3600:.2f} h | done "
            f"{sum(c[0] for c in count.values())}/{len(tasks)} tasks: {'; '.join(parts)}; controls "
            f"{sum(c[0] in done for c in controls)}/{len(controls)} | error: {errors} rows, {to_retry} tasks to run "
            f"again | expected rest {rest_s / timing['k'] / 3600:.2f} h (cycle 11 times, k = {timing['k']:.2f}) | "
            f"peak memory {peak_bytes / 2 ** 30:.2f} GiB")


def sessions(entries):
    """Per session from the journal: start, end (stopped by plan and why, ended with tasks left, finished the run,
    or cut off - no end written), wall, tasks taken from the journal at its start, tasks it recorded, started tasks
    finished after the stop signal."""
    out, ids = [], []
    for e in entries:
        kind = entry_kind(e["task"])
        if kind == "session" and e["task"].endswith("|start"):
            out.append({"session": e["session"], "started_utc": e["utc"], "hours": e["hours"],
                        "taken_from_journal": e["taken_from_journal"], "end": "cut off", "wall_s": None,
                        "last_progress_wall_s": None, "recorded_tasks": 0, "finished_after_signal": None})
            ids.append(set())
        elif not out:
            continue
        elif kind == "session":
            out[-1] |= {"end": e["end"], "wall_s": e["wall_s"], "finished_after_signal": e["finished_after_signal"]}
        elif kind == "progress":
            out[-1]["last_progress_wall_s"] = e["wall_s"]
        elif kind in ("task", "control"):
            ids[-1].add(e["task"])
            out[-1]["recorded_tasks"] = len(ids[-1])
    return out


def tee(log, path):
    """The output of a session goes to `log` and to its own file, line by line."""
    def out(text):
        log(text)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"{text}\n")
    return out


class Runner:
    def __init__(self, paths, units, plan, journal, work, timing, session, t0, stop_after=None, log=print,
                 deadline_s=None, flag=None, clock=time.monotonic, poll_s=POLL_S, progress_s=PROGRESS_S):
        self.paths, self.units, self.plan, self.journal, self.work = paths, units, plan, journal, work
        self.timing, self.session, self.t0, self.stop_after, self.log = timing, session, t0, stop_after, log
        self.deadline_s, self.flag, self.clock, self.poll_s, self.progress_s = deadline_s, flag, clock, poll_s, progress_s
        self.levels = {u: i["level"] for u, i in units.items()}
        self.new, self.stopping, self.after_signal, self.retried = 0, None, 0, set()
        self.peak, self.probe, self.last_progress = 0, None, t0

    def record(self, task, rows, **extra):
        self.journal.append(task, rows, **extra)
        self.new += 1
        if self.stop_after is not None and self.new >= self.stop_after:
            raise RunStopped(f"stopped after {self.new} new tasks (test stop)")

    def check_stop(self, running):
        """A stop by plan: the hours of the session are over or the flag file lies beside the journal. Once noticed,
        no task is given any more; the `running` started ones are finished into the journal."""
        if self.stopping is None:
            if self.deadline_s is not None and self.clock() - self.t0 >= self.deadline_s:
                self.stopping = "deadline"
            elif self.flag is not None and self.flag.exists():
                self.stopping = "flag"
            if self.stopping:
                self.log(f"  planned stop ({self.stopping}): no new task is given; {running} started tasks are "
                         "finished into the journal")
        return self.stopping

    def progress(self, force=False):
        now = self.clock()
        if not force and now - self.last_progress < self.progress_s:
            return
        self.last_progress = now
        peak = max(self.peak, self.probe.peak_total if self.probe else 0)
        self.log(progress_line(self.levels, self.plan, self.timing, self.journal.last, self.session, now - self.t0, peak))
        self.journal.append(f"progress|{self.session}|{utc_now()}", [], wall_s=now - self.t0, peak_total_bytes=peak)

    def execute(self, task):
        info = self.units[task["unit"]]
        image = np.ascontiguousarray(info["array"])
        if task["codec"] == "vtm":
            return [vtm_task(self.paths, task["unit"], task["level"], image, task["param"], self.work)
                    | {"phase": task["kind"]}]
        return still_task(self.paths, task["unit"], task["level"], image, task["codec"], self.plan[task["codec"]], self.work)

    def work_on(self, task, session):
        return [r | {"session": session, "order": task["position"]} for r in self.execute(task)]

    def pool(self, name, tasks):
        """Unfinished tasks, slow first, at most `pool` at a time. A failed task runs again once in this session, right
        after the failure. After a stop signal no task is given and the started ones are finished into the journal."""
        todo = sorted((t for t in tasks if t["id"] not in self.journal.done), key=order_key)
        if not todo or self.check_stop(0):
            return
        for position, task in enumerate(todo):
            task["position"] = position
        session = utc_now()
        self.log(f"  {name}: {len(todo)} tasks, {self.plan['pool']} processes, whole images first, slow first")
        queue, running, done_ids, t0 = collections.deque(todo), {}, [], time.perf_counter()
        with e4_book.MemoryProbe() as probe:
            self.probe = probe
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.plan["pool"])
            try:
                while True:
                    while queue and len(running) < self.plan["pool"] and not self.check_stop(len(running)):
                        task = queue.popleft()
                        running[executor.submit(self.work_on, task, session)] = task
                    if not running:
                        break
                    finished, _ = concurrent.futures.wait(running, timeout=self.poll_s,
                                                          return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in finished:
                        task = running.pop(future)
                        rows = future.result()
                        self.after_signal += self.stopping is not None
                        done_ids.append(task["id"])
                        self.record(task["id"], rows)
                        if e4_book.failed(rows) and task["id"] not in self.retried and not self.stopping:
                            self.retried.add(task["id"])
                            queue.appendleft(task)
                            self.log(f"  {task['id']}: failed - runs again once, now")
                    self.check_stop(len(running))
                    self.progress()
            except (KeyboardInterrupt, RunStopped) as exc:
                executor.shutdown(wait=True, cancel_futures=True)
                raise RunStopped(str(exc) or "stopped by Ctrl+C") from None
            executor.shutdown(wait=True)
        self.peak, self.probe = max(self.peak, probe.peak_total), None
        self.journal.append(f"pool|{name}|{session}", [], wall_seconds=time.perf_counter() - t0, tasks=done_ids,
                            peak_total_bytes=probe.peak_total, peak_encoder_bytes=probe.peak_encoder,
                            stopped=self.stopping)

    def controls(self):
        for task, unit, qp in control_ids(self.levels, self.plan):
            if self.check_stop(0):
                return
            if task not in self.journal.done:
                info = self.units[unit]
                self.record(task, [vtm_task(self.paths, unit, info["level"], np.ascontiguousarray(info["array"]), qp,
                                            self.work) | {"phase": "control"}])


def header_of(plan, state, registry_sha, tiles_sha, paths):
    versions = {"jpeg": jpeg.versions(), "heif": heif.versions(), "avif_libheif": avif_libheif.EXPECTED_VERSIONS,
                "vtm_gpac_sha256": {k: sha256_file(paths[k]) if Path(paths[k]).is_file() else None
                                    for k in ("encoder", "decoder", "cfg", "mp4box", "gpac")}}
    return {"declared": DECLARED, "plan": plan, "core_commit": state["core_commit"], "core_tree": state["core_tree"],
            "codec_versions": versions, "registry_sha256": registry_sha, "tiles_sha256": tiles_sha}


def summarise(rows, entries, controls_rows, refine):
    pools = {name: [e for e in entries if e["task"].startswith(f"pool|{name}|")] for name in ("whole", "fragments")}
    k = None
    if pools["fragments"]:
        pool_ids = {e["task"].split("|", 2)[2] for e in pools["fragments"]}
        secs = sum(r["seconds"] for r in rows if r["codec"] == "vtm" and r.get("session") in pool_ids
                   and r.get("phase") == "fragment" and r["seconds"] is not None)
        wall = sum(e["wall_seconds"] for e in pools["fragments"])
        k = secs / wall if wall else None
    memory = {name: {"peak_pool_gib": max((e["peak_total_bytes"] for e in pools[name]), default=0) / 2 ** 30,
                     "peak_encoder_gib": max((e["peak_encoder_bytes"] for e in pools[name]), default=0) / 2 ** 30}
              for name in pools}
    by_key = {r["key"]: r for r in rows}
    det = [{"key": r["key"], "control_sha256": r["file_sha256"], "pool_sha256": (by_key.get(r["key"]) or {}).get("file_sha256"),
            "equal": r["file_sha256"] is not None and r["file_sha256"] == (by_key.get(r["key"]) or {}).get("file_sha256")}
           for r in controls_rows]
    counts = {}
    for r in rows:
        key = f"{r['level']}|{r['codec']}|{r['status']}"
        counts[key] = counts.get(key, 0) + 1
    return {"counts": counts, "k": k, "memory": memory, "determinism": det, "refinement": refine,
            "pool_walls": {name: [e["wall_seconds"] for e in pools[name]] for name in pools},
            "sessions": sessions(entries)}


def summary_lines(record):
    s = record["summary"]
    lines = [("Единиц на уровне «фрагменты», шт", sum(1 for v in record["units"].values() if v == "f")),
             ("Единиц на уровне «снимки», шт", sum(1 for v in record["units"].values() if v == "m")),
             ("Строк всего, шт", len(record["rows"]))]
    for key, n in sorted(s["counts"].items()):
        level, codec, status = key.split("|")
        lines.append((f"Строк: {e4_book.LEVELS[level]}, {e4_book.CODEC_NAMES[codec]}, {status}, шт", n))
    for unit, ref in sorted(s["refinement"].items()):
        lines.append((f"Уточнение PCC (VVC) снимка {unit}", ", ".join(map(str, ref["qps"])) if ref["qps"] else ref["reason"]))
    for x in s["sessions"]:
        wall = x["wall_s"] if x["wall_s"] is not None else x["last_progress_wall_s"]
        lines.append((f"Сессия {x['session']}: начата, UTC; конец; стена, ч; взято из журнала, задач; записано задач; "
                      "доведено после сигнала остановки",
                      f"{x['started_utc']}; {SESSION_END_RU.get(x['end'], x['end'])}; "
                      f"{'—' if wall is None else f'{wall / 3600:.2f}'}"
                      f"{'' if x['wall_s'] is not None else ' (не меньше: до последней строки прогресса)'}; "
                      f"{x['taken_from_journal']}; {x['recorded_tasks']}; "
                      f"{'—' if x['finished_after_signal'] is None else x['finished_after_signal']}"))
    a = record.get("assembled")
    if a:
        lines += [("Собран из журнала", a["line"]),
                  ("Журнал: sha256; байт; оборванных строк", f"{a['journal_sha256']}; {a['journal_bytes']}; {a['torn_lines']}"),
                  ("Коммит журнала; коммит сборки", f"{a['journal_commit']}; {a['assembly_commit']}"),
                  *((f"Незаконченная задача {u['task']}: попыток, шт; причина последней попытки",
                     f"{u['attempts']}; {u['detail']}") for u in a["unfinished"])]
    lines += [("Стена прогона по сессиям (сумма), с" if a else "Стена прогона (эта сессия), с", record["wall_seconds"]),
              ("Стена этапа «снимки» по сессиям, с", json.dumps(s["pool_walls"]["whole"])),
              ("Стена этапа «фрагменты» по сессиям, с", json.dumps(s["pool_walls"]["fragments"])),
              ("Ускорение пула k, раз", s["k"]),
              ("Память: этап «снимки», наибольшая сумма RSS, ГиБ", s["memory"]["whole"]["peak_pool_gib"]),
              ("Память: этап «фрагменты», наибольшая сумма RSS, ГиБ", s["memory"]["fragments"]["peak_pool_gib"]),
              ("Память: один EncoderApp, снимки, ГиБ", s["memory"]["whole"]["peak_encoder_gib"]),
              ("Память: один EncoderApp, фрагменты, ГиБ", s["memory"]["fragments"]["peak_encoder_gib"]),
              ("Контроль детерминизма", json.dumps(s["determinism"], ensure_ascii=False))]
    if a:
        return lines
    lines += [("Продолжение: задач взято из журнала (эта сессия), шт", record["resumed_tasks"]),
              ("Продолжение: задач со сбоем, выполненных снова (эта сессия), шт", record["retried_tasks"]),
              ("Повтор сразу после сбоя (эта сессия), задач, шт", record["retried_in_session"]),
              ("Продолжение: оборванных строк журнала, шт", record["torn_lines"])]
    return lines


def describe(record):
    a = record.get("assembled")
    built = [("Собран из журнала", f"{a['line']}; {a['utc']}; команда {a['command']}"),
             ("Журнал, sha256", a["journal_sha256"]),
             ("Коммит и дерево сборки", f"{a['assembly_commit']}; {a['assembly_tree']}")] if a else []
    return [("Начат, UTC", record["started_utc"]), ("Закончен, UTC", record["finished_utc"]),
            ("Коммит core", record["core_commit"]), ("Дерево core", record["core_tree"]), *built,
            *((f"{k} (до чисел)", v) for k, v in record["declared_before_numbers"].items()),
            ("Машина", json.dumps(record["machine"], ensure_ascii=False)),
            ("Где что", "«Просмотр» — единица в B1 (список: снимки D01…, фрагменты DNN_rRR_cCC); «Обзор» — уровень в B1 и "
                        "метрика в B2; «Сводка» — счёт строк, уточнение снимков, сессии, время, память; «Строки» — все строки"),
            ("На пальцах", "Каждый снимок и каждый его фрагмент сжаты четырьмя кодеками при каждом значении PCC (у целых "
                           "снимков VVC — по сетке и уточнению у порога), и у каждого файла посчитаны три метрики; книга "
                           "показывает поведение метрик и степени сжатия.")]


def _check(run_path, book_path):
    e4_book.check_book(run_path, book_path, describe, RunRefusal)


def build_book(run_path, book_path):
    e4_book.build_book(run_path, book_path, describe, _check)


def run(paths, units, plan, out_dir, state, timing, registry_sha, tiles_sha, hours=None, stop_after=None, log=print,
        clock=time.monotonic, poll_s=POLL_S, progress_s=PROGRESS_S):
    """One session: continue the journal until nothing is left or the session stops by plan; the session that leaves
    no task unfinished writes the run file and the book. Returns (run file, book), or None when the run goes on."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path, book_path, journal_path = out_dir / RUN_FILE, out_dir / BOOK_FILE, out_dir / JOURNAL_FILE
    flag = out_dir / STOP_FLAG
    if run_path.exists():
        raise RunRefusal(f"{run_path} already exists: a run file of E4 is never overwritten")
    if flag.exists():
        raise RunRefusal(f"the flag {flag} lies beside the journal: a session would stop at once - remove it to start")
    if journal_path.is_file():
        began = e4_book.read_journal(journal_path)[0].get("core_commit")
        if began != state["core_commit"]:
            raise RunRefusal(f"the journal was started at core commit {began}, the code is now at "
                             f"{state['core_commit']}: the code does not change between sessions - return to that "
                             "commit to continue")
    try:
        journal = e4_book.Journal(journal_path, header_of(plan, state, registry_sha, tiles_sha, paths))
    except e4_book.JournalRefusal as exc:
        raise RunRefusal(str(exc)) from exc
    levels = {u: i["level"] for u, i in units.items()}
    k = 1 + sum(1 for e in journal.entries if entry_kind(e["task"]) == "session" and e["task"].endswith("|start"))
    log = tee(log, out_dir / SESSION_LOG.format(k=k))
    taken = sum(1 for t in journal.done if entry_kind(t) in ("task", "control"))
    started, t0 = datetime.datetime.now(datetime.timezone.utc), clock()
    journal.append(f"session|{k}|start", [], session=k, utc=started.isoformat(timespec="seconds"), hours=hours,
                   core_commit=state["core_commit"], taken_from_journal=taken, failed_to_retry=journal.to_retry,
                   torn_lines=journal.torn_lines)
    work = out_dir / WORK_DIR
    shutil.rmtree(work, ignore_errors=True)
    if work.exists():
        log(f"  {WORK_DIR}: files of an earlier session could not be removed (a coder of a cut session may still hold "
            "them); this session works beside them")
    (work / f"session-{k}").mkdir(parents=True)
    log(f"{COMMAND}\n{QUESTION}\nDeclared before the numbers:")
    for key, text in DECLARED.items():
        log(f"  [{key}] {text}")
    log(f"  session {k}, " + (f"{hours:g} h" if hours else "no time limit") + f"; stop by plan: the empty file {flag}")
    log(f"  journal: {taken} finished tasks taken over, {journal.to_retry} failed tasks to run again, "
        f"{journal.torn_lines} torn lines ignored")
    runner = Runner(paths, units, plan, journal, work / f"session-{k}", timing, k, t0, stop_after, log,
                    None if hours is None else hours * 3600, flag, clock, poll_s, progress_s)
    try:
        runner.pool("whole", stage_tasks(levels, "m", plan, timing))
        if not runner.stopping:
            refine = refinements(levels, plan, journal.last)
            runner.pool("fragments", refine_tasks(refine, plan, timing) + stage_tasks(levels, "f", plan, timing))
        if not runner.stopping:
            runner.controls()
    except (KeyboardInterrupt, RunStopped) as exc:
        log(f"  {exc or 'stopped by Ctrl+C'}: the started tasks are lost, the journal keeps the finished ones")
        raise RunStopped(str(exc) or "stopped by Ctrl+C") from None
    finally:
        shutil.rmtree(work, ignore_errors=True)
    missing, incomplete = unfinished(levels, plan, timing, journal.last)
    wall = clock() - t0
    runner.progress(force=True)
    end = (f"stopped by plan ({runner.stopping})" if runner.stopping and missing else
           "ended with tasks left" if missing else "finished the run")
    journal.append(f"session|{k}|end", [], session=k, utc=utc_now(), end=end, wall_s=wall,
                   finished_after_signal=runner.after_signal if runner.stopping else None, left=len(missing))
    if flag.exists():
        os.replace(flag, out_dir / f"{STOP_FLAG}.done-{k}")
    if missing:
        shutil.copyfile(journal_path, out_dir / SESSION_COPY.format(k=k))
        log(f"  journal copied: {SESSION_COPY.format(k=k)}")
        log(STOPPED if runner.stopping else
            f"сессия закончена, не готово задач: {len(missing)} (снимков без законченной сетки: {len(incomplete)}) — "
            "та же команда повторит их")
        return None
    rows, controls_rows = [], []
    for task, task_rows in journal.last.items():
        if entry_kind(task) == "control":
            controls_rows += task_rows
        elif entry_kind(task) == "task":
            rows += task_rows
    order = list(units)
    rows.sort(key=lambda r: (order.index(r["unit"]), e4_book.CODECS.index(r["codec"]), r["param"]))
    summary = summarise(rows, journal.entries, controls_rows, refinements(levels, plan, journal.last))
    record = {
        "run": RUN_FILE, "study": "P001", "cycle": CYCLE, "question": QUESTION, "command": COMMAND,
        "started_utc": started.isoformat(timespec="seconds"), "finished_utc": utc_now(),
        "wall_seconds": round(wall, 1), **state, "declared_before_numbers": DECLARED, "plan": plan,
        "machine": e4_book.machine(), "environment": {"python": platform.python_version(),
                                                     "imstudies": imstudies.__version__, "numpy": np.__version__},
        "header": journal.header, "resumed_tasks": taken, "retried_tasks": journal.to_retry,
        "retried_in_session": len(runner.retried), "torn_lines": journal.torn_lines,
        "units": {u: i["level"] for u, i in units.items()},
        "rows": rows, "controls": controls_rows, "summary": summary, "overview": e4_book.overview(rows),
    }
    record["summary_lines"] = summary_lines(record)
    text = json.dumps(e4_book.jsonable(record), ensure_ascii=False, indent=1, allow_nan=False)
    with open(run_path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    build_book(run_path, book_path)
    for label, value in record["summary_lines"]:
        log(f"  {label}: {value}")
    log(f"Run file: {run_path} (sha256 {sha256_file(run_path)})\nBook: {book_path}")
    return run_path, book_path


def status(out_dir, levels, plan, timing, now=None, log=print):
    """Progress from the journal, running no task, while a session runs, after it stopped or after it was cut off:
    the progress line, the last session and whether it runs - by the freshness of the journal."""
    out_dir = Path(out_dir)
    journal_path = out_dir / JOURNAL_FILE
    if not journal_path.is_file():
        log(f"no journal {JOURNAL_FILE} in {out_dir}: E4 has not started")
        return None
    now = time.time() if now is None else now
    _, entries, torn = e4_book.read_journal(journal_path)
    table = sessions(entries)
    if not table:
        log(f"the journal {JOURNAL_FILE} has no session yet")
        return None
    last = {}
    for e in entries:
        last[e["task"]] = e["rows"]
    age = now - journal_path.stat().st_mtime
    s = table[-1]
    if (out_dir / RUN_FILE).is_file():
        state = f"the run is finished, {RUN_FILE} is written"
    elif s["end"] != "cut off":
        state = s["end"]
    elif age < FRESH_S:
        state = "running"
    else:
        state = f"cut off - no end written, the journal silent for {age / 60:.0f} min"
    if state == "running":
        wall = now - datetime.datetime.fromisoformat(s["started_utc"]).timestamp()
    else:
        wall = s["wall_s"] if s["wall_s"] is not None else (s["last_progress_wall_s"] or 0.0)
    peak = max((e.get("peak_total_bytes", 0) for e in entries if entry_kind(e["task"]) in ("pool", "progress")), default=0)
    log(progress_line(levels, plan, timing, last, s["session"], wall, peak))
    log(f"session {s['session']}: {state}; the journal changed {age / 60:.1f} min ago"
        + (f"; torn lines {torn}" if torn and state != "running" else ""))
    return state


def assemble(out_dir, levels, timing, state, journal_sha256=JOURNAL13_SHA256, log=print):
    """The run file and the book from the journal (STUDY.md section 7, «Сборка из журнала»): no task is run, no coder
    called, no pixel read. Every task gives its last entry, as the header of the journal declares; a task that failed
    at every attempt gives its error rows with the cause of its last attempt and the number of attempts. A task that
    never ran or a grid not finished is refused - the assembly closes failures, not an unfinished run. The journal
    may come from another commit: both commits go into the file. Returns (run file, book)."""
    out_dir = Path(out_dir)
    run_path, book_path, journal_path = out_dir / RUN_FILE, out_dir / BOOK_FILE, out_dir / JOURNAL_FILE
    if run_path.exists():
        raise RunRefusal(f"{run_path} already exists: a run file of E4 is never overwritten")
    if not journal_path.is_file():
        raise RunRefusal(f"no journal {journal_path} to assemble from")
    sha = sha256_file(journal_path)
    if sha != journal_sha256:
        raise RunRefusal(f"the journal {JOURNAL_FILE} has sha256 {sha}, the assembly is declared for {journal_sha256}")
    header, entries, torn = e4_book.read_journal(journal_path)
    plan, last, attempts = header["plan"], {}, collections.Counter()
    for e in entries:
        last[e["task"]] = e["rows"]
        if entry_kind(e["task"]) in ("task", "control"):
            attempts[e["task"]] += 1
    missing, incomplete = unfinished(levels, plan, timing, last)
    never = [t for t in missing if t not in last]
    if never or incomplete:
        raise RunRefusal(f"the journal is not a finished run: {len(never)} tasks never ran, grids of {incomplete} not "
                         f"finished - continue it with {COMMAND}; the assembly closes only tasks that failed every time")
    rows, controls_rows = [], []
    for task, task_rows in last.items():
        kind = entry_kind(task)
        if kind == "control":
            controls_rows += task_rows
        elif kind == "task":
            rows += [r | {"attempts": attempts[task]} for r in task_rows] if task in missing else task_rows
    order = list(levels)
    rows.sort(key=lambda r: (order.index(r["unit"]), e4_book.CODECS.index(r["codec"]), r["param"]))
    table = sessions(entries)
    ends = [e["utc"] for e in entries if entry_kind(e["task"]) == "session" and e["task"].endswith("|end")]
    wall = sum(x["wall_s"] if x["wall_s"] is not None else (x["last_progress_wall_s"] or 0.0) for x in table)
    record = {
        "run": RUN_FILE, "study": "P001", "cycle": CYCLE, "question": QUESTION, "command": COMMAND,
        "assembled": {"line": ASSEMBLED.format(n=len(missing)), "command": f"{COMMAND} --assemble", "utc": utc_now(),
                      "journal": JOURNAL_FILE, "journal_sha256": sha, "journal_bytes": journal_path.stat().st_size,
                      "torn_lines": torn, "journal_commit": header["core_commit"], "journal_tree": header["core_tree"],
                      "assembly_commit": state["core_commit"], "assembly_tree": state["core_tree"],
                      "unfinished": [{"task": t, "attempts": attempts[t], "detail": last[t][0]["detail"]}
                                     for t in missing]},
        "started_utc": table[0]["started_utc"] if table else None, "finished_utc": ends[-1] if ends else None,
        "wall_seconds": round(wall, 1), "core_commit": header["core_commit"], "core_tree": header["core_tree"],
        "declared_before_numbers": header["declared"], "plan": plan, "machine": e4_book.machine(),
        "environment": {"python": platform.python_version(), "imstudies": imstudies.__version__, "numpy": np.__version__},
        "header": header, "units": dict(levels), "rows": rows, "controls": controls_rows,
        "summary": summarise(rows, entries, controls_rows, refinements(levels, plan, last)),
        "overview": e4_book.overview(rows),
    }
    record["summary_lines"] = summary_lines(record)
    text = json.dumps(e4_book.jsonable(record), ensure_ascii=False, indent=1, allow_nan=False)
    with open(run_path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    build_book(run_path, book_path)
    for label, value in record["summary_lines"]:
        log(f"  {label}: {value}")
    log(f"{record['assembled']['line']}\nRun file: {run_path} (sha256 {sha256_file(run_path)})\n"
        f"Book: {book_path} (sha256 {sha256_file(book_path)}), checked cell by cell")
    return run_path, book_path


def plan_counts(registry, tile_rows, plan, timing):
    """The plan from metadata (LAB-EXPERIMENT, chain, 6): per image fragments and tasks; tasks and files per codec;
    expected hours for 1 and for 4 processes (times of cycle 11, A09; refinement at the QP of the dry run)."""
    v = plan["vtm"]
    n_still = {c: len(plan[c]) for c in ("jpeg", "heif", "avif")}
    per_image, totals, seconds = [], {c: {"tasks": 0, "files": 0} for c in e4_book.CODECS}, 0.0
    refine_n = v["refine_step"] - 1
    for entry in sorted(registry["images"], key=lambda e: e["code"]):
        n = sum(1 for r in tile_rows if r["code"] == entry["code"])
        units = 1 + n
        vtm_whole, vtm_frag = len(v["m_grid"]) + refine_n, n * len(v["f"])
        tasks = vtm_whole + vtm_frag + 3 * units
        per_image.append({"code": entry["code"], "H": entry["H"], "W": entry["W"], "fragments": n,
                          "vtm_whole": vtm_whole, "vtm_fragments": vtm_frag, "still_tasks": 3 * units, "tasks": tasks})
        totals["vtm"]["tasks"] += vtm_whole + vtm_frag
        totals["vtm"]["files"] += vtm_whole + vtm_frag
        for c in ("jpeg", "heif", "avif"):
            totals[c]["tasks"] += units
            totals[c]["files"] += units * n_still[c]
        seconds += expected_seconds(timing, "m", "vtm", v["m_grid"] + timing["refine_qps"][:refine_n])
        seconds += n * expected_seconds(timing, "f", "vtm", v["f"])
        for c in ("jpeg", "heif", "avif"):
            seconds += expected_seconds(timing, "m", c, plan[c]) + n * expected_seconds(timing, "f", c, plan[c])
    return {"images": per_image, "units_fragments": sum(p["fragments"] for p in per_image),
            "units_whole": len(per_image), "totals": totals,
            "tasks": sum(t["tasks"] for t in totals.values()), "files": sum(t["files"] for t in totals.values()),
            "hours_1": seconds / 3600, "hours_4": seconds / timing["k"] / 3600, "k": timing["k"]}


def print_plan(p, log=print):
    log("E4 plan from metadata (registry and grid, no pixels):")
    for i in p["images"]:
        log(f"  {i['code']}: {i['H']} x {i['W']}, fragments {i['fragments']}, VTM whole {i['vtm_whole']}, "
            f"VTM fragments {i['vtm_fragments']}, JPEG/HEIF/AVIF tasks {i['still_tasks']}, tasks {i['tasks']}")
    for c, t in p["totals"].items():
        log(f"  {e4_book.CODEC_NAMES[c]}: tasks {t['tasks']}, files {t['files']}")
    log(f"  units: {p['units_whole']} whole images, {p['units_fragments']} fragments; tasks {p['tasks']}, "
        f"files {p['files']}; expected {p['hours_1']:.1f} h in 1 process, {p['hours_4']:.1f} h in 4 (k = {p['k']:.2f})")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=float, help="a session of H hours: then no new task is given, the started ones "
                                                    "are finished; the same command continues")
    parser.add_argument("--status", action="store_true", help="progress from the journal; runs no task")
    parser.add_argument("--assemble", action="store_true", help="write the run file and the book from the journal of "
                                                                "cycle 13: no task, no coder, no pixel")
    parser.add_argument("--plan", action="store_true", help="print the plan from the registry and the grid; no pixels")
    parser.add_argument("--rebuild-book", action="store_true", help="rebuild the book from the existing run file")
    parser.add_argument("--out", type=Path, help="with --rebuild-book: the book to write (default: beside the run file); "
                                                 "an existing book is refused")
    args = parser.parse_args(argv)
    try:
        paths = load_paths()
        out_dir = paths["cache_root"] / "cycles" / str(CYCLE)
        if args.rebuild_book:
            if not (out_dir / RUN_FILE).is_file():
                raise RunRefusal(f"no run file to rebuild the book from: {out_dir / RUN_FILE}")
            book = out_dir / BOOK_FILE
            if args.out is not None:
                book = args.out if args.out.is_absolute() else (Path.cwd() / args.out)
                if book.exists():
                    raise RunRefusal(f"{book} already exists: --out never overwrites a book")
                book.parent.mkdir(parents=True, exist_ok=True)
            build_book(out_dir / RUN_FILE, book)
            print(f"Book rebuilt and checked cell by cell: {book}")
            return 0
        if args.hours is not None and not args.hours > 0:
            raise RunRefusal(f"--hours {args.hours:g}: a session needs a positive number of hours")
        tiles_path = paths["cache_root"] / "cycles" / "9" / "p001_tiles.json"
        registry, tile_rows, tile = load_inputs(paths["data_root"], tiles_path, copies=not args.assemble)
        cycle11_path = paths["cache_root"] / "cycles" / "11" / "e4_dry.json"
        if sha256_file(cycle11_path) != CYCLE11_SHA256:
            raise RunRefusal(f"{cycle11_path.name}: sha256 differs from the pinned {CYCLE11_SHA256}")
        timing = timing_from(json.loads(cycle11_path.read_text(encoding="utf-8")))
        if tile != FULL_PLAN["tile"]:
            raise RunRefusal(f"the grid has tiles of {tile} px, the plan {FULL_PLAN['tile']}")
        if args.plan:
            print_plan(plan_counts(registry, tile_rows, FULL_PLAN, timing))
            return 0
        if args.status:
            status(out_dir, unit_levels(registry, tile_rows), FULL_PLAN, timing)
            return 0
        if args.assemble:
            assemble(out_dir, unit_levels(registry, tile_rows), timing, git_state(ROOT))
            return 0
        if (out_dir / RUN_FILE).exists():
            raise RunRefusal(f"{out_dir / RUN_FILE} already exists: a run file of E4 is never overwritten")
        state = git_state(ROOT)
        units = load_units(paths["data_root"], registry, tile_rows, tile)
        run(paths, units, FULL_PLAN, out_dir, state, timing, REGISTRY_SHA256, TILES_SHA256, hours=args.hours)
    except RunStopped as stop:
        print(f"STOPPED: {stop}. The journal keeps every finished task: run the same command to continue.",
              file=sys.stderr)
        return 3
    except RunRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
