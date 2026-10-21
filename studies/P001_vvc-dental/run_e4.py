"""E4 of P001: at which PCC do VVC, JPEG, HEIF and AVIF reach the PSNR-HVS-M threshold on 154 fragments and 16 whole radiographs, and with what CR - the main run? Run: python studies/P001_vvc-dental/run_e4.py   (--plan: the plan from metadata, no pixels)

Study P001, contract E4 of STUDY.md section 7, search of section 5. Units: the 16 whole images of the copy
data/P001 (B10) and the 154 fragments of the centred grid (B11 0.2.0, cycles/9/p001_tiles.json). Whole images:
VTM grid QP -12, -8, ..., 60 and, after the grid of an image, 3 QP between the nodes around the first crossing of
PSNR-HVS-M = 42 dB from the high-quality side (no crossing - no_crossing with the reason); fragments: VTM QP
-12..63; JPEG, HEIF, AVIF - every value on both levels. A pool of 4 coder processes, slow tasks first by the
times of cycle 11, whole images before fragments; rows go to the journal e4_main.partial.jsonl as tasks finish;
the same command continues it (a failed task runs again; another header is refused); peak memory; determinism
control. Writes <cache_root>/cycles/<CYCLE>/e4_main.json once and the book e4_main.xlsx (e4_book.py). Only codes
- no file names, no pixels - leave the run. `--plan` reads the registry and the grid only.
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
import platform
import shutil
import statistics
import sys
import threading
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
CYCLE = 13  # the cycle that runs E4; the task of that cycle confirms or changes it
RUN_FILE, BOOK_FILE, JOURNAL_FILE, WORK_DIR = "e4_main.json", "e4_main.xlsx", "e4_main.partial.jsonl", "e4_main_work"
REGISTRY_SHA256 = "03affc6c150cb3f5beaf2a6c99c1f55861aa877666d77a9e20cc382e30d8f949"  # data/registry/P001.json, cycle 8
TILES_SHA256 = "aad8e80ccb5b972ef3ce7d9f22225f34c15e56ce08dc0247efbdf0af8fcc10a1"  # cycles/9/p001_tiles.json
CYCLE11_SHA256 = "a0365d9766cae3813d56b7a0c2f822cb9fce4c8bb11a4e11a37c0d3bf16979d9"  # cycles/11/e4_dry.json
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
               "незаконченной или со сбойной задачей — уточнение ждёт следующего запуска. JPEG 1…100, HEIF 0…100, AVIF "
               "0…100 (B12, пакет) — оба уровня полностью."),
    "per_file": ("Размер, CR по файлу, bpp, PSNR, PSNR-HVS-M, HaarPSI, секунды; sha256 файла и восстановления; статусы "
                 "valid, lossless, error."),
    "order": ("Пул из 4 процессов кодера. Этап 1 — все задачи снимков (сетка VTM, JPEG, HEIF, AVIF), этап 2 — уточнение "
              "снимков и все задачи фрагментов; внутри этапа — снимки раньше фрагментов, медленные первыми: ожидаемые "
              "секунды по циклу 11 (фрагменты — медиана шести изображений по значению PCC, снимки — мозаика, VTM — "
              "интерполяция по QP, вне — ближайшая; плюс медиана секунд метрик уровня), при равенстве — по кодеку, PCC, "
              "единице. JPEG и HEIF единицы — одна задача на все значения."),
    "resume": ("Журнал e4_main.partial.jsonl: шапка — объявленное, план, коммит, версии кодеков, sha256 реестра и сетки; "
               "строка на законченную задачу; та же команда продолжает; иная шапка — отказ словами; задача со всеми "
               "строками error выполняется снова, в файл прогона идёт её последняя запись."),
    "control": "Детерминизм: первый фрагмент QP 22 и 37 и первый снимок QP 40 повторяются последовательно — sha256 = пулу.",
    "memory": "Опрос каждые 0,5 с (psutil): сумма RSS прогона и дочерних, RSS одного EncoderApp — по этапам, ГиБ.",
    "book": "Книга по STUDY.md §7 («Книга E4», «Вид рядов»): PCC, таблица стиля, «Просмотр», «Обзор»; страниц нет (E6).",
}


class RunStopped(Exception):
    """The run was stopped (Ctrl+C or the test stop); the journal holds every finished task."""


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


def load_inputs(data_root, tiles_path, pins=(REGISTRY_SHA256, TILES_SHA256)):
    """Registry and grid (metadata only): pinned sha256, every copy present, tiles inside their image and as many as
    the registry says. No pixel is read here."""
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
        if not (data_root / "P001" / f"{code}.png").is_file():
            raise RunRefusal(f"{code}: the copy data/P001/{code}.png does not exist")
        mine = [r for r in rows if r["code"] == code]
        if len(mine) != entry["full_tiles"]:
            raise RunRefusal(f"{code}: {len(mine)} tiles in the grid, the registry says {entry['full_tiles']}")
        for r in mine:
            if not (0 <= r["x"] and r["x"] + tile <= entry["W"] and 0 <= r["y"] and r["y"] + tile <= entry["H"]):
                raise RunRefusal(f"{r['tile']}: the tile lies outside its image")
    return registry, rows, tile


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


def order_key(task):
    """Whole images before fragments; slow first; then codec, PCC, unit."""
    return (task["level"] != "m", -task["expected"], e4_book.CODECS.index(task["codec"]), str(task["param"]), task["unit"])


class Runner:
    def __init__(self, paths, units, plan, journal, work, timing, stop_after=None, log=print):
        self.paths, self.units, self.plan, self.journal, self.work = paths, units, plan, journal, work
        self.timing, self.stop_after, self.log, self.new = timing, stop_after, log, 0

    def record(self, task, rows, **extra):
        self.journal.append(task, rows, **extra)
        self.new += 1
        if self.stop_after is not None and self.new >= self.stop_after:
            raise RunStopped(f"stopped after {self.new} new tasks (test stop)")

    def vtm(self, unit, qp, kind):
        level = self.units[unit]["level"]
        return {"id": task_id(unit, "vtm", qp), "unit": unit, "level": level, "codec": "vtm", "param": qp, "kind": kind,
                "expected": expected_seconds(self.timing, level, "vtm", [qp])}

    def still(self, unit, codec):
        level = self.units[unit]["level"]
        param = "batch"
        return {"id": task_id(unit, codec, param), "unit": unit, "level": level, "codec": codec, "param": param,
                "kind": "still", "expected": expected_seconds(self.timing, level, codec, self.plan[codec])}

    def whole_tasks(self):
        tasks = []
        for unit, info in self.units.items():
            if info["level"] == "m":
                tasks += [self.vtm(unit, qp, "grid") for qp in self.plan["vtm"]["m_grid"]]
                tasks += [self.still(unit, codec) for codec in ("jpeg", "heif", "avif")]
        return tasks

    def fragment_tasks(self):
        tasks = []
        for unit, info in self.units.items():
            if info["level"] == "f":
                tasks += [self.vtm(unit, qp, "fragment") for qp in self.plan["vtm"]["f"]]
                tasks += [self.still(unit, codec) for codec in ("jpeg", "heif", "avif")]
        return tasks

    def refinements(self):
        """Per whole image: refinement QPs after a complete grid, or the reason why there are none (yet)."""
        out = {}
        v = self.plan["vtm"]
        for unit, info in self.units.items():
            if info["level"] != "m":
                continue
            grid = [self.journal.last.get(task_id(unit, "vtm", qp)) for qp in v["m_grid"]]
            if any(g is None or g[0]["status"] == "error" for g in grid):
                out[unit] = {"qps": [], "reason": "grid incomplete: a node is missing or failed - waits for the next run"}
                continue
            qps, reason = e4_book.refinement([(g[0]["param"], g[0]["psnr_hvs_m_db"]) for g in grid], v["threshold_db"],
                                             v["refine_step"])
            out[unit] = {"qps": qps, "reason": reason}
        return out

    def execute(self, task):
        info = self.units[task["unit"]]
        image = np.ascontiguousarray(info["array"])
        if task["codec"] == "vtm":
            return [vtm_task(self.paths, task["unit"], task["level"], image, task["param"], self.work)
                    | {"phase": task["kind"]}]
        return still_task(self.paths, task["unit"], task["level"], image, task["codec"], self.plan[task["codec"]], self.work)

    def pool(self, name, tasks):
        todo = sorted((t for t in tasks if t["id"] not in self.journal.done), key=order_key)
        if not todo:
            return
        session = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        self.log(f"  {name}: {len(todo)} tasks, {self.plan['pool']} processes, whole images first, slow first")
        stop = threading.Event()

        def work(position, task):
            if stop.is_set():
                return None
            rows = self.execute(task)
            return [r | {"session": session, "order": position} for r in rows]

        t0 = time.perf_counter()
        done_ids = []
        with e4_book.MemoryProbe() as probe:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.plan["pool"])
            futures = {executor.submit(work, k, t): t for k, t in enumerate(todo)}
            try:
                for future in concurrent.futures.as_completed(futures):
                    rows = future.result()
                    if rows is None or stop.is_set():
                        continue
                    done_ids.append(futures[future]["id"])
                    self.record(futures[future]["id"], rows)
            except (KeyboardInterrupt, RunStopped) as exc:
                stop.set()
                executor.shutdown(wait=True, cancel_futures=True)
                raise RunStopped(str(exc) or "stopped by Ctrl+C") from None
            executor.shutdown(wait=True)
        self.journal.append(f"pool|{name}|{session}", [], wall_seconds=time.perf_counter() - t0, tasks=done_ids,
                            peak_total_bytes=probe.peak_total, peak_encoder_bytes=probe.peak_encoder)

    def controls(self):
        c = self.plan["controls"]
        fragment = min((u for u, i in self.units.items() if i["level"] == "f"), default=None)
        whole = min((u for u, i in self.units.items() if i["level"] == "m"), default=None)
        for unit, qp in ([(fragment, q) for q in c["fragment_qps"]] if fragment else []) + \
                        ([(whole, c["whole_qp"])] if whole else []):
            task = f"control|{unit}|vtm|{qp}"
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
        sessions = {e["task"].split("|", 2)[2] for e in pools["fragments"]}
        secs = sum(r["seconds"] for r in rows if r["codec"] == "vtm" and r.get("session") in sessions
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
            "pool_walls": {name: [e["wall_seconds"] for e in pools[name]] for name in pools}}


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
    lines += [("Стена прогона (эта сессия), с", record["wall_seconds"]),
              ("Стена этапа «снимки» по сессиям, с", json.dumps(s["pool_walls"]["whole"])),
              ("Стена этапа «фрагменты» по сессиям, с", json.dumps(s["pool_walls"]["fragments"])),
              ("Ускорение пула k, раз", s["k"]),
              ("Память: этап «снимки», наибольшая сумма RSS, ГиБ", s["memory"]["whole"]["peak_pool_gib"]),
              ("Память: этап «фрагменты», наибольшая сумма RSS, ГиБ", s["memory"]["fragments"]["peak_pool_gib"]),
              ("Память: один EncoderApp, снимки, ГиБ", s["memory"]["whole"]["peak_encoder_gib"]),
              ("Память: один EncoderApp, фрагменты, ГиБ", s["memory"]["fragments"]["peak_encoder_gib"]),
              ("Контроль детерминизма", json.dumps(s["determinism"], ensure_ascii=False)),
              ("Продолжение: задач взято из журнала, шт", record["resumed_tasks"]),
              ("Продолжение: задач со сбоем, выполненных снова, шт", record["retried_tasks"]),
              ("Продолжение: оборванных строк журнала, шт", record["torn_lines"])]
    return lines


def describe(record):
    return [("Начат, UTC", record["started_utc"]), ("Закончен, UTC", record["finished_utc"]),
            ("Коммит core", record["core_commit"]), ("Дерево core", record["core_tree"]),
            *((f"{k} (до чисел)", v) for k, v in record["declared_before_numbers"].items()),
            ("Машина", json.dumps(record["machine"], ensure_ascii=False)),
            ("Где что", "«Просмотр» — единица в B1 (список: снимки D01…, фрагменты DNN_rRR_cCC); «Обзор» — уровень в B1 и "
                        "метрика в B2; «Сводка» — счёт строк, уточнение снимков, время, память; «Строки» — все строки"),
            ("На пальцах", "Каждый снимок и каждый его фрагмент сжаты четырьмя кодеками при каждом значении PCC (у целых "
                           "снимков VVC — по сетке и уточнению у порога), и у каждого файла посчитаны три метрики; книга "
                           "показывает поведение метрик и степени сжатия.")]


def _check(run_path, book_path):
    e4_book.check_book(run_path, book_path, describe, RunRefusal)


def build_book(run_path, book_path):
    e4_book.build_book(run_path, book_path, describe, _check)


def run(paths, units, plan, out_dir, state, timing, registry_sha, tiles_sha, stop_after=None, log=print):
    """One session: continue the journal; at the end write the run file and the book."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path, book_path, journal_path = out_dir / RUN_FILE, out_dir / BOOK_FILE, out_dir / JOURNAL_FILE
    if run_path.exists():
        raise RunRefusal(f"{run_path} already exists: a run file of E4 is never overwritten")
    try:
        journal = e4_book.Journal(journal_path, header_of(plan, state, registry_sha, tiles_sha, paths))
    except e4_book.JournalRefusal as exc:
        raise RunRefusal(str(exc)) from exc
    work = out_dir / WORK_DIR
    if work.exists():
        shutil.rmtree(work)
        log(f"  removed the working folder of a stopped session: {WORK_DIR}")
    work.mkdir()
    started, t0 = datetime.datetime.now(datetime.timezone.utc), time.perf_counter()
    log(f"{COMMAND}\n{QUESTION}\nDeclared before the numbers:")
    for key, text in DECLARED.items():
        log(f"  [{key}] {text}")
    log(f"  journal: {journal.resumed} finished tasks taken over, {journal.to_retry} failed tasks to run again, "
        f"{journal.torn_lines} torn lines ignored")
    runner = Runner(paths, units, plan, journal, work, timing, stop_after, log)
    try:
        runner.pool("whole", runner.whole_tasks())
        refine = runner.refinements()
        refine_tasks = [runner.vtm(unit, qp, "refine") for unit, ref in refine.items() for qp in ref["qps"]]
        runner.pool("fragments", refine_tasks + runner.fragment_tasks())
        runner.controls()
    except KeyboardInterrupt:
        raise RunStopped("stopped by Ctrl+C") from None
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rows, controls_rows = [], []
    for task, task_rows in journal.last.items():
        if task.startswith("control|"):
            controls_rows += task_rows
        elif not task.startswith("pool|"):
            rows += task_rows
    order = list(units)
    rows.sort(key=lambda r: (order.index(r["unit"]), e4_book.CODECS.index(r["codec"]), r["param"]))
    summary = summarise(rows, journal.entries, controls_rows, runner.refinements())
    record = {
        "run": RUN_FILE, "study": "P001", "cycle": CYCLE, "question": QUESTION, "command": COMMAND,
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.perf_counter() - t0, 1), **state, "declared_before_numbers": DECLARED, "plan": plan,
        "machine": e4_book.machine(), "environment": {"python": platform.python_version(),
                                                     "imstudies": imstudies.__version__, "numpy": np.__version__},
        "header": journal.header, "resumed_tasks": journal.resumed, "retried_tasks": journal.to_retry,
        "torn_lines": journal.torn_lines, "units": {u: i["level"] for u, i in units.items()},
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
        stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", action="store_true", help="print the plan from the registry and the grid; no pixels")
    parser.add_argument("--rebuild-book", action="store_true", help="rebuild the book from the existing run file")
    args = parser.parse_args(argv)
    try:
        paths = load_paths()
        out_dir = paths["cache_root"] / "cycles" / str(CYCLE)
        if args.rebuild_book:
            if not (out_dir / RUN_FILE).is_file():
                raise RunRefusal(f"no run file to rebuild the book from: {out_dir / RUN_FILE}")
            build_book(out_dir / RUN_FILE, out_dir / BOOK_FILE)
            print(f"Book rebuilt and checked cell by cell: {out_dir / BOOK_FILE}")
            return 0
        tiles_path = paths["cache_root"] / "cycles" / "9" / "p001_tiles.json"
        registry, tile_rows, tile = load_inputs(paths["data_root"], tiles_path)
        cycle11_path = paths["cache_root"] / "cycles" / "11" / "e4_dry.json"
        if sha256_file(cycle11_path) != CYCLE11_SHA256:
            raise RunRefusal(f"{cycle11_path.name}: sha256 differs from the pinned {CYCLE11_SHA256}")
        timing = timing_from(json.loads(cycle11_path.read_text(encoding="utf-8")))
        if tile != FULL_PLAN["tile"]:
            raise RunRefusal(f"the grid has tiles of {tile} px, the plan {FULL_PLAN['tile']}")
        if args.plan:
            print_plan(plan_counts(registry, tile_rows, FULL_PLAN, timing))
            return 0
        if (out_dir / RUN_FILE).exists():
            raise RunRefusal(f"{out_dir / RUN_FILE} already exists: a run file of E4 is never overwritten")
        state = git_state(ROOT)
        units = load_units(paths["data_root"], registry, tile_rows, tile)
        run(paths, units, FULL_PLAN, out_dir, state, timing, REGISTRY_SHA256, TILES_SHA256)
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
