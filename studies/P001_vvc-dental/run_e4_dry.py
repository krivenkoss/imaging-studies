"""E4 dry run: does the main run of P001 work end to end - all codecs, all values, metrics, the E4 book with charts, resume after a stop, memory - on test images? Run: python studies/P001_vvc-dental/run_e4_dry.py

Study P001, cycle 11. The path of E4 on test images (not P001): six 512 x 512 images of Images/BMP (level
"fragment") and the mosaic 1504 x 2773 of cycle 10 (level "whole image"); VTM (B05) with HEIF packing (B09) -
fragments QP -12..63, mosaic QP -12, -8, ..., 60 - by a pool of 4 coder processes, slow tasks first; JPEG
(B06), HEIF (B07), AVIF (B12, batch) - every value; PSNR, PSNR-HVS-M, HaarPSI (B02-B04) and CR, bpp of every
file. Rows go to the journal <cache_root>/cycles/11/e4_dry.partial.jsonl as tasks finish; the same command
started again continues it (another header is refused in words). At the end writes e4_dry.json (refuses by
name if it exists) and the book e4_dry.xlsx (e4_book.py); `--rebuild-book` rebuilds the book.
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

import imstudies
from imstudies.core import metrics
from imstudies.lab.coders import avif_libheif, heif, jpeg, vtm, vvc_heif
from imstudies.lab.images import load_test_images
from imstudies.lab.runs import RunRefusal, git_state, json_number, sha256_file

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
_spec = importlib.util.spec_from_file_location("e4_book", STUDY / "e4_book.py")
e4_book = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e4_book)

COMMAND = "python studies/P001_vvc-dental/run_e4_dry.py"
QUESTION = ("E4, сухой прогон: проходит ли путь основного прогона целиком (все кодеки и значения, метрики, книга с "
            "графиками, продолжение после остановки, память) на тестовых изображениях, и сколько часов займёт E4?")
CYCLE, RUN_FILE, BOOK_FILE, JOURNAL_FILE, WORK_DIR = 11, "e4_dry.json", "e4_dry.xlsx", "e4_dry.partial.jsonl", "e4_dry_work"
UNITS = ["BABOON.BMP", "BARBARA.BMP", "goldhill.bmp", "lenna.bmp", "frisco.bmp", "DIEGO.BMP"]  # also the mosaic order
MOSAIC_SHAPE = (1504, 2773)
CYCLE10_SHA256 = "a4acfec49938d89d12e1565a326207d2eae244fe8ae471b17ecacec96dc9bfd6"  # cycles/10/e4_timing.json
LEVEL, FRAME_RATE = "5", 1  # B05
TIMEOUT_S = {"f": 600, "m": 3600}
POOL = 4
N_FRAGMENTS, N_WHOLE, N_SEARCH = 154, 16, 8
THRESHOLD_DB = 42.0
FULL_PLAN = {
    "vtm": {"f": list(range(vtm.QP_MIN, vtm.QP_MAX + 1)), "m": list(range(-12, 61, 4))},
    "jpeg": list(range(1, 101)), "heif": list(range(0, 101)), "avif": list(range(0, 101)),
    "controls": [["frisco", "vtm", 22], ["frisco", "vtm", 37], ["mosaic", "vtm", 40]],
    "mosaic_shape": list(MOSAIC_SHAPE), "pool": POOL,
}
DECLARED = {
    "images": ("Тестовые, в P001 не входят: (ф) baboon, barbara, goldhill, lenna, frisco, diego — 512 × 512 из Images/BMP, "
               "уровень «фрагменты», n = 6; (м) мозаика 1504 × 2773 цикла 10 (sha256 пикселей сверяется) — уровень "
               "«снимки», n = 1. Коды — имена файлов строчными; mosaic."),
    "codecs": ("VTM (B05 + B09, Level 5, FrameRate 1): (ф) QP −12…63 (76), (м) QP −12, −8, …, 60 (19); JPEG 1…100, HEIF 0…100, "
               "AVIF (B12, пакетный вызов на единицу) 0…100 — на обоих уровнях полностью."),
    "per_file": ("Размер, CR по файлу = W·H/байт, bpp = 8·байт/(W·H); PSNR, PSNR-HVS-M, HaarPSI против исходного 8-битного; "
                 "секунды кодера и трёх метрик. Без потерь (восстановление = исходному): статус lossless, значения "
                 "хранятся (PSNR = +inf), на графиках точки нет, под графиком — счёт."),
    "order": ("Сначала контроль пути: JPEG, HEIF, AVIF и метрики на всех единицах и по одной задаче VTM на уровень — с "
              "наименьшим ожидаемым временем; затем пул VTM (ф), затем пул (м), по 4 процесса кодера (4 потока, у каждой "
              "задачи свой временный каталог обёртки). Медленные первыми: ожидаемые секунды задачи — по циклу 10 "
              "(ф — по QP из cycles/10/e4_timing.json, м — интерполяция четырёх точек по QP, вне — ближайшая), по "
              "убыванию; при равенстве — по QP, затем по единице."),
    "determinism": ("После пулов последовательно повторяются frisco QP 22, 37 и mosaic QP 40: sha256 файлов = строкам пула, "
                    "иначе записанный стоп."),
    "memory": ("Опрос каждые 0,5 с (psutil): сумма RSS процесса прогона и всех дочерних — наибольшая за пул (ф) и за пул (м); "
               "наибольшая RSS одного EncoderApp на (ф) и (м), ГиБ."),
    "resume": ("Журнал e4_dry.partial.jsonl: первая строка — объявленное, план, коммит, версии кодеков, sha256 пикселей; далее "
               "строка на законченную задачу. Повтор той же командой: шапка совпала — готовые задачи не пересчитываются; "
               "иначе — отказ словами. Сбой вызова — строка error с причиной, прогон идёт дальше; задача, все строки "
               "которой — error (в том числе оборванная остановкой: Ctrl+C доходит и до кодера), при повторном запуске "
               "выполняется снова, в файл прогона идёт её последняя запись. k — по сессиям пула, дошедшим до конца."),
    "estimate": ("Оценка E4 (1 и 4 процесса; k = Σ секунд вызовов VTM пула (ф) / стена пула (ф)). Фрагменты: 154 · Σ по "
                 "значениям медианы по шести единицам секунд вызова — для каждого кодека (у AVIF — стена пакета / число "
                 "значений); метрики — 154 · Σn · медиана по шести единицам медианы по их файлам суммы секунд трёх метрик. "
                 "Целые снимки: А — полный (16 · Σ по 76 QP t_м, t_м — линейная интерполяция по 19 QP мозаики, вне — "
                 "ближайшая); Б — грубая сетка и уточнение (16 · (Σ по 19 QP сетки + Σ по 3 QP между узлами вокруг "
                 "пересечения 42 дБ)); В — поиск порога (16 · 8 · t_м(QP пересечения)); пересечение — первое пересечение "
                 "PSNR-HVS-M = 42 дБ мозаики со стороны высокого качества (STUDY.md §6), линейная интерполяция по сетке; "
                 "JPEG, HEIF, AVIF и метрики снимков — полностью во всех вариантах (16 · Σ по значениям t мозаики; метрики "
                 "16 · число файлов варианта · медиана суммы трёх метрик мозаики). ПРЕДПОЛОЖЕНИЕ: время и пересечение на "
                 "тестовых изображениях близки к снимкам P001."),
    "book": ("Книга по STUDY.md §7 и LAB-EXPERIMENT: «Описание», «Сводка», «Просмотр» (13 графиков), «Обзор» (9 графиков), "
             "«Строки», «Обзор-данные» (медиана и квартили numpy.percentile, linear, по единицам уровня), «Списки»."),
}


class RunStopped(Exception):
    """The run was stopped (Ctrl+C or the test stop); the journal holds every finished task."""


def load_paths():
    path = ROOT / "lab_paths.json"
    if not path.is_file():
        raise RunRefusal(f"{path} not found: copy lab_paths.example.json to lab_paths.json and set the paths")
    raw = json.loads(path.read_text(encoding="utf-8"))
    paths = {}
    for key in ("images_root", "cache_root", "avif_libheif_python"):
        if not raw.get(key) or raw[key].startswith("<"):
            raise RunRefusal(f"lab_paths.json has no key '{key}' or it is the example placeholder")
        paths[key] = (ROOT / raw[key]).resolve()
    for codec, keys in (("vtm", ("encoder", "decoder", "cfg")), ("gpac", ("mp4box", "gpac"))):
        for key in keys:
            value = (raw.get("coders", {}).get(codec) or {}).get(key)
            if not value:
                raise RunRefusal(f"lab_paths.json has no key coders.{codec}.{key}")
            paths[key] = (ROOT / value).resolve()
            if not paths[key].is_file():
                raise RunRefusal(f"{codec} {key} not found: {paths[key]}")
    return paths


def mosaic(tiles, shape):
    """The mosaic of cycle 10: 3 rows x 6 columns of the tiles, row r shifted left by r, top-left crop."""
    rows = [np.concatenate([tiles[(c + r) % len(tiles)] for c in range(len(tiles))], axis=1) for r in range(3)]
    return np.ascontiguousarray(np.concatenate(rows, axis=0)[:shape[0], :shape[1]])


def load_units(images_root, cycle10, shape=MOSAIC_SHAPE, names=UNITS):
    tiles = load_test_images(images_root, names, (512, 512))
    units = {t["code"]: {"level": "f", "array": t["array"]} for t in tiles}
    m = mosaic([t["array"] for t in tiles], shape)
    if cycle10 is not None and hashlib.sha256(m.tobytes()).hexdigest() != cycle10["images"]["m"]["pixels_sha256"]:
        raise RunRefusal("the mosaic differs from cycle 10 (pixels sha256)")
    units["mosaic"] = {"level": "m", "array": m}
    return units


def expected_seconds(cycle10, level, qp):
    """Expected seconds of a VTM task from cycle 10: per QP on fragments; interpolation of 4 mosaic points."""
    seq = [r for r in cycle10["rows"] if r["codec"] == "vtm" and r["mode"] == 1]
    if level == "f":
        table = {r["param"]: r["seconds"] for r in seq if r["image"] == "f"}
        return table[qp]
    points = sorted((r["param"], r["seconds"]) for r in seq if r["image"] == "m")
    return interp(points, qp)


def interp(points, x):
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if x1 <= x <= x2:
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    raise ValueError(x)


def slow_first(tasks, expected):
    """Tasks (unit, qp) ordered by expected seconds descending; ties by QP, then by unit."""
    return sorted(tasks, key=lambda t: (-expected(t), t[1], t[0]))


def measure(original, recon):
    t0 = time.perf_counter()
    out = {"psnr_db": metrics.psnr(original, recon, max_pixels=original.size).value,
           "psnr_hvs_m_db": metrics.psnr_hvs_m(original, recon, max_pixels=original.size).value,
           "haarpsi": metrics.haarpsi(original, recon, max_pixels=original.size).value}
    out["seconds_metrics"] = time.perf_counter() - t0
    return out


def make_row(unit, level, codec, param, image, file, recon, seconds, **extra):
    h, w = image.shape
    row = {"key": f"{unit}|{codec}|{param}", "unit": unit, "level": level, "codec": codec, "param": param,
           "file_bytes": len(file), "cr_file": h * w / len(file), "bpp": 8 * len(file) / (h * w),
           "seconds": seconds, "file_sha256": hashlib.sha256(file).hexdigest(), "detail": None, **extra}
    row |= measure(image, recon)
    row["status"] = "lossless" if np.array_equal(image, recon) else "valid"
    return _jsonable(row)


def error_row(unit, level, codec, param, exc, **extra):
    return {"key": f"{unit}|{codec}|{param}", "unit": unit, "level": level, "codec": codec, "param": param,
            "status": "error", "detail": f"{type(exc).__name__}: {str(exc)[:400]}", "file_bytes": None, "cr_file": None,
            "bpp": None, "psnr_db": None, "psnr_hvs_m_db": None, "haarpsi": None, "seconds": None,
            "seconds_metrics": None, "file_sha256": None, **extra}


def _jsonable(obj):
    if isinstance(obj, float):
        return json_number(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def vtm_task(paths, unit, level, image, qp, work):
    try:
        t0 = time.perf_counter()
        coded = vtm.encode_decode(image, qp, max_pixels=image.size, level=LEVEL, frame_rate=FRAME_RATE,
                                  timeout_s=TIMEOUT_S[level], encoder=paths["encoder"], decoder=paths["decoder"],
                                  cfg=paths["cfg"], work_root=work)
        t1 = time.perf_counter()
        packed = vvc_heif.pack(coded.bitstream, coded.recon, timeout_s=TIMEOUT_S[level], mp4box=paths["mp4box"],
                               gpac=paths["gpac"], decoder=paths["decoder"], work_root=work)
        t2 = time.perf_counter()
        return make_row(unit, level, "vtm", qp, image, packed.file, coded.recon, t2 - t0, seconds_b05=t1 - t0,
                        seconds_b09=t2 - t1, bitstream_sha256=hashlib.sha256(coded.bitstream).hexdigest())
    except Exception as exc:  # noqa: BLE001 - status `error`, the run goes on
        return error_row(unit, level, "vtm", qp, exc)


def still_rows(paths, unit, level, image, codec, values, work):
    if codec == "avif":
        try:
            t0 = time.perf_counter()
            coded = avif_libheif.encode_decode(image, list(values), max_pixels=image.size, timeout_s=TIMEOUT_S[level],
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
    """RSS of this process and all children every 0.5 s; peak total and peak of one EncoderApp, bytes."""

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


class Runner:
    """One session of the dry run on a journal; `stop_after` stops after that many new journal entries (tests)."""

    def __init__(self, paths, units, plan, journal, work, cycle10, stop_after=None, log=print):
        self.paths, self.units, self.plan, self.journal, self.work = paths, units, plan, journal, work
        self.cycle10, self.stop_after, self.log, self.new = cycle10, stop_after, log, 0

    def record(self, task, rows, **extra):
        self.journal.append(task, rows, **extra)
        self.new += 1
        if self.stop_after is not None and self.new >= self.stop_after:
            raise RunStopped(f"stopped after {self.new} new tasks (test stop)")

    def expected(self, level):
        return lambda t: expected_seconds(self.cycle10, level, t[1])

    def vtm_tasks(self, level):
        return [(u, qp) for u, info in self.units.items() if info["level"] == level for qp in self.plan["vtm"][level]]

    def path_control(self):
        for unit, info in self.units.items():
            for codec in ("jpeg", "heif"):
                for q in self.plan[codec]:
                    task = f"{unit}|{codec}|{q}"
                    if task not in self.journal.done:
                        self.record(task, still_rows(self.paths, unit, info["level"], info["array"], codec, [q], self.work))
            task = f"{unit}|avif|batch"
            if task not in self.journal.done:
                self.record(task, still_rows(self.paths, unit, info["level"], info["array"], "avif", self.plan["avif"],
                                             self.work))
        for level in ("f", "m"):
            unit, qp = min(self.vtm_tasks(level), key=lambda t: (self.expected(level)(t), t[1], t[0]))
            task = f"{unit}|vtm|{qp}"
            if task not in self.journal.done:
                info = self.units[unit]
                self.record(task, [vtm_task(self.paths, unit, level, info["array"], qp, self.work) | {"phase": "path"}])

    def pool(self, level):
        todo = [t for t in slow_first(self.vtm_tasks(level), self.expected(level))
                if f"{t[0]}|vtm|{t[1]}" not in self.journal.done]
        if not todo:
            return
        session = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        self.log(f"  pool ({level}): {len(todo)} VTM tasks, {POOL} processes, slow first")
        stop = threading.Event()

        def work(position, unit, qp):
            if stop.is_set():
                return None
            row = vtm_task(self.paths, unit, level, self.units[unit]["array"], qp, self.work)
            return row | {"phase": f"pool {level}", "session": session, "order": position}

        t0 = time.perf_counter()
        done_ids = []
        with MemoryProbe() as probe:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=POOL)
            futures = {executor.submit(work, k, u, qp): (u, qp) for k, (u, qp) in enumerate(todo)}
            try:
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    if row is None or stop.is_set():
                        continue
                    unit, qp = futures[future]
                    done_ids.append(f"{unit}|vtm|{qp}")
                    self.record(f"{unit}|vtm|{qp}", [row])
            except (KeyboardInterrupt, RunStopped) as exc:
                stop.set()
                executor.shutdown(wait=True, cancel_futures=True)
                raise RunStopped(str(exc) or "stopped by Ctrl+C") from None
            executor.shutdown(wait=True)
        wall = time.perf_counter() - t0
        self.journal.append(f"pool|{level}|{session}", [], wall_seconds=wall, tasks=done_ids,
                            peak_total_bytes=probe.peak_total, peak_encoder_bytes=probe.peak_encoder)

    def controls(self):
        for unit, codec, qp in self.plan["controls"]:
            task = f"control|{unit}|{codec}|{qp}"
            if task in self.journal.done or unit not in self.units:
                continue
            info = self.units[unit]
            self.record(task, [vtm_task(self.paths, unit, info["level"], info["array"], qp, self.work)
                               | {"phase": "control"}])


def header_of(plan, state, units, paths):
    versions = {"jpeg": jpeg.versions(), "heif": heif.versions(), "avif_libheif": avif_libheif.EXPECTED_VERSIONS,
                "vtm_gpac_sha256": {k: sha256_file(paths[k]) for k in ("encoder", "decoder", "cfg", "mp4box", "gpac")}}
    return {"declared": DECLARED, "plan": plan, "core_commit": state["core_commit"], "core_tree": state["core_tree"],
            "codec_versions": versions, "images": {u: hashlib.sha256(i["array"].tobytes()).hexdigest()
                                                   for u, i in units.items()}}


# ---------------------------------------------------------------- summary and estimate

def crossing_qp(rows, threshold=THRESHOLD_DB):
    """First crossing of PSNR-HVS-M = threshold from the high-quality side (low QP) on the mosaic VTM grid."""
    pts = sorted((r["param"], math.inf if r["psnr_hvs_m_db"] == "+inf" else r["psnr_hvs_m_db"]) for r in rows
                 if r["unit"] == "mosaic" and r["codec"] == "vtm" and r["status"] in ("valid", "lossless"))
    for (q1, v1), (q2, v2) in zip(pts, pts[1:]):
        if v1 >= threshold > v2:
            return q1 + (v1 - threshold) * (q2 - q1) / (v1 - v2) if math.isfinite(v1) else float(q1)
    return None


def estimate(rows, k, n_values):
    """The declared estimate: fragments, three variants of whole images, hours for 1 and for 4 processes."""
    seconds = lambda r: r["seconds"]
    frag_units = sorted({r["unit"] for r in rows if r["level"] == "f"})

    def median_over_units(codec, param):
        values = [seconds(r) for r in rows if (r["codec"], r["param"], r["level"]) == (codec, param, "f")
                  and r["status"] != "error"]
        return statistics.median(values) if values else 0.0

    fragments = {c: N_FRAGMENTS * sum(median_over_units(c, p) for p in n_values[c]) for c in n_values}
    per_unit_metric = [statistics.median(r["seconds_metrics"] for r in rows if r["unit"] == u and r["status"] != "error")
                       for u in frag_units]
    n_files = sum(len(v) for v in n_values.values())
    fragments["metrics"] = N_FRAGMENTS * n_files * statistics.median(per_unit_metric)
    mosaic_rows = [r for r in rows if r["unit"] == "mosaic" and r["status"] != "error"]
    t_metric_m = statistics.median(r["seconds_metrics"] for r in mosaic_rows)
    grid = sorted((r["param"], seconds(r)) for r in mosaic_rows if r["codec"] == "vtm")
    still_m = sum(seconds(r) for r in mosaic_rows if r["codec"] != "vtm")
    n_still = sum(len(n_values[c]) for c in ("jpeg", "heif", "avif"))
    cross = crossing_qp(rows)
    variants = {}
    variants["A"] = {"vtm_calls": len(n_values["vtm"]), "vtm_s": N_WHOLE * sum(interp(grid, q) for q in n_values["vtm"])}
    if cross is not None:
        q_grid = [q for q, _ in grid]
        lower = max(q for q in q_grid if q <= cross)
        upper = min((q for q in q_grid if q > lower), default=lower)
        between = [q for q in range(lower + 1, upper)]
        variants["B"] = {"vtm_calls": len(grid) + len(between),
                         "vtm_s": N_WHOLE * (sum(t for _, t in grid) + sum(interp(grid, q) for q in between)),
                         "refine_qps": between}
        variants["C"] = {"vtm_calls": N_SEARCH, "vtm_s": N_WHOLE * N_SEARCH * interp(grid, cross)}
    else:
        variants["B"] = variants["C"] = None
    out = {"fragments_s": fragments, "fragments_total_s": sum(fragments.values()), "k": k, "crossing_qp": cross,
           "whole_still_s": N_WHOLE * still_m, "t_metric_mosaic_s": t_metric_m, "variants": {}}
    for name, v in variants.items():
        if v is None:
            out["variants"][name] = None
            continue
        metric_s = N_WHOLE * (v["vtm_calls"] + n_still) * t_metric_m
        total = out["fragments_total_s"] + v["vtm_s"] + out["whole_still_s"] + metric_s
        out["variants"][name] = v | {"whole_metrics_s": metric_s, "total_1_s": total, "hours_1": total / 3600,
                                     "hours_4": total / k / 3600 if k else None}
    return out


def summarise(rows, entries, controls_rows, plan):
    pools = {lvl: [e for e in entries if e["task"].startswith(f"pool|{lvl}|")] for lvl in ("f", "m")}
    pool_rows = {lvl: [r for r in rows if r["codec"] == "vtm" and r.get("phase") == f"pool {lvl}"] for lvl in ("f", "m")}
    k = None
    if pools["f"]:
        sessions = {e["task"].split("|", 2)[2]: e for e in pools["f"]}
        secs = sum(r["seconds"] for r in pool_rows["f"] if r.get("session") in sessions and r["seconds"] is not None)
        k = secs / sum(e["wall_seconds"] for e in sessions.values())
    memory = {lvl: {"peak_pool_gib": max((e["peak_total_bytes"] for e in pools[lvl]), default=0) / 2 ** 30,
                    "peak_encoder_gib": max((e["peak_encoder_bytes"] for e in pools[lvl]), default=0) / 2 ** 30}
              for lvl in ("f", "m")}
    by_key = {r["key"]: r for r in rows}
    det = []
    for r in controls_rows:
        pool_row = by_key.get(r["key"])
        det.append({"key": r["key"], "control_sha256": r["file_sha256"], "pool_sha256": pool_row and pool_row["file_sha256"],
                    "equal": pool_row is not None and r["file_sha256"] is not None
                    and r["file_sha256"] == pool_row["file_sha256"]})
    counts = {}
    for r in rows:
        counts.setdefault(f"{r['level']}|{r['codec']}|{r['status']}", 0)
        counts[f"{r['level']}|{r['codec']}|{r['status']}"] += 1
    n_values = {"vtm": plan["vtm"]["f"], "jpeg": plan["jpeg"], "heif": plan["heif"], "avif": plan["avif"]}
    return {"counts": counts, "k": k, "memory": memory, "determinism": det,
            "pool_walls": {lvl: [e["wall_seconds"] for e in pools[lvl]] for lvl in ("f", "m")},
            "estimate": estimate(rows, k, n_values) if k else None}


def summary_lines(record):
    s, e = record["summary"], record["summary"]["estimate"]
    lines = [("Единиц на уровне «фрагменты», шт", sum(1 for u in record["units"].values() if u == "f")),
             ("Единиц на уровне «снимки», шт", sum(1 for u in record["units"].values() if u == "m")),
             ("Строк всего, шт", len(record["rows"]))]
    for key, n in sorted(s["counts"].items()):
        level, codec, status = key.split("|")
        lines.append((f"Строк: {e4_book.LEVELS[level]}, {e4_book.CODEC_NAMES[codec]}, {status}, шт", n))
    lines += [("Стена прогона (эта сессия), с", record["wall_seconds"]),
              ("Стена пула (ф) по сессиям, с", json.dumps(s["pool_walls"]["f"])),
              ("Стена пула (м) по сессиям, с", json.dumps(s["pool_walls"]["m"])),
              ("Ускорение пула k, раз", s["k"]),
              ("Память: пул (ф), наибольшая сумма RSS, ГиБ", s["memory"]["f"]["peak_pool_gib"]),
              ("Память: пул (м), наибольшая сумма RSS, ГиБ", s["memory"]["m"]["peak_pool_gib"]),
              ("Память: один EncoderApp (ф), наибольшая RSS, ГиБ", s["memory"]["f"]["peak_encoder_gib"]),
              ("Память: один EncoderApp (м), наибольшая RSS, ГиБ", s["memory"]["m"]["peak_encoder_gib"]),
              ("Контроль детерминизма", json.dumps(s["determinism"], ensure_ascii=False)),
              ("Продолжение: задач взято из журнала в последней сессии, шт", record["resumed_tasks"]),
              ("Продолжение: задач со сбоем, выполненных снова в последней сессии, шт", record["retried_tasks"]),
              ("Продолжение: оборванных строк журнала, шт", record["torn_lines"])]
    if e:
        lines += [("QP пересечения PSNR-HVS-M = 42 дБ на мозаике", e["crossing_qp"]),
                  ("Оценка: фрагменты (все кодеки и метрики), ч в 1 процесс", e["fragments_total_s"] / 3600)]
        for name, label in (("A", "А — полный"), ("B", "Б — грубая сетка и уточнение"), ("C", "В — поиск порога")):
            v = e["variants"][name]
            lines += [(f"Оценка E4, вариант {label}, ч в 1 процесс", v and v["hours_1"]),
                      (f"Оценка E4, вариант {label}, ч в 4 процесса", v and v["hours_4"]),
                      (f"Оценка E4, вариант {label}, вызовов VTM на снимок, шт", v and v["vtm_calls"])]
    return lines


def describe(record):
    lines = [("Начат, UTC", record["started_utc"]), ("Закончен, UTC", record["finished_utc"]),
             ("Коммит core", record["core_commit"]), ("Дерево core", record["core_tree"]),
             *((f"{k} (до чисел)", v) for k, v in record["declared_before_numbers"].items()),
             ("Машина", json.dumps(record["machine"], ensure_ascii=False)),
             ("Где что", "«Просмотр» — выбор единицы в B1 (список); «Обзор» — уровень в B1 и метрика в B2 (списки); "
                         "«Сводка» — счёт строк, время, память, оценка E4; «Строки» — все строки; «Обзор-данные» — "
                         "медиана и квартили; «Списки» — значения выпадающих списков"),
             ("На пальцах", "Тестовые снимки сжаты всеми четырьмя кодеками при каждом значении параметра, и у каждого файла "
                            "посчитаны три метрики; книга показывает их поведение графиками для выбранного снимка и сводно, "
                            "а сводка прикидывает, сколько часов займёт основной прогон.")]
    return lines


def _check(run_path, book_path):
    e4_book.check_book(run_path, book_path, describe, RunRefusal)


def build_book(run_path, book_path):
    e4_book.build_book(run_path, book_path, describe, _check)


def machine():
    out = {"processor": platform.processor(), "logical_cpus": os.cpu_count(), "platform": platform.platform()}
    try:
        import psutil
        out |= {"physical_cpus": psutil.cpu_count(logical=False),
                "memory_total_gib": round(psutil.virtual_memory().total / 2 ** 30, 1)}
    except ImportError:
        pass
    return out


def run(paths, units, plan, out_dir, state, cycle10, stop_after=None, log=print):
    """One session: continue the journal, finish the tasks; at the end write the run file and the book."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path, book_path, journal_path = out_dir / RUN_FILE, out_dir / BOOK_FILE, out_dir / JOURNAL_FILE
    if run_path.exists():
        raise RunRefusal(f"{run_path} already exists: a run file of cycle {CYCLE} is never overwritten")
    try:
        journal = e4_book.Journal(journal_path, header_of(plan, state, units, paths))
    except e4_book.JournalRefusal as exc:
        raise RunRefusal(str(exc)) from exc
    work = out_dir / WORK_DIR
    if work.exists():  # temporary files of a stopped session
        shutil.rmtree(work)
        log(f"  removed the working folder of a stopped session: {WORK_DIR}")
    work.mkdir()
    started, t0 = datetime.datetime.now(datetime.timezone.utc), time.perf_counter()
    log(f"{COMMAND}\n{QUESTION}\nDeclared before the numbers:")
    for key, text in DECLARED.items():
        log(f"  [{key}] {text}")
    log(f"  journal: {journal.resumed} finished tasks taken over, {journal.to_retry} failed tasks to run again, "
        f"{journal.torn_lines} torn lines ignored")
    runner = Runner(paths, units, plan, journal, work, cycle10, stop_after, log)
    try:
        runner.path_control()
        log(f"  path control done: {len(journal.done)} journal entries")
        runner.pool("f")
        runner.pool("m")
        runner.controls()
    except KeyboardInterrupt:
        raise RunStopped("stopped by Ctrl+C") from None
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rows, controls_rows = [], []
    for task, task_rows in journal.last.items():  # the last entry of every task counts
        if task.startswith("control|"):
            controls_rows += task_rows
        elif not task.startswith("pool|"):
            rows += task_rows
    rows.sort(key=lambda r: (list(units).index(r["unit"]), e4_book.CODECS.index(r["codec"]), r["param"]))
    summary = summarise(rows, journal.entries, controls_rows, plan)
    stopped = None if all(d["equal"] for d in summary["determinism"]) else "determinism control: a file differs from the pool"
    record = {
        "run": RUN_FILE, "study": "P001", "cycle": CYCLE, "question": QUESTION, "command": COMMAND,
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.perf_counter() - t0, 1), **state, "declared_before_numbers": DECLARED, "plan": plan,
        "machine": machine(), "environment": {"python": platform.python_version(), "imstudies": imstudies.__version__,
                                              "numpy": np.__version__},
        "header": journal.header, "resumed_tasks": journal.resumed, "retried_tasks": journal.to_retry,
        "torn_lines": journal.torn_lines,
        "units": {u: i["level"] for u, i in units.items()}, "stopped": stopped,
        "rows": rows, "controls": controls_rows, "summary": summary, "overview": e4_book.overview(rows),
    }
    record["summary_lines"] = summary_lines(record)
    text = json.dumps(_jsonable(record), ensure_ascii=False, indent=1, allow_nan=False)
    with open(run_path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    build_book(run_path, book_path)
    for label, value in record["summary_lines"]:
        log(f"  {label}: {value}")
    log(f"Run file: {run_path} (sha256 {sha256_file(run_path)})\nBook: {book_path}")
    return run_path, book_path


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
        if (out_dir / RUN_FILE).exists():
            raise RunRefusal(f"{out_dir / RUN_FILE} already exists: a run file of cycle {CYCLE} is never overwritten")
        cycle10_path = paths["cache_root"] / "cycles" / "10" / "e4_timing.json"
        if sha256_file(cycle10_path) != CYCLE10_SHA256:
            raise RunRefusal(f"{cycle10_path.name}: sha256 differs from the pinned {CYCLE10_SHA256}")
        cycle10 = json.loads(cycle10_path.read_text(encoding="utf-8"))
        state = git_state(ROOT)
        units = load_units(paths["images_root"], cycle10)
        run(paths, units, FULL_PLAN, out_dir, state, cycle10)
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
