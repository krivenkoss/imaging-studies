"""E5 of P001: does VVC give a higher CR than JPEG, HEIF and AVIF at the PSNR-HVS-M threshold, per patient, by the rule declared before the numbers? Run: python studies/P001_vvc-dental/run_e5.py

Study P001, STUDY.md section 6 («Статистический план», «Объявление E5»); contracts B13 (MGM) and B14 (paired
comparison). Input: the run file of E4 <cache_root>/cycles/13/e4_main.json (pinned sha256); the copy data/P001
and the grid of cycle 9 (B10, B11, sha256 checked) - only for MGM and for JPEG re-encoded with optimised Huffman
tables. Per unit, codec and T = 40, 41, 42 dB: the threshold PCC by the declared walk, or why the unit is not
judged; per patient (n = 16): the median CR at the threshold over the judged fragments, r = CR_VVC / CR_codec, B14
on ln r with Holm over the three comparisons at 42 dB (40 and 41 dB - descriptive); the inference rule of section 6
beside the numbers. Descriptive: whole images (VVC at 40 and 41 dB also interpolated in ln CR between grid nodes),
MGM and Spearman rho, the JPEG point PCC 50 (named after Fidler et al. only if the quantisation table of the
owner's IrfanView file equals ours), JPEG with optimize=True (its reconstruction must equal E4). Writes
<cache_root>/cycles/15/e5_analysis.json once and the book e5_analysis.xlsx, every cell checked against the run
file. Only codes leave the run.
"""

import argparse
import datetime
import hashlib
import importlib.util
import io
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import PIL
import scipy
from PIL import Image
from scipy import stats

import imstudies
from imstudies.core import complexity, paired
from imstudies.core.metrics import Refusal
from imstudies.lab.coders import jpeg
from imstudies.lab.coders.still import check_versions, decode_gray
from imstudies.lab.images import load_test_images
from imstudies.lab.runs import RunRefusal, git_state, sha256_file

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]


def _module(name):
    spec = importlib.util.spec_from_file_location(name, STUDY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_e4 = _module("run_e4")
e4_book = run_e4.e4_book

COMMAND = "python studies/P001_vvc-dental/run_e5.py"
QUESTION = ("E5, P001: даёт ли VVC больший CR, чем JPEG, HEIF и AVIF, в пороге PSNR-HVS-M — по пациентам, по правилу, "
            "объявленному до чисел?")
CYCLE = 17  # cycle 15 first; cycle 17 - whole images at 40 and 41 dB with the interpolated CR of VVC (STUDY.md section 6)
BASELINE_SHA256 = "be161b1e46b3d81c62c22c859ed4840b53f4e4d99b264a181070d7ab7e7dc16a"  # cycles/15/e5_analysis.json (A13)
CHANGED = ("whole_ratios", "whole_comparisons")  # the only fields that may differ from cycle 15, at 40 and 41 dB
RUN_FILE, BOOK_FILE = "e5_analysis.json", "e5_analysis.xlsx"
E4_RUN_SHA256 = "44a1733512fdfbb9d02d210ff81d492b9a5d929ec8d547d84676d47350875f2a"  # cycles/13/e4_main.json (A12)
THRESHOLDS = (40.0, 41.0, 42.0)
MAIN_T = 42.0
CODECS = ("vtm", "jpeg", "heif", "avif")
JPEG_OPT = "jpeg_opt"  # JPEG re-encoded with optimize=True: a variant of JPEG, not a fifth codec
NAMES = {**e4_book.CODEC_NAMES, JPEG_OPT: "JPEG (оптимизированные таблицы Хаффмана)"}
PAIRS = (("vtm", "jpeg"), ("vtm", "heif"), ("vtm", "avif"))  # the three comparisons of the statistical plan
OPT_PAIR = ("vtm", JPEG_OPT)  # descriptive (STUDY.md section 6, Huffman sensitivity)
GRID_STEP = 4  # the VTM grid of whole images (STUDY.md section 5)
FIDLER_PCC = 50
FIDLER_IMAGE = "frisco.bmp"
NOT_REACHED = "не судится: порог не достигнут"
ABOVE_BEST = "не судится: порог выше наилучшего качества"
GAP = "не судится: пропуск у порога"
RULE = ("«VVC выгоднее кодека X» — только если нижняя граница 95 % ДИ > 1 и p по Холму < 0,05; «VVC выгоднее всех» — "
        "только если это выполнено для всех трёх; иначе — отрицательный или неопределённый результат, сообщается как "
        "есть (STUDY.md §6, решение владельца 2026-10-14).")
DECLARED = {
    "input": ("Файл прогона E4 cycles/13/e4_main.json (sha256 сверяется); пиксели копии data/P001 и сетка цикла 9 — только "
              "для MGM и для перекодирования JPEG с optimize=True."),
    "threshold": ("Пороговый PCC единицы и кодека при T: PCC обходятся от высокого качества (VVC — от QP −12 вверх, "
                  "JPEG, HEIF, AVIF — от 100 вниз); lossless — выше порога; пороговый — последний PCC перед первым PCC "
                  "с PSNR-HVS-M < T. Все точки ≥ T — «не судится: порог не достигнут»; первая же точка < T — «не "
                  "судится: порог выше наилучшего качества»; точка error в обходе пропускается, но между последним ≥ T "
                  "и первым < T — «не судится: пропуск у порога». Уточнение исполнителя до чисел: error до первой "
                  "точки ≥ T тоже считается пропуском у порога. CR в пороге — CR файла при пороговом PCC; рядом — "
                  "PSNR, HaarPSI, bpp той же точки."),
    "patients": ("Пациент — снимок, n = 16. Для кодека — медиана CR в пороге по судимым фрагментам снимка (numpy.median; "
                 "рядом — судимо из скольких); r = медиана CR_VVC / медиана CR_кодек; пациент без судимых фрагментов по "
                 "паре — вне теста с причиной. B14 по d = ln r (точный двусторонний Вилкоксон scipy, оценка "
                 "Ходжеса–Лемана, 95 % ДИ), Холм на 3 сравнения при T = 42 дБ; при T = 40 и 41 дБ — то же описательно."),
    "rule": RULE,
    "whole": ("Целые снимки — описательно: пороговый PCC по точкам прогона (у VVC — сетка и уточнение); при T = 40 и 41 "
              "дБ у VVC, где между двумя узлами сетки нет уточнения, — CR, линейно интерполированный по ln CR между "
              "узлами, окружающими пересечение PSNR-HVS-M = T; r по снимку и B14 — описательно. Уточнение (решение "
              "владельца 2026-10-26, с цикла 17): в r и B14 снимков при T = 40 и 41 дБ у VVC входит интерполированный "
              "CR (где есть уточнение — значение в узле), r по узлу печатается рядом; в цикле 15 в r вошёл узловой."),
    "complexity": ("MGM (B13) — по исходному фрагменту и по целому снимку. Описательно: ρ Спирмена (scipy.stats.spearmanr) "
                   "MGM с пороговым PCC, ln CR при 42 дБ по кодекам и ln r трёх пар — по 154 фрагментам и по пациентам "
                   "(медианы, n = 16). Ожидание до чисел: чем сложнее фрагмент, тем меньше CR в пороге (Lukin et al., "
                   "2022); опровергнет ρ(MGM, ln CR_VVC) ≥ 0 на фрагментах."),
    "jpeg50": ("Точка JPEG PCC 50 на всех единицах: PSNR-HVS-M, PSNR, HaarPSI, CR; доля единиц с PSNR-HVS-M ≥ 40 / 41 / "
               "42 дБ; медианы по пациентам. Имя «уровень 50 по Fidler et al.» — только если таблица квантования файла "
               "IrfanView владельца равна нашей при PCC 50; иначе — «JPEG PCC 50». CR у Fidler et al. — размер TIFF / "
               "размер JPEG, у нас — 8·W·H / биты файла."),
    "huffman": ("Чувствительность JPEG к таблицам Хаффмана — описательно: все 170 единиц при всех PCC — Pillow 12.3.0 "
                "quality = q, optimize = True, progressive = False; sha256 восстановления = строке E4, иначе отказ "
                "словами; размер, CR_opt, CR_opt / CR; пороговый PCC тот же, CR в пороге — CR_opt; r_VVC / JPEG_opt и "
                "B14 при T = 40, 41, 42 дБ — p печатаются, вывод не меняют; при PCC 50 CR_opt — «CR как у IrfanView»."),
}


# ---------------------------------------------------------------- the threshold PCC

def walk(codec, rows):
    """PCC from the high-quality side: VVC from the lowest QP up, JPEG, HEIF, AVIF from 100 down."""
    return sorted(rows, key=lambda r: r["param"], reverse=codec != "vtm")


def above(row, t):
    return row["status"] == "lossless" or (row["status"] == "valid" and row["psnr_hvs_m_db"] >= t)


def threshold(codec, rows, t):
    """(row of the threshold PCC, None) or (None, the reason the unit is not judged) - the declared walk."""
    last, gap = None, False
    for row in walk(codec, rows):
        if row["status"] == "error":
            gap = True
        elif above(row, t):
            last, gap = row, False
        elif gap:
            return None, GAP
        elif last is None:
            return None, ABOVE_BEST
        else:
            return last, None
    return None, NOT_REACHED


def interpolated(codec, rows, chosen, t):
    """Whole images, VVC, T below the main one: CR at PSNR-HVS-M = T, linear in ln CR between the grid node `chosen`
    (>= T) and the next node (< T), when no refinement lies between them. (CR or None, words)."""
    after = [r for r in walk(codec, rows) if r["param"] > chosen["param"] and r["status"] != "error"]
    b = after[0] if after else None
    if b is None or b["param"] - chosen["param"] != GRID_STEP:
        return None, "есть уточнение между узлами — значение в узле"
    pa, pb = chosen["psnr_hvs_m_db"], b["psnr_hvs_m_db"]
    if chosen["status"] != "valid" or not (pb < t <= pa):
        return None, "интерполяция неприменима: узел без потерь"
    ln_cr = math.log(chosen["cr_file"]) + (t - pa) / (pb - pa) * (math.log(b["cr_file"]) - math.log(chosen["cr_file"]))
    return math.exp(ln_cr), f"между PCC {chosen['param']} и {b['param']}"


# ---------------------------------------------------------------- JPEG with optimised Huffman tables

def encode_opt(image, quality):
    """Pillow quality=q, optimize=True, progressive=False (the rest as B06); decoding as B06. A module name so that
    a test can put a coder of its own in its place."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image), "L").save(buf, "JPEG", quality=quality, optimize=True, progressive=False)
    file = buf.getvalue()
    return file, decode_gray(file, image.shape, "JPEG")


def recon_sha256(recon):
    return hashlib.sha256(np.ascontiguousarray(recon).tobytes()).hexdigest()


def huffman(arrays, jpeg_rows):
    """Every unit and JPEG PCC re-encoded with optimize=True; the reconstruction must equal the E4 row."""
    out = []
    for unit, image in arrays.items():
        h, w = image.shape
        for q in sorted(p for (u, p) in jpeg_rows if u == unit):
            row = jpeg_rows[(unit, q)]
            file, recon = encode_opt(image, q)
            sha = recon_sha256(recon)
            if sha != row["recon_sha256"]:
                raise RunRefusal(f"{unit}, JPEG PCC {q}: the reconstruction with optimize=True has sha256 {sha}, the E4 "
                                 f"row {row['recon_sha256']} - optimised Huffman tables must not change the pixels")
            cr_opt = h * w / len(file)
            out.append({"unit": unit, "pcc": q, "file_bytes": row["file_bytes"], "file_bytes_opt": len(file),
                        "cr": row["cr_file"], "cr_opt": cr_opt, "ratio": cr_opt / row["cr_file"], "recon_equal": True})
    return out


def fidler_scale(folder, image):
    """The owner's IrfanView file of frisco.bmp at level 50 against our JPEG (B06) at PCC 50 on the same image."""
    files = sorted(Path(folder).glob("frisco_irfanview-*_q50.jpg")) if folder and Path(folder).is_dir() else []
    if not files:
        return {"present": False, "equal": False, "words": "файла IrfanView нет к запуску E5"}
    path = files[-1]
    data = path.read_bytes()
    ours = jpeg.encode_decode(image, FIDLER_PCC, max_pixels=image.size)
    opt, _ = encode_opt(image, FIDLER_PCC)
    with Image.open(io.BytesIO(data)) as im:
        mode, tables, info = im.mode, {k: list(v) for k, v in im.quantization.items()}, dict(im.info)
        decoded = np.asarray(im).copy()
    with Image.open(io.BytesIO(ours.file)) as im:
        ours_tables = {k: list(v) for k, v in im.quantization.items()}
    t_irf, t_ours = tables.get(0, []), ours_tables.get(0, [])
    equal = mode == "L" and len(tables) == 1 and len(ours_tables) == 1 and t_irf == t_ours
    return {"present": True, "file": path.name, "version": path.stem.removeprefix("frisco_irfanview-").removesuffix("_q50"),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "mode": mode, "tables": len(tables),
            "progressive": bool(info.get("progressive") or info.get("progression")),
            "table_irfanview": t_irf, "table_ours": t_ours, "equal": equal,
            "differ_at": [k for k, (a, b) in enumerate(zip(t_irf, t_ours)) if a != b],
            "ours_bytes": len(ours.file), "ours_sha256": hashlib.sha256(ours.file).hexdigest(),
            "optimize_bytes": len(opt), "optimize_sha256": hashlib.sha256(opt).hexdigest(),
            "equals_pillow_optimize": data == opt,
            "decoded_equal_ours": bool(decoded.shape == ours.recon.shape and np.array_equal(decoded, ours.recon)),
            "words": "таблицы квантования равны" if equal else "таблицы квантования не равны"}


# ---------------------------------------------------------------- per unit, per patient, comparisons

def units_table(e4_rows, levels, image_of, opt):
    """Unit x codec x T: the threshold PCC and its point, or the reason it is not judged."""
    by = {}
    for r in e4_rows:
        by.setdefault((r["unit"], r["codec"]), []).append(r)
    out = []
    for unit, level in levels.items():
        for codec in CODECS:
            rows = by.get((unit, codec), [])
            for t in THRESHOLDS:
                row, reason = threshold(codec, rows, t)
                rec = {"unit": unit, "level": level, "image": image_of[unit], "codec": codec, "t_db": t,
                       "judged": row is not None, "reason": reason, "pcc": None, "cr": None, "cr_interp": None,
                       "interp_note": None, "cr_opt": None, "psnr_db": None, "psnr_hvs_m_db": None, "haarpsi": None,
                       "bpp": None, "status": None}
                if row is not None:
                    rec |= {"pcc": row["param"], "cr": row["cr_file"], "psnr_db": row["psnr_db"], "status": row["status"],
                            "psnr_hvs_m_db": row["psnr_hvs_m_db"], "haarpsi": row["haarpsi"], "bpp": row["bpp"]}
                    if codec == "jpeg":
                        rec["cr_opt"] = opt[(unit, row["param"])]["cr_opt"]
                    if level == "m" and codec == "vtm" and t != MAIN_T:
                        rec["cr_interp"], rec["interp_note"] = interpolated(codec, rows, row, t)
                out.append(rec)
    return out


def cr_of(rec, codec):
    return rec["cr_opt"] if codec == JPEG_OPT else rec["cr"]


def base(codec):
    return "jpeg" if codec == JPEG_OPT else codec


def patients_table(units, images):
    out = []
    for image in images:
        for codec in (*CODECS, JPEG_OPT):
            for t in THRESHOLDS:
                frs = [u for u in units if u["level"] == "f" and u["image"] == image and u["codec"] == base(codec)
                       and u["t_db"] == t]
                crs = [cr_of(u, codec) for u in frs if u["judged"]]
                out.append({"image": image, "codec": codec, "t_db": t, "judged": len(crs), "of": len(frs),
                            "median_cr": float(np.median(crs)) if crs else None})
    return out


def ratios_table(medians, images, level_name):
    """r per patient (or per whole image) for each pair and T; out of the test with the reason."""
    out = []
    for t in THRESHOLDS:
        for a, b in (*PAIRS, OPT_PAIR):
            for image in images:
                va, vb = medians.get((image, a, t)), medians.get((image, b, t))
                rec = {"level": level_name, "image": image, "pair": f"{a}/{b}", "t_db": t, "r": None, "ln_r": None,
                       "in_test": False, "reason": None}
                if va is None:
                    rec["reason"] = f"нет судимых единиц {NAMES[a]}"
                elif vb is None:
                    rec["reason"] = f"нет судимых единиц {NAMES[b]}"
                else:
                    rec |= {"r": va / vb, "ln_r": math.log(va / vb), "in_test": True}
                out.append(rec)
    return out


def compare(ratios, level_name):
    """B14 per pair and T; Holm over the three comparisons of the plan at each T (a comparison without a p counts
    as p = 1); the verdict by the rule only at the main T and the patient level."""
    out = []
    for t in THRESHOLDS:
        found = []
        for a, b in (*PAIRS, OPT_PAIR):
            d = [x["ln_r"] for x in ratios if x["t_db"] == t and x["pair"] == f"{a}/{b}" and x["in_test"]]
            rec = {"level": level_name, "pair": f"{a}/{b}", "t_db": t, "n": len(d), "status": "valid", "reason": None,
                   "w_plus": None, "p": None, "p_holm": None, "hl": None, "ci_low": None, "ci_high": None, "k": None,
                   "ratio": None, "ratio_low": None, "ratio_high": None, "verdict": None,
                   "role": ("основное" if (t == MAIN_T and level_name == "пациенты" and (a, b) in PAIRS) else "описательно")}
            try:
                res = paired.wilcoxon_hl(np.array(d, dtype=np.float64))
                rec |= {"w_plus": res.w_plus, "p": res.p_value, "hl": res.hl, "ci_low": res.ci95[0], "ci_high": res.ci95[1],
                        "k": res.k, "ratio": res.ratio, "ratio_low": res.ratio_ci95[0], "ratio_high": res.ratio_ci95[1]}
            except Refusal as exc:
                rec |= {"status": "no_result", "reason": exc.reason}
            found.append(rec)
        family = [r for r in found if tuple(r["pair"].split("/")) in PAIRS]
        for rec, p in zip(family, paired.holm([r["p"] if r["p"] is not None else 1.0 for r in family])):
            rec["p_holm"] = p
        for rec in family:
            if rec["role"] == "основное":
                ok = rec["status"] == "valid" and rec["ratio_low"] > 1 and rec["p_holm"] < 0.05
                name = NAMES[rec["pair"].split("/")[1]]
                rec["verdict"] = (f"VVC выгоднее {name}" if ok else
                                  f"«VVC выгоднее {name}» не установлено — отрицательный или неопределённый результат")
        out += found
    return out


def spearman(x, y):
    """(n, rho) over the pairs where both are known; rho is not defined (None) below 3 pairs or on a constant."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    xs, ys = [a for a, _ in pairs], [b for _, b in pairs]
    if len(pairs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return len(pairs), None
    return len(pairs), float(stats.spearmanr(xs, ys).statistic)


def complexity_tables(units, mgm, images, ratios):
    """Per fragment and per patient at the main T: MGM, threshold PCC and ln CR per codec, ln r per pair; rho."""
    at = {(u["unit"], u["codec"]): u for u in units if u["t_db"] == MAIN_T}
    fragments = []
    for unit in sorted(u for u, lv in {x["unit"]: x["level"] for x in units}.items() if lv == "f"):
        rec = {"unit": unit, "image": at[(unit, "vtm")]["image"], "mgm": mgm[unit]}
        for c in CODECS:
            rec[f"pcc_{c}"] = at[(unit, c)]["pcc"]
            rec[f"ln_cr_{c}"] = math.log(at[(unit, c)]["cr"]) if at[(unit, c)]["judged"] else None
        for a, b in PAIRS:
            rec[f"ln_r_{b}"] = (rec[f"ln_cr_{a}"] - rec[f"ln_cr_{b}"]) if None not in (rec[f"ln_cr_{a}"], rec[f"ln_cr_{b}"]) else None
        fragments.append(rec)
    patients = []
    for image in images:
        mine = [f for f in fragments if f["image"] == image]
        rec = {"image": image, "mgm": float(np.median([f["mgm"] for f in mine]))}
        for c in CODECS:
            pccs = [f[f"pcc_{c}"] for f in mine if f[f"pcc_{c}"] is not None]
            crs = [math.exp(f[f"ln_cr_{c}"]) for f in mine if f[f"ln_cr_{c}"] is not None]
            rec[f"pcc_{c}"] = float(np.median(pccs)) if pccs else None
            rec[f"ln_cr_{c}"] = math.log(float(np.median(crs))) if crs else None
        for a, b in PAIRS:
            r = next(x for x in ratios if x["image"] == image and x["pair"] == f"{a}/{b}" and x["t_db"] == MAIN_T)
            rec[f"ln_r_{b}"] = r["ln_r"]
        patients.append(rec)
    rhos = []
    for level, table in (("фрагменты", fragments), ("пациенты", patients)):
        for key, name in ([(f"pcc_{c}", f"пороговый PCC, {NAMES[c]}") for c in CODECS]
                          + [(f"ln_cr_{c}", f"ln CR в пороге 42 дБ, {NAMES[c]}") for c in CODECS]
                          + [(f"ln_r_{b}", f"ln r, VVC / {NAMES[b]}") for _, b in PAIRS]):
            n, rho = spearman([t["mgm"] for t in table], [t[key] for t in table])
            rhos.append({"level": level, "x": "MGM", "y": name, "key": key, "n": n, "rho": rho})
    vvc = next(r for r in rhos if r["level"] == "фрагменты" and r["key"] == "ln_cr_vtm")
    expectation = {"rho": vvc["rho"], "n": vvc["n"],
                   "words": ("ожидание не опровергнуто: ρ(MGM, ln CR_VVC) < 0 на фрагментах" if vvc["rho"] is not None
                             and vvc["rho"] < 0 else "ожидание опровергнуто: ρ(MGM, ln CR_VVC) ≥ 0 на фрагментах")}
    return fragments, patients, rhos, expectation


def jpeg50_tables(e4_rows, levels, image_of, opt, images, scale):
    rows = {r["unit"]: r for r in e4_rows if r["codec"] == "jpeg" and r["param"] == FIDLER_PCC}
    units = [{"unit": u, "level": lv, "image": image_of[u], "psnr_hvs_m_db": rows[u]["psnr_hvs_m_db"],
              "psnr_db": rows[u]["psnr_db"], "haarpsi": rows[u]["haarpsi"], "cr": rows[u]["cr_file"],
              "cr_opt": opt[(u, FIDLER_PCC)]["cr_opt"], "status": rows[u]["status"]} for u, lv in levels.items()]
    shares = []
    for level, name in (("f", "фрагменты"), ("m", "снимки")):
        mine = [u for u in units if u["level"] == level]
        for t in THRESHOLDS:
            k = sum(1 for u in mine if u["status"] == "lossless" or u["psnr_hvs_m_db"] >= t)
            shares.append({"level": name, "t_db": t, "n": len(mine), "count": k, "share_pct": 100.0 * k / len(mine)})
    patients = []
    for image in images:
        mine = [u for u in units if u["level"] == "f" and u["image"] == image]
        patients.append({"image": image, "n": len(mine), **{f"median_{k}": float(np.median([u[k] for u in mine]))
                                                             for k in ("psnr_hvs_m_db", "psnr_db", "haarpsi", "cr", "cr_opt")}})
    name = "уровень 50 по Fidler et al." if scale["equal"] else "JPEG PCC 50"
    note = ("CR у Fidler et al. — размер TIFF / размер JPEG, у нас — 8·W·H / биты файла (по файлу хранения, STUDY.md §5); "
            "CR_opt — «CR как у IrfanView» (оптимизированные таблицы Хаффмана)")
    if not scale["equal"]:
        note += "; оговорка: шкала IrfanView не сверена или не равна нашей"
    return {"name": name, "note": note, "units": units, "shares": shares, "patients": patients}


def quartiles(values):
    v = np.array(values, dtype=np.float64)
    return {"n": int(v.size), "median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75)), "min": float(v.min()), "max": float(v.max())}


# ---------------------------------------------------------------- the run

def analyse(e4, levels, image_of, arrays, frisco, scale_folder):
    """Everything of E5 from the E4 rows, the pixels (MGM, JPEG re-encoding) and the owner's IrfanView file."""
    rows = e4["rows"]
    images = sorted({v for u, v in image_of.items() if levels[u] == "m"})
    jpeg_rows = {(r["unit"], r["param"]): r for r in rows if r["codec"] == "jpeg"}
    opt_list = huffman(arrays, jpeg_rows)
    opt = {(o["unit"], o["pcc"]): o for o in opt_list}
    units = units_table(rows, levels, image_of, opt)
    mgm = {u: complexity.mgm(np.ascontiguousarray(a)) for u, a in arrays.items()}
    for rec in units:
        rec["mgm"] = mgm[rec["unit"]]
    patients = patients_table(units, images)
    medians = {(p["image"], p["codec"], p["t_db"]): p["median_cr"] for p in patients}
    ratios = ratios_table(medians, images, "пациенты")
    comparisons = compare(ratios, "пациенты")
    whole = [u for u in units if u["level"] == "m"]
    # whole images: VVC at 40 and 41 dB enters r with its interpolated CR where there is one (cycle 17), else the node
    whole_medians = {(u["image"], u["codec"], u["t_db"]):
                     (u["cr_interp"] if u["codec"] == "vtm" and u["cr_interp"] is not None else u["cr"])
                     for u in whole if u["judged"]}
    whole_medians |= {(u["image"], JPEG_OPT, u["t_db"]): u["cr_opt"] for u in whole if u["judged"] and u["codec"] == "jpeg"}
    whole_ratios = ratios_table(whole_medians, images, "снимки")
    node = {(u["image"], u["t_db"]): u["cr"] for u in whole if u["judged"] and u["codec"] == "vtm"}
    for x in whole_ratios:
        if x["t_db"] != MAIN_T:  # the r of the node beside it
            x["r_node"] = node[(x["image"], x["t_db"])] / whole_medians[(x["image"], x["pair"].split("/")[1],
                                                                         x["t_db"])] if x["in_test"] else None
    whole_comparisons = compare(whole_ratios, "снимки")
    fragments, patient_complexity, rhos, expectation = complexity_tables(units, mgm, images, ratios)
    scale = fidler_scale(scale_folder, frisco)
    jpeg50 = jpeg50_tables(rows, levels, image_of, opt, images, scale)
    main = [c for c in comparisons if c["role"] == "основное"]
    all_ok = all(c["verdict"].startswith("VVC выгоднее") for c in main)
    verdict = {"pairs": [{"pair": c["pair"], "verdict": c["verdict"]} for c in main],
               "all": "VVC выгоднее всех" if all_ok else "«VVC выгоднее всех» не установлено"}
    return {"units": units, "patients": patients, "ratios": ratios, "comparisons": comparisons, "verdict": verdict,
            "whole_ratios": whole_ratios, "whole_comparisons": whole_comparisons,
            "complexity": {"fragments": fragments, "patients": patient_complexity, "rho": rhos, "expectation": expectation,
                           "images": [{"image": i, "mgm": mgm[i]} for i in images]},
            "jpeg50": jpeg50, "fidler_scale": scale,
            "huffman": {"rows": opt_list, "recon_equal": sum(o["recon_equal"] for o in opt_list), "of": len(opt_list),
                        "ratio": quartiles([o["ratio"] for o in opt_list]),
                        "ratio_fragments": quartiles([o["ratio"] for o in opt_list if levels[o["unit"]] == "f"]),
                        "ratio_images": quartiles([o["ratio"] for o in opt_list if levels[o["unit"]] == "m"])},
            "counts": counts(units)}


def counts(units):
    """Judged units by level, codec and T, of how many; reasons of the others."""
    out = []
    for level, name in (("f", "фрагменты"), ("m", "снимки")):
        for codec in CODECS:
            for t in THRESHOLDS:
                mine = [u for u in units if u["level"] == level and u["codec"] == codec and u["t_db"] == t]
                reasons = {}
                for u in mine:
                    if not u["judged"]:
                        reasons[u["reason"]] = reasons.get(u["reason"], 0) + 1
                out.append({"level": name, "codec": codec, "t_db": t, "judged": sum(u["judged"] for u in mine),
                            "of": len(mine), "reasons": json.dumps(reasons, ensure_ascii=False) if reasons else None})
    return out


def description(record):
    """The lines of «Описание», all from the run file."""
    a = record["analysis"]
    lines = [("Вопрос", QUESTION), ("Команда", COMMAND),
             ("Файл прогона", f"{RUN_FILE} (cycles/{CYCLE}/ под cache_root из lab_paths.json)"),
             ("Начат, UTC", record["started_utc"]), ("Закончен, UTC", record["finished_utc"]),
             ("Коммит core; дерево", f"{record['core_commit']}; {record['core_tree']}"),
             ("Вход: файл прогона E4, sha256", record["inputs"]["e4_run_sha256"]),
             ("Вход: реестр B10 и сетка B11, sha256", f"{record['inputs']['registry_sha256']}; {record['inputs']['tiles_sha256']}"),
             ("Версии", json.dumps(record["environment"], ensure_ascii=False)),
             *((f"{k} (до чисел)", v) for k, v in record["declared_before_numbers"].items()),
             ("Правило вывода (STUDY.md §6)", RULE)]
    lines += [(f"Вывод при T = 42 дБ: VVC / {NAMES[v['pair'].split('/')[1]]}", v["verdict"]) for v in a["verdict"]["pairs"]]
    lines += [("Вывод для всех трёх", a["verdict"]["all"]),
              ("Ожидание MGM (до чисел) и исход", f"{a['complexity']['expectation']['words']}; "
                                                  f"ρ = {a['complexity']['expectation']['rho']}, n = {a['complexity']['expectation']['n']}"),
              ("Точка JPEG PCC 50 — имя", a["jpeg50"]["name"]),
              ("Сверка шкалы IrfanView", a["fidler_scale"]["words"]),
              ("JPEG с optimize=True: восстановление равно E4", f"{a['huffman']['recon_equal']} из {a['huffman']['of']}"),
              e4_book.PCC_LINE,
              ("Где что", "«Сравнение» — тест по парам и T; «Пациенты» — медианы CR в пороге и r; «Единицы» — пороговый "
                          "PCC каждой единицы; «Сложность» — MGM, ρ и рассеяния; «JPEG 50», «JPEG Хаффман», «Снимки» — "
                          "описательно"),
              ("На пальцах", "Для каждого фрагмента и кодека найдено самое сильное сжатие, при котором PSNR-HVS-M ещё не "
                             "ниже порога; по каждому пациенту сравнено, во сколько раз сильнее сжимает VVC, чем другой "
                             "кодек; вывод «выгоднее» делается только по правилу, объявленному до чисел.")]
    return lines


# ---------------------------------------------------------------- the book

SHEETS = ("Описание", "Сравнение", "Пациенты", "Единицы", "Сложность", "JPEG 50", "JPEG Хаффман", "Снимки")
COMPARE_COLUMNS = (("T, дБ", "t_db"), ("Пара", "pair"), ("Уровень", "level"), ("Роль", "role"), ("n, шт", "n"),
                   ("Статус", "status"), ("Причина", "reason"), ("W⁺", "w_plus"), ("p", "p"), ("p по Холму", "p_holm"),
                   ("Эффект, раз", "ratio"), ("95 % ДИ от, раз", "ratio_low"), ("95 % ДИ до, раз", "ratio_high"),
                   ("ln-оценка HL", "hl"), ("k", "k"), ("Вывод по правилу", "verdict"))
UNIT_COLUMNS = (("Единица", "unit"), ("Уровень", "level"), ("Снимок", "image"), ("Кодек", "codec"), ("T, дБ", "t_db"),
                ("Судится", "judged"), ("Причина", "reason"), ("Пороговый PCC", "pcc"), ("Статус точки", "status"),
                ("CR в пороге, безразм.", "cr"), ("CR интерп., безразм.", "cr_interp"), ("Интерполяция", "interp_note"),
                ("CR_opt в пороге, безразм.", "cr_opt"), ("PSNR, дБ", "psnr_db"), ("PSNR-HVS-M, дБ", "psnr_hvs_m_db"),
                ("HaarPSI, безразм.", "haarpsi"), ("bpp, бит/пикс.", "bpp"), ("MGM, безразм.", "mgm"))
PATIENT_COLUMNS = (("Снимок", "image"), ("Кодек", "codec"), ("T, дБ", "t_db"), ("Судимо, шт", "judged"), ("Из, шт", "of"),
                   ("Медиана CR в пороге, безразм.", "median_cr"))
RATIO_COLUMNS = (("Снимок", "image"), ("Пара", "pair"), ("T, дБ", "t_db"), ("r, раз", "r"), ("ln r", "ln_r"),
                 ("В тесте", "in_test"), ("Причина", "reason"))
WHOLE_RATIO_COLUMNS = RATIO_COLUMNS + (("r по узлу VVC (40 и 41 дБ), раз", "r_node"),)
COUNT_COLUMNS = (("Уровень", "level"), ("Кодек", "codec"), ("T, дБ", "t_db"), ("Судимо, шт", "judged"), ("Из, шт", "of"),
                 ("Причины остальных", "reasons"))
FRAGMENT_COLUMNS = (("Единица", "unit"), ("Снимок", "image"), ("MGM, безразм.", "mgm"),
                    *((f"ln CR 42 дБ, {NAMES[c]}", f"ln_cr_{c}") for c in CODECS),
                    *((f"ln r, VVC / {NAMES[b]}", f"ln_r_{b}") for _, b in PAIRS),
                    *((f"Пороговый PCC, {NAMES[c]}", f"pcc_{c}") for c in CODECS))
PATIENT_MGM_COLUMNS = (("Снимок", "image"), ("MGM медиана, безразм.", "mgm"),
                       *((f"ln медианы CR 42 дБ, {NAMES[c]}", f"ln_cr_{c}") for c in CODECS),
                       *((f"ln r, VVC / {NAMES[b]}", f"ln_r_{b}") for _, b in PAIRS),
                       *((f"Медиана порогового PCC, {NAMES[c]}", f"pcc_{c}") for c in CODECS))
RHO_COLUMNS = (("Уровень", "level"), ("x", "x"), ("y", "y"), ("n, шт", "n"), ("ρ Спирмена", "rho"))
J50_COLUMNS = (("Единица", "unit"), ("Уровень", "level"), ("Снимок", "image"), ("Статус", "status"),
               ("PSNR-HVS-M, дБ", "psnr_hvs_m_db"), ("PSNR, дБ", "psnr_db"), ("HaarPSI, безразм.", "haarpsi"),
               ("CR, безразм.", "cr"), ("CR как у IrfanView, безразм.", "cr_opt"))
SHARE_COLUMNS = (("Уровень", "level"), ("T, дБ", "t_db"), ("n, шт", "n"), ("PSNR-HVS-M ≥ T, шт", "count"),
                 ("Доля, %", "share_pct"))
J50_PATIENT_COLUMNS = (("Снимок", "image"), ("n фрагментов, шт", "n"), ("Медиана PSNR-HVS-M, дБ", "median_psnr_hvs_m_db"),
                       ("Медиана PSNR, дБ", "median_psnr_db"), ("Медиана HaarPSI, безразм.", "median_haarpsi"),
                       ("Медиана CR, безразм.", "median_cr"), ("Медиана CR как у IrfanView, безразм.", "median_cr_opt"))
HUFF_COLUMNS = (("Единица", "unit"), ("PCC", "pcc"), ("Файл, байт", "file_bytes"), ("Файл opt, байт", "file_bytes_opt"),
                ("CR, безразм.", "cr"), ("CR_opt, безразм.", "cr_opt"), ("CR_opt / CR, безразм.", "ratio"),
                ("Восстановление = E4", "recon_equal"))
QUART_COLUMNS = (("Единицы", "what"), ("n, шт", "n"), ("Медиана", "median"), ("q25", "q25"), ("q75", "q75"),
                 ("Мин.", "min"), ("Макс.", "max"))


def put_table(cells, sheet, top, left, columns, records):
    """A table: header row, the key row of the run file under it, then the records."""
    for j, (head, key) in enumerate(columns):
        cells[sheet, top, left + j], cells[sheet, top + 1, left + j] = head, key
    for i, rec in enumerate(records):
        for j, (_, key) in enumerate(columns):
            value = rec.get(key)
            if value is not None:
                cells[sheet, top + 2 + i, left + j] = value
    return top + 2, top + 1 + len(records)  # the first and the last data row


def book_cells(record, run_sha256):
    """All cells of the book, {(sheet, row, column): value}; positions of the chart data."""
    a, cells, where = record["analysis"], {}, {}
    for i, (k, v) in enumerate([("sha256 файла прогона", run_sha256), *record["description"]], start=1):
        cells["Описание", i, 1], cells["Описание", i, 2] = k, v
    cells["Сравнение", 1, 1], cells["Сравнение", 1, 2] = "Правило вывода", RULE
    for i, v in enumerate(a["verdict"]["pairs"], start=2):
        cells["Сравнение", i, 1], cells["Сравнение", i, 2] = f"T = 42 дБ, VVC / {NAMES[v['pair'].split('/')[1]]}", v["verdict"]
    cells["Сравнение", 5, 1], cells["Сравнение", 5, 2] = "Для всех трёх", a["verdict"]["all"]
    put_table(cells, "Сравнение", 7, 1, COMPARE_COLUMNS, a["comparisons"])
    top = 9 + len(a["comparisons"]) + 2
    put_table(cells, "Сравнение", top, 1, COUNT_COLUMNS, a["counts"])
    put_table(cells, "Пациенты", 1, 1, PATIENT_COLUMNS, a["patients"])
    put_table(cells, "Пациенты", 1, len(PATIENT_COLUMNS) + 2, RATIO_COLUMNS, a["ratios"])
    put_table(cells, "Единицы", 1, 1, UNIT_COLUMNS, a["units"])
    c = a["complexity"]
    cells["Сложность", 1, 1], cells["Сложность", 1, 2] = "Ожидание до чисел и исход", c["expectation"]["words"]
    where["fragments"] = put_table(cells, "Сложность", 3, 1, FRAGMENT_COLUMNS, c["fragments"])
    left = len(FRAGMENT_COLUMNS) + 2
    where["patients"] = put_table(cells, "Сложность", 3, left, PATIENT_MGM_COLUMNS, c["patients"])
    where["patients_left"] = left
    put_table(cells, "Сложность", 3 + 2 + len(c["patients"]) + 2, left, RHO_COLUMNS, c["rho"])
    j = a["jpeg50"]
    for i, (k, v) in enumerate([("Имя точки", j["name"]), ("CR", j["note"]),
                                ("Сверка шкалы IrfanView", json.dumps({k: v for k, v in a["fidler_scale"].items()
                                                                       if not k.startswith("table_")}, ensure_ascii=False))],
                               start=1):
        cells["JPEG 50", i, 1], cells["JPEG 50", i, 2] = k, v
    put_table(cells, "JPEG 50", 5, 1, SHARE_COLUMNS, j["shares"])
    put_table(cells, "JPEG 50", 5, len(SHARE_COLUMNS) + 2, J50_PATIENT_COLUMNS, j["patients"])
    put_table(cells, "JPEG 50", 5 + 2 + max(len(j["shares"]), len(j["patients"])) + 2, 1, J50_COLUMNS, j["units"])
    h = a["huffman"]
    cells["JPEG Хаффман", 1, 1], cells["JPEG Хаффман", 1, 2] = "Восстановление = E4", f"{h['recon_equal']} из {h['of']}"
    put_table(cells, "JPEG Хаффман", 3, 1, QUART_COLUMNS,
              [{"what": "CR_opt / CR, все", **h["ratio"]}, {"what": "CR_opt / CR, фрагменты", **h["ratio_fragments"]},
               {"what": "CR_opt / CR, снимки", **h["ratio_images"]}])
    put_table(cells, "JPEG Хаффман", 9, 1, HUFF_COLUMNS, h["rows"])
    whole = [u for u in a["units"] if u["level"] == "m"]
    put_table(cells, "Снимки", 1, 1, UNIT_COLUMNS, whole)
    put_table(cells, "Снимки", 1, len(UNIT_COLUMNS) + 2, WHOLE_RATIO_COLUMNS, a["whole_ratios"])
    put_table(cells, "Снимки", 3 + len(a["whole_ratios"]) + 2, len(UNIT_COLUMNS) + 2, COMPARE_COLUMNS, a["whole_comparisons"])
    return cells, where


def _points(ws, x_col, y_col, first, last, title, style):
    """A scatter series of points: the colour and the marker of the style table, no line."""
    from openpyxl.chart import Reference, Series
    from openpyxl.chart.marker import Marker
    from openpyxl.chart.shapes import GraphicalProperties
    from openpyxl.drawing.line import LineProperties

    color, _, marker = style
    s = Series(Reference(ws, min_col=y_col, min_row=first, max_row=last),
               Reference(ws, min_col=x_col, min_row=first, max_row=last), title=title)
    s.graphicalProperties = GraphicalProperties(ln=LineProperties(noFill=True))
    s.marker = Marker(symbol=marker, size=5)
    s.marker.graphicalProperties = GraphicalProperties(solidFill=color, ln=LineProperties(solidFill=color))
    s.smooth = False
    return s


def add_charts(ws, where):
    """«Сложность»: MGM - ln CR (four codecs) and MGM - ln r (three pairs), on fragments and on patients; the charts
    stand below the table of fragments."""
    below = where["fragments"][1] + 4
    for k, (name, (first, last), left) in enumerate((("фрагменты", where["fragments"], 1),
                                                     ("пациенты", where["patients"], where["patients_left"]))):
        cols = PATIENT_MGM_COLUMNS if name == "пациенты" else FRAGMENT_COLUMNS
        index = {key: left + i for i, (_, key) in enumerate(cols)}
        chart = e4_book._scatter(f"MGM — ln CR в пороге 42 дБ, {name}", "MGM, безразм.", "ln CR")
        for c in CODECS:
            chart.series.append(_points(ws, index["mgm"], index[f"ln_cr_{c}"], first, last, NAMES[c], e4_book.CODEC_STYLE[c]))
        ws.add_chart(chart, f"A{below + 18 * k}")
        chart = e4_book._scatter(f"MGM — ln r в пороге 42 дБ, {name}", "MGM, безразм.", "ln r")
        for _, b in PAIRS:
            chart.series.append(_points(ws, index["mgm"], index[f"ln_r_{b}"], first, last, f"VVC / {NAMES[b]}",
                                        e4_book.CODEC_STYLE[b]))
        ws.add_chart(chart, f"K{below + 18 * k}")


def build_book(run_path, book_path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    run_bytes = Path(run_path).read_bytes()
    record = json.loads(run_bytes.decode("utf-8"))
    cells, where = book_cells(record, hashlib.sha256(run_bytes).hexdigest())
    wb = Workbook()
    wb.remove(wb.active)
    for name in SHEETS:
        wb.create_sheet(name)
    for (sheet, r, c), v in cells.items():
        wb[sheet].cell(row=r, column=c, value=v)
    add_charts(wb["Сложность"], where)
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
    cells, _ = book_cells(json.loads(run_bytes.decode("utf-8")), hashlib.sha256(run_bytes).hexdigest())
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


def summary_lines(record):
    """What the console prints after the numbers, with n beside each."""
    a = record["analysis"]
    out = [f"Rule (STUDY.md section 6): {RULE}"]
    for c in a["comparisons"]:
        if c["t_db"] == MAIN_T and c["status"] == "valid":
            out.append(f"  {c['pair']} T 42 dB [{c['role']}]: n = {c['n']} patients, effect {c['ratio']:.4f} x "
                       f"(95 % CI {c['ratio_low']:.4f} - {c['ratio_high']:.4f}), p = {c['p']:.6g}, "
                       f"p Holm = {c['p_holm'] if c['p_holm'] is None else format(c['p_holm'], '.6g')}"
                       + (f" -> {c['verdict']}" if c["verdict"] else ""))
        elif c["t_db"] == MAIN_T:
            out.append(f"  {c['pair']} T 42 dB: no result ({c['reason']}), n = {c['n']}")
    out.append(f"  {a['verdict']['all']}")
    out.append(f"  MGM: {a['complexity']['expectation']['words']} (rho = {a['complexity']['expectation']['rho']}, "
               f"n = {a['complexity']['expectation']['n']})")
    out.append(f"  JPEG PCC 50: {a['jpeg50']['name']}; IrfanView: {a['fidler_scale']['words']}")
    out.append(f"  Huffman: reconstruction equal {a['huffman']['recon_equal']} of {a['huffman']['of']}; CR_opt / CR "
               f"median {a['huffman']['ratio']['median']:.4f} (n = {a['huffman']['ratio']['n']})")
    return out


_MISSING = object()


def regression(analysis, baseline):
    """Paths of the fields of `analysis` that differ from the analysis of cycle 15; one outside whole_ratios and
    whole_comparisons at 40 and 41 dB is a refusal in words (the regression declared for cycle 17)."""
    new, found = json.loads(json.dumps(e4_book.jsonable(analysis), ensure_ascii=False)), []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                walk(a.get(k, _MISSING), b.get(k, _MISSING), f"{path}.{k}" if path else k)
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif a is _MISSING or b is _MISSING or a != b:
            found.append(path)

    def allowed(path):
        head, sep, rest = path.partition("[")
        if head not in CHANGED or not sep:
            return False
        i = int(rest.split("]", 1)[0])
        return new[head][i]["t_db"] in (40.0, 41.0) and baseline[head][i]["t_db"] in (40.0, 41.0)

    walk(new, baseline, "")
    other = [p for p in found if not allowed(p)]
    if other:
        raise RunRefusal(f"E5 differs from cycle 15 outside {' and '.join(CHANGED)} at 40 and 41 dB in {len(other)} "
                         f"fields: {other[:8]}")
    return found


def run(e4_path, data_root, tiles_path, images_root, scale_folder, out_dir, state, e4_pin=E4_RUN_SHA256,
        registry_pins=(run_e4.REGISTRY_SHA256, run_e4.TILES_SHA256), log=print, baseline=None):
    out_dir = Path(out_dir)
    run_path, book_path = out_dir / RUN_FILE, out_dir / BOOK_FILE
    for path in (run_path, book_path):
        if path.exists():
            raise RunRefusal(f"{path} already exists: a run file and a book of E5 are never overwritten")
    if not Path(e4_path).is_file():
        raise RunRefusal(f"the run file of E4 {e4_path} does not exist")
    e4_sha = sha256_file(e4_path)
    if e4_sha != e4_pin:
        raise RunRefusal(f"the run file of E4 has sha256 {e4_sha}, E5 is declared for {e4_pin}")
    try:
        check_versions("JPEG", jpeg.EXPECTED_VERSIONS, jpeg.versions())
    except RuntimeError as exc:
        raise RunRefusal(str(exc)) from exc
    started = datetime.datetime.now(datetime.timezone.utc)
    log(f"{COMMAND}\n{QUESTION}\nDeclared before the numbers:")
    for key, text in DECLARED.items():
        log(f"  [{key}] {text}")
    e4 = json.loads(Path(e4_path).read_text(encoding="utf-8"))
    registry, tile_rows, tile = run_e4.load_inputs(data_root, tiles_path, pins=registry_pins)
    units = run_e4.load_units(data_root, registry, tile_rows, tile)
    levels = {u: i["level"] for u, i in units.items()}
    if levels != e4["units"]:
        raise RunRefusal("the units of the copy and the grid differ from the units of the E4 run file")
    image_of = {u: (u if i["level"] == "m" else i["image"]) for u, i in units.items()}
    (frisco,) = load_test_images(images_root, [FIDLER_IMAGE], (512, 512))
    arrays = {u: np.ascontiguousarray(i["array"]) for u, i in units.items()}
    log("  E4 rows, the copy and the grid read; re-encoding JPEG with optimize=True, MGM, analysis")
    analysis = analyse(e4, levels, image_of, arrays, frisco["array"], scale_folder)
    checked = None
    if baseline is not None:  # (run file of cycle 15, its pinned sha256)
        if not Path(baseline[0]).is_file() or sha256_file(baseline[0]) != baseline[1]:
            raise RunRefusal(f"the run file of cycle 15 {baseline[0]} is missing or its sha256 is not {baseline[1]}")
        differences = regression(analysis, json.loads(Path(baseline[0]).read_text(encoding="utf-8"))["analysis"])
        checked = {"baseline": "cycles/15/e5_analysis.json", "baseline_sha256": baseline[1], "differences": differences,
                   "rule": "равны все поля analysis, кроме whole_ratios и whole_comparisons при t_db 40 и 41"}
        log(f"  regression against cycle 15: {len(differences)} fields differ, all in whole_ratios and "
            "whole_comparisons at 40 and 41 dB")
    record = {"run": RUN_FILE, "study": "P001", "cycle": CYCLE, "question": QUESTION, "command": COMMAND,
              "started_utc": started.isoformat(timespec="seconds"),
              "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), **state,
              "declared_before_numbers": DECLARED, "thresholds_db": list(THRESHOLDS),
              "inputs": {"e4_run": "cycles/13/e4_main.json", "e4_run_sha256": e4_sha,
                         "registry_sha256": registry_pins[0], "tiles_sha256": registry_pins[1],
                         "fidler_image": f"BMP/{FIDLER_IMAGE}", "fidler_image_sha256": frisco["source_sha256"]},
              "environment": {"python": platform.python_version(), "imstudies": imstudies.__version__,
                              "numpy": np.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__,
                              **{f"jpeg_{k}": v for k, v in jpeg.versions().items()}},
              "machine": e4_book.machine(), "analysis": analysis, "regression": checked}
    record["description"] = description(record)
    record["summary_lines"] = summary_lines(record)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(run_path, "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(e4_book.jsonable(record), ensure_ascii=False, indent=1, allow_nan=False))
    build_book(run_path, book_path)
    for line in record["summary_lines"]:
        log(line)
    log(f"Run file: {run_path} (sha256 {sha256_file(run_path)})\nBook: {book_path} (sha256 {sha256_file(book_path)}), "
        "checked cell by cell")
    return run_path, book_path


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace", line_buffering=True)
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    try:
        paths = run_e4.load_paths()
        raw = json.loads((ROOT / "lab_paths.json").read_text(encoding="utf-8"))
        if not raw.get("images_root"):
            raise RunRefusal("lab_paths.json has no key 'images_root'")
        cycles = paths["cache_root"] / "cycles"
        run(cycles / "13" / run_e4.RUN_FILE, paths["data_root"], cycles / "9" / "p001_tiles.json",
            (ROOT / raw["images_root"]).resolve(), paths["cache_root"].parent / "fidler_scale", cycles / str(CYCLE),
            git_state(ROOT), baseline=(cycles / "15" / RUN_FILE, BASELINE_SHA256))
    except RunRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
