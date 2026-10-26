"""E6 of P001: at the threshold PCC of PSNR-HVS-M 42 dB, do the four codecs look like the original to the owner's eye at 1 : 1 - pages of 14 fragments chosen by the declared rule? Run: python studies/P001_vvc-dental/run_e6.py   (--rebuild-book: the book again, carrying the owner's verdicts)

Study P001, STUDY.md section 6 («Объявление E6»); ../REFERENCE.md#lab-pages. Inputs, each by its pinned sha256:
the run file of E5 <cache_root>/cycles/15/e5_analysis.json (features of the fragments, threshold PCC at 42 dB), the
run file of E4 cycles/13/e4_main.json (the points), the copy data/P001 and the grid of cycle 9 (B10, B11). The 14
fragments are chosen by imstudies.lab.p001_e6.select; every compressed panel is encoded again with the wrappers and
arguments of E4 (B05 + B09, B06, B07, B12) and must equal the E4 row by sha256 of the file and of the
reconstruction - otherwise a refusal in words. Writes into <cache_root>/cycles/16/: the run file e6_pages.json,
14 pages e6_<fragment>.png (8-bit grayscale, panels 1 : 1), pages.csv (a machine copy, no verdict) and the book
e6_pages.xlsx - the only place of the owner's verdict (a drop-down list of four words), every other cell checked
against the run file. Pages are medical images: they stay in the cache, outside Git. No pixel in the run file.
"""

import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import imstudies
from imstudies.lab import p001_e6
from imstudies.lab.coders import avif_libheif, heif, jpeg, vtm, vvc_heif
from imstudies.lab.runs import RunRefusal, git_state, sha256_file

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
_spec = importlib.util.spec_from_file_location("run_e4", STUDY / "run_e4.py")
run_e4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_e4)
e4_book = run_e4.e4_book

COMMAND = "python studies/P001_vvc-dental/run_e6.py"
QUESTION = ("E6, P001: в пороговом PCC при PSNR-HVS-M 42 дБ выглядят ли четыре кодека как исходный фрагмент для глаза "
            "владельца при масштабе 1 : 1 — страницы 14 фрагментов, выбранных объявленным правилом?")
CYCLE = 16
RUN_FILE, BOOK_FILE, CSV_FILE, WORK_DIR = "e6_pages.json", "e6_pages.xlsx", "pages.csv", "e6_work"
E5_SHA256 = "be161b1e46b3d81c62c22c859ed4840b53f4e4d99b264a181070d7ab7e7dc16a"  # cycles/15/e5_analysis.json (A13)
E4_SHA256 = "44a1733512fdfbb9d02d210ff81d492b9a5d929ec8d547d84676d47350875f2a"  # cycles/13/e4_main.json (A12)
N_PAGES = 14
VERDICTS = ("ок", "не тот класс", "ложная тревога", "пропуск")  # LAB-PAGES 6
DECLARED = {
    "choice": ("14 фрагментов из 154 по признакам analysis.complexity.fragments файла E5, списки по порядку: (1) 3 с "
               "наибольшим MGM; (2) 3 с наименьшим MGM; (3) 3 с наибольшим ln CR VVC в пороге; (4) 3 с наибольшим ln r "
               "VVC / JPEG; (5) 2 из оставшихся — numpy.random.default_rng(20261024).choice(<коды по возрастанию>, 2, "
               "replace=False). Фрагмент, выбранный раньше, пропускается — берётся следующий в том же списке; при "
               "равенстве — меньший код."),
    "points": ("Пороговый PCC каждого кодека — запись analysis.units файла E5 при t_db = 42.0; точка — строка E4 с этим "
               "PCC. Каждая сжатая панель перекодируется теми же обёртками и аргументами, что в E4 (B05 + B09, B06, B07, "
               "B12); sha256 файла и восстановления = строке E4, иначе отказ словами."),
    "page": ("PNG, 8 бит, полутон; панели по порядку — исходный, VVC, JPEG, HEIF, AVIF; масштаб 1 : 1 — пиксель панели "
             "равен пикселю фрагмента, без интерполяции, сглаживания и растяжения уровней; над панелью — кодек, PCC, "
             "PSNR-HVS-M, CR."),
    "claim": ("Утверждение — одна фраза из строки файла: фрагмент и снимок, список и значение признака; по каждому кодеку "
              "— PCC, PSNR-HVS-M в дБ, CR; вывод — «у всех четырёх кодеков PSNR-HVS-M не ниже 42 дБ — по метрике "
              "искажения незаметны» (иначе — названы кодеки ниже порога). Сверяется с полями строки по одному."),
    "verdict": ("Вердикт — лист «Страницы», выпадающий список: ок · не тот класс · ложная тревога · пропуск; «пропуск» — "
                "кодек и место в «Примечании». Взгляд — контроль, не измерение: числа E5 не меняет."),
}


def encode(paths, codec, image, pcc, work):
    """One point with the coder of E4 - B05 + B09 for VVC, B06, B07, B12 - and its arguments; (stored file,
    reconstruction). A module name so that a test can put a coder of its own in its place."""
    timeout = run_e4.TIMEOUT_S["f"]
    if codec == "vtm":
        coded = vtm.encode_decode(image, pcc, max_pixels=image.size, level=e4_book.LEVEL_VTM,
                                  frame_rate=e4_book.FRAME_RATE_VTM, timeout_s=timeout, encoder=paths["encoder"],
                                  decoder=paths["decoder"], cfg=paths["cfg"], work_root=work)
        packed = vvc_heif.pack(coded.bitstream, coded.recon, timeout_s=timeout, mp4box=paths["mp4box"],
                               gpac=paths["gpac"], decoder=paths["decoder"], work_root=work)
        return packed.file, coded.recon
    if codec == "avif":
        (coded,) = avif_libheif.encode_decode(image, [pcc], max_pixels=image.size, timeout_s=timeout,
                                              python=paths["avif_libheif_python"], work_root=work)
        return coded.file, coded.recon
    coded = {"jpeg": jpeg, "heif": heif}[codec].encode_decode(image, pcc, max_pixels=image.size)
    return coded.file, coded.recon


def load_pinned(path, pin, what):
    if not Path(path).is_file():
        raise RunRefusal(f"the run file of {what} {path} does not exist")
    sha = sha256_file(path)
    if sha != pin:
        raise RunRefusal(f"the run file of {what} has sha256 {sha}, E6 is declared for {pin}")
    return json.loads(Path(path).read_text(encoding="utf-8")), sha


def pages_of(picks, e5, e4, arrays, paths, work, out_dir):
    """Every page: its points re-encoded and checked against E4, its claim checked against its entry, its PNG."""
    at = {(u["unit"], u["codec"]): u for u in e5["analysis"]["units"] if u["t_db"] == p001_e6.THRESHOLD_DB}
    rows = {(r["unit"], r["codec"], r["param"]): r for r in e4["rows"]}
    pages, table = [], []
    for pick in picks:
        image = np.ascontiguousarray(arrays[pick["unit"]])
        page = {"file": f"e6_{pick['unit']}.png", **pick, "codecs": []}
        recons = []
        for codec in p001_e6.CODECS:
            unit = at[(pick["unit"], codec)]
            if not unit["judged"]:
                raise RunRefusal(f"{pick['unit']}, {p001_e6.NAMES[codec]}: not judged at 42 dB ({unit['reason']}) - "
                                 "the page has no point to show")
            row = rows[(pick["unit"], codec, unit["pcc"])]
            file, recon = encode(paths, codec, image, unit["pcc"], work)
            file_equal = hashlib.sha256(file).hexdigest() == row["file_sha256"]
            recon_equal = hashlib.sha256(np.ascontiguousarray(recon).tobytes()).hexdigest() == row["recon_sha256"]
            if not (file_equal and recon_equal):
                raise RunRefusal(f"{pick['unit']}, {p001_e6.NAMES[codec]} PCC {unit['pcc']}: encoded again, the "
                                 f"{'file' if not file_equal else 'reconstruction'} differs from E4 by sha256")
            point = {"unit": pick["unit"], "image": pick["image"], "list": pick["list"], "feature": pick["feature"],
                     "value": pick["value"], "codec": codec, "pcc": unit["pcc"], "status": row["status"],
                     "psnr_hvs_m_db": row["psnr_hvs_m_db"], "psnr_db": row["psnr_db"], "haarpsi": row["haarpsi"],
                     "cr": row["cr_file"], "bpp": row["bpp"], "file_sha256_equal": file_equal,
                     "recon_sha256_equal": recon_equal}
            page["codecs"].append(point)
            table.append(point)
            recons.append(np.ascontiguousarray(recon))
        page["title"] = p001_e6.title(page)
        page["claim"] = p001_e6.statement(page)
        p001_e6.check_statement(page["claim"], page)
        page["explanation"] = list(p001_e6.EXPLANATION)
        page["labels"] = [list(pair) for pair in p001_e6.panel_labels(page)]
        p001_e6.draw_page(image, recons, page["labels"], page["title"], page["claim"], out_dir / page["file"])
        pages.append(page)
    return pages, table


def run(e5_path, e4_path, data_root, tiles_path, out_dir, state, paths, pins=(E5_SHA256, E4_SHA256),
        registry_pins=(run_e4.REGISTRY_SHA256, run_e4.TILES_SHA256), log=print):
    out_dir = Path(out_dir)
    for name in (RUN_FILE, BOOK_FILE, CSV_FILE):
        if (out_dir / name).exists():
            raise RunRefusal(f"{out_dir / name} already exists: the pages of cycle {CYCLE} are never overwritten")
    e5, e5_sha = load_pinned(e5_path, pins[0], "E5")
    e4, e4_sha = load_pinned(e4_path, pins[1], "E4")
    started = datetime.datetime.now(datetime.timezone.utc)
    log(f"{COMMAND}\n{QUESTION}\nDeclared before viewing:")
    for key, text in DECLARED.items():
        log(f"  [{key}] {text}")
    picks = p001_e6.select(e5["analysis"]["complexity"]["fragments"])
    if len({p["unit"] for p in picks}) != N_PAGES or len(picks) != N_PAGES:
        raise RunRefusal(f"the choice gave {len(picks)} pages, {len({p['unit'] for p in picks})} different - "
                         f"{N_PAGES} different are declared")
    registry, tile_rows, tile = run_e4.load_inputs(data_root, tiles_path, pins=registry_pins)
    units = run_e4.load_units(data_root, registry, tile_rows, tile)
    arrays = {p["unit"]: units[p["unit"]]["array"] for p in picks}
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / WORK_DIR
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    try:
        pages, table = pages_of(picks, e5, e4, arrays, paths, work, out_dir)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    record = {"run": RUN_FILE, "study": "P001", "cycle": CYCLE, "question": QUESTION, "command": COMMAND,
              "started_utc": started.isoformat(timespec="seconds"),
              "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), **state,
              "declared_before_viewing": DECLARED,
              "inputs": {"e5_run": "cycles/15/e5_analysis.json", "e5_run_sha256": e5_sha, "e4_run": "cycles/13/e4_main.json",
                         "e4_run_sha256": e4_sha, "registry_sha256": registry_pins[0], "tiles_sha256": registry_pins[1]},
              "environment": {"imstudies": imstudies.__version__, "numpy": np.__version__},
              "selection": picks, "table": table, "pages": pages,
              "equal": {"points": len(table), "file_and_recon_equal": sum(p["file_sha256_equal"] and p["recon_sha256_equal"]
                                                                          for p in table)}}
    with open(out_dir / RUN_FILE, "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(e4_book.jsonable(record), ensure_ascii=False, indent=1, allow_nan=False))
    write_csv(out_dir / RUN_FILE, out_dir / CSV_FILE)
    build_book(out_dir / RUN_FILE, out_dir / BOOK_FILE)
    for p in pages:
        log(f"  {p['file']}: list {p['list']} - {p001_e6.feature_text(p)}")
    log(f"Encoded again and equal to E4 by sha256: {record['equal']['file_and_recon_equal']} of {record['equal']['points']}")
    log(f"Run file: {out_dir / RUN_FILE} (sha256 {sha256_file(out_dir / RUN_FILE)})\nBook: {out_dir / BOOK_FILE} "
        f"(sha256 {sha256_file(out_dir / BOOK_FILE)}), checked cell by cell; pages.csv sha256 {sha256_file(out_dir / CSV_FILE)}")
    return out_dir / RUN_FILE, out_dir / BOOK_FILE


CSV_COLUMNS = ("file", "unit", "image", "list", "feature", "value", "claim")


def write_csv(run_path, csv_path):
    """The machine copy of the page list beside the pictures - no verdict (LAB-PAGES 5)."""
    record = json.loads(Path(run_path).read_text(encoding="utf-8"))
    with open(csv_path, "x", encoding="utf-8", newline="") as f:
        out = csv.writer(f)
        out.writerow(CSV_COLUMNS)
        for p in record["pages"]:
            out.writerow(["" if p[k] is None else p[k] for k in CSV_COLUMNS])


# ---------------------------------------------------------------- the book

TABLE = (("Фрагмент", "unit"), ("Снимок", "image"), ("Список", "list"), ("Признак", "feature"),
         ("Значение признака, безразм.", "value"), ("Кодек", "codec"), ("Пороговый PCC", "pcc"), ("Статус точки", "status"),
         ("PSNR-HVS-M, дБ", "psnr_hvs_m_db"), ("PSNR, дБ", "psnr_db"), ("HaarPSI, безразм.", "haarpsi"),
         ("CR, безразм.", "cr"), ("bpp, бит/пикс.", "bpp"), ("sha256 файла = E4", "file_sha256_equal"),
         ("sha256 восстановления = E4", "recon_sha256_equal"))
PAGES = (("Страница", "file"), ("Фрагмент", "unit"), ("Список", "list"), ("Утверждение", "claim"),
         ("На пальцах", "explanation"))
VERDICT_COLUMN, NOTE_COLUMN = len(PAGES) + 1, len(PAGES) + 2


def book_cells(record, run_sha256):
    i = record["inputs"]
    lines = [("Вопрос", record["question"]), ("Команда", record["command"]),
             ("Файл прогона", f"{record['run']} (cycles/{record['cycle']}/ под cache_root из lab_paths.json)"),
             ("sha256 файла прогона", run_sha256), ("Начат, UTC", record["started_utc"]),
             ("Коммит core; дерево", f"{record['core_commit']}; {record['core_tree']}"),
             ("Вход: файл E5, sha256", f"{i['e5_run']}; {i['e5_run_sha256']}"),
             ("Вход: файл E4, sha256", f"{i['e4_run']}; {i['e4_run_sha256']}"),
             ("Вход: реестр B10 и сетка B11, sha256", f"{i['registry_sha256']}; {i['tiles_sha256']}"),
             *((f"{k} (до просмотра)", v) for k, v in record["declared_before_viewing"].items()),
             ("Перекодировано, sha256 = E4", f"{record['equal']['file_and_recon_equal']} из {record['equal']['points']}"),
             ("Как смотреть", "Открывайте страницу гиперссылкой на листе «Страницы» и смотрите при масштабе 100 % (1 : 1): "
                              "иначе программа просмотра сама уменьшит или сгладит панели."),
             ("Слова вердикта", " · ".join(VERDICTS) + " (LAB-PAGES, п. 6); вердикт — только на листе «Страницы»"),
             ("Ключи листов", "под шапкой — имя поля файла прогона: «Таблица» — table[], «Страницы» — pages[]; колонки "
                              "«Вердикт» и «Примечание» — владельца, в файле прогона их нет"),
             ("На пальцах", "Для 14 фрагментов, выбранных правилом до просмотра, рядом показаны исходный фрагмент и он же "
                            "после каждого из четырёх кодеков в том сжатии, где метрика ещё не ниже порога 42 дБ; "
                            "глаз проверяет, не видно ли там отличий.")]
    cells = {}
    for r, (k, v) in enumerate(lines, start=1):
        cells["Описание", r, 1], cells["Описание", r, 2] = k, v

    def table(sheet, columns, records):
        for j, (head, key) in enumerate(columns, start=1):
            cells[sheet, 1, j], cells[sheet, 2, j] = head, key
        for r, rec in enumerate(records, start=3):
            for j, (_, key) in enumerate(columns, start=1):
                value = rec.get(key)
                if value is not None:
                    cells[sheet, r, j] = "\n".join(value) if isinstance(value, list) else value

    table("Таблица", TABLE, record["table"])
    table("Страницы", PAGES, record["pages"])
    cells["Страницы", 1, VERDICT_COLUMN], cells["Страницы", 2, VERDICT_COLUMN] = "Вердикт", "владелец"
    cells["Страницы", 1, NOTE_COLUMN], cells["Страницы", 2, NOTE_COLUMN] = "Примечание", "владелец"
    return cells


def build_book(run_path, book_path, verdicts=None):
    """The book from the run file; `verdicts` - {page file: (verdict, note)} carried from a previous book."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    run_bytes = Path(run_path).read_bytes()
    record = json.loads(run_bytes.decode("utf-8"))
    wb = Workbook()
    wb.remove(wb.active)
    for name in ("Описание", "Таблица", "Страницы"):
        wb.create_sheet(name)
    for (sheet, r, c), v in book_cells(record, hashlib.sha256(run_bytes).hexdigest()).items():
        wb[sheet].cell(row=r, column=c, value=v)
    pages = wb["Страницы"]
    for r, page in enumerate(record["pages"], start=3):
        pages.cell(row=r, column=1).hyperlink = page["file"]
        verdict, note = (verdicts or {}).get(page["file"], (None, None))
        pages.cell(row=r, column=VERDICT_COLUMN, value=verdict)
        pages.cell(row=r, column=NOTE_COLUMN, value=note)
        for c in (4, 5):
            pages.cell(row=r, column=c).alignment = Alignment(wrap_text=True, vertical="top")
    last = 2 + len(record["pages"])
    letter = e4_book._col(VERDICT_COLUMN)
    dv = DataValidation(type="list", formula1='"' + ",".join(VERDICTS) + '"', allow_blank=True, showErrorMessage=True,
                        errorTitle="Вердикт", error="Только одно из четырёх слов: " + ", ".join(VERDICTS))
    pages.add_data_validation(dv)
    dv.add(f"{letter}3:{letter}{last}")
    wb["Описание"].column_dimensions["A"].width = 36
    wb["Описание"].column_dimensions["B"].width = 140
    for row in wb["Описание"].iter_rows(min_col=2, max_col=2):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    for name, widths in (("Страницы", {"A": 24, "D": 90, "E": 70, letter: 16, e4_book._col(NOTE_COLUMN): 40}),
                         ("Таблица", {"A": 14, "D": 34})):
        for c in wb[name][1]:
            c.font = Font(bold=True)
        wb[name].freeze_panes = "A3"
        for col, width in widths.items():
            wb[name].column_dimensions[col].width = width
    wb.save(book_path)
    check_book(run_path, book_path)


def check_book(run_path, book_path):
    """Every cell equals the run-file value; only the verdict and note columns may hold the owner's text; the verdict
    column carries the drop-down list of the four words."""
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
        extra = {e for e in extra - set(cells) if not (sheet == "Страницы" and e[2] in (VERDICT_COLUMN, NOTE_COLUMN)
                                                      and e[1] >= 3)}
        if extra:
            raise RunRefusal(f"book sheet {sheet} has cells not taken from the run file: {sorted(extra)[:3]}")
    lists = [dv for dv in book["Страницы"].data_validations.dataValidation if dv.type == "list"]
    letter = e4_book._col(VERDICT_COLUMN)
    if not any(dv.formula1 == '"' + ",".join(VERDICTS) + '"' and f"{letter}3" in str(dv.sqref) for dv in lists):
        raise RunRefusal("the verdict column has no drop-down list of the four words")


def read_verdicts(book_path):
    """The owner's verdicts and notes of an existing book by page file."""
    from openpyxl import load_workbook

    sheet = load_workbook(book_path)["Страницы"]
    out = {}
    for r in range(3, sheet.max_row + 1):
        verdict, note = sheet.cell(row=r, column=VERDICT_COLUMN).value, sheet.cell(row=r, column=NOTE_COLUMN).value
        if verdict not in (None, "") or note not in (None, ""):
            out[sheet.cell(row=r, column=1).value] = (verdict, note)
    return out


def rebuild_book(run_path, book_path):
    """The book again from the run file, carrying verdicts and notes; a page with a verdict that is gone from the run
    file or from the disk is refused by name."""
    verdicts = read_verdicts(book_path) if Path(book_path).is_file() else {}
    pages = {p["file"] for p in json.loads(Path(run_path).read_text(encoding="utf-8"))["pages"]}
    gone = sorted(set(verdicts) - pages)
    if gone:
        raise RunRefusal(f"pages with a verdict are gone from the run file: {gone}")
    missing = sorted(f for f in verdicts if not (Path(run_path).parent / f).is_file())
    if missing:
        raise RunRefusal(f"pages with a verdict are gone from the disk: {missing}")
    build_book(run_path, book_path, verdicts)
    return verdicts


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild-book", action="store_true", help="rebuild the book, carrying the owner's verdicts")
    args = parser.parse_args(argv)
    try:
        paths = run_e4.load_paths()
        cycles = paths["cache_root"] / "cycles"
        out_dir = cycles / str(CYCLE)
        if args.rebuild_book:
            if not (out_dir / RUN_FILE).is_file():
                raise RunRefusal(f"no run file to rebuild the book from: {out_dir / RUN_FILE}")
            carried = rebuild_book(out_dir / RUN_FILE, out_dir / BOOK_FILE)
            print(f"Book rebuilt and checked cell by cell, {len(carried)} verdicts carried: {out_dir / BOOK_FILE}")
            return 0
        run(cycles / "15" / "e5_analysis.json", cycles / "13" / run_e4.RUN_FILE, paths["data_root"],
            cycles / "9" / "p001_tiles.json", out_dir, git_state(ROOT), paths)
    except RunRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
