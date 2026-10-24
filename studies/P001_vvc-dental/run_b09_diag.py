"""B09 diagnosis, P001: why does MP4Box not pack into HEIF the VVC stream of fragment D05_r00_c01 at PCC 46 - in that stream or in the state of the machine? Run: python studies/P001_vvc-dental/run_b09_diag.py

Cycle 14; owner decision 2026-10-24 (message before 12:04 UTC): the pixels of this one fragment may be read outside
E4, for this diagnosis only; nothing found here goes into E4. Inputs: the journal of cycle 13 (pinned sha256, read
only), the copy data/P001/D05.png (png_sha256 of the registry) cut by its tile of the grid of cycle 9; B05 and B09
with the paths of lab_paths.json, their sha256 equal to the header of the journal.
1. Path control first: PCC 45 and 47 through B05 and B09 as in E4 must give the sha256 of the stream, of the HEIF
   file and of the recon recorded in the journal; otherwise outcome (в) and nothing more is run.
2. PCC 46: B05 gives the stream (kept in cycles/14/b09_diag/, outside Git); MP4Box packs it twice with the command of
   B09 - exit code, stdout and stderr (last 8000 characters); if it fails, once more with GPAC logs at debug level
   (-logs=all@debug:ncl, from the help of the installed build), the whole output beside the stream; if it works,
   B09 with its round trip and the sha256 of the file.
3. The default GPAC profile (%APPDATA%/GPAC): names, sizes and times of its files, no content.
The outcome follows the rule declared before the numbers. Writes <cache_root>/cycles/14/b09_diag.json once.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from imstudies.lab.coders import vtm, vvc_heif
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

COMMAND = "python studies/P001_vvc-dental/run_b09_diag.py"
QUESTION = ("B09, P001: почему MP4Box не упаковывает в HEIF поток VVC фрагмента D05_r00_c01 при PCC 46 — дело в "
            "потоке или в состоянии машины?")
CYCLE = 14
RUN_FILE, STORE = "b09_diag.json", "b09_diag"
UNIT, IMAGE, CONTROLS, FAILING = "D05_r00_c01", "D05", (45, 47), 46
TAIL = 8000  # characters of stdout and stderr kept per attempt
LOGS = "-logs=all@debug:ncl"  # `gpac -h log` of the installed build: every tool at debug level, no colour
PROFILE_AFTER = datetime.datetime(2026, 10, 16, tzinfo=datetime.timezone.utc)  # "changed after 2026-10-15"
PROBE_UTC = datetime.datetime(2026, 10, 24, 9, 44, tzinfo=datetime.timezone.utc)  # core/probe/ was written then
NEAR_S = 3600
DECLARED = {
    "unit": ("Только фрагмент D05_r00_c01 (допуск владельца 2026-10-24): копия D05.png по sha256 реестра, вырезка по "
             "сетке цикла 9; PCC 45, 46, 47; B05 и B09 — те же cfg и исполняемые файлы, что в E4 (sha256 = шапке "
             "журнала цикла 13)."),
    "control": ("До разбора: PCC 45 и 47 через B05 и B09, как в E4, — sha256 потока, файла HEIF и восстановления = "
                "строкам журнала; иначе исход (в), дальше не идти."),
    "diagnosis": ("PCC 46: размер и sha256 потока; MP4Box той же командой, что в B09, — 2 раза: код выхода, stdout, "
                  "stderr (последние 8000 знаков); при коде ≠ 0 — ещё раз с -logs=all@debug:ncl, весь вывод — в "
                  "b09_diag/; при коде 0 — B09 с проверкой туда-обратно и sha256 файла. Профиль GPAC по умолчанию — "
                  "имена, размеры, даты файлов, без содержимого."),
    "outcome": ("(а) сбой воспроизводится, контроли равны журналу — дефект ограничен этим потоком, причина — по выводу; "
                "(б) сбой не воспроизводится, контроли равны — сбой зависел от состояния машины в сессиях 6–7; (в) "
                "контроли не равны журналу — среда изменилась, дальше не идти. Ни при каком исходе точка в E4 не "
                "вписывается."),
    "privacy": "В файле разбора — коды, sha256, размеры и вывод программ; пикселей нет; поток PCC 46 — в b09_diag/ вне Git.",
}


def utc(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)


def load_fragment(data_root, tiles_path):
    """The pixels of D05_r00_c01 only: the copy D05.png (png_sha256 of the registry, mode L, its size) cut by its tile."""
    from PIL import Image

    registry, tile_rows, size = run_e4.load_inputs(data_root, tiles_path, copies=False)
    entry = next(e for e in registry["images"] if e["code"] == IMAGE)
    tile = next(r for r in tile_rows if r["tile"] == UNIT)
    path = Path(data_root) / "P001" / f"{IMAGE}.png"
    if not path.is_file():
        raise RunRefusal(f"the copy data/P001/{IMAGE}.png does not exist")
    if sha256_file(path) != entry["png_sha256"]:
        raise RunRefusal(f"{IMAGE}: png_sha256 of the copy differs from the registry")
    with Image.open(path) as im:
        if im.mode != "L" or im.size != (entry["W"], entry["H"]):
            raise RunRefusal(f"{IMAGE}: the copy is mode {im.mode} {im.size}, the registry says L {(entry['W'], entry['H'])}")
        fragment = np.array(np.asarray(im)[tile["y"]:tile["y"] + size, tile["x"]:tile["x"] + size])
    return fragment, {"x": tile["x"], "y": tile["y"], "tile": size}


def journal_rows(journal_path, pin):
    """The header of the journal of cycle 13 and the last VTM rows of the unit at PCC 45, 46, 47 (read only)."""
    if not Path(journal_path).is_file():
        raise RunRefusal(f"no journal {journal_path}")
    sha = sha256_file(journal_path)
    if sha != pin:
        raise RunRefusal(f"the journal has sha256 {sha}, the diagnosis is declared for {pin}")
    header, entries, _ = e4_book.read_journal(journal_path)
    last = {e["task"]: e["rows"][0] for e in entries if e["task"].startswith(f"{UNIT}|vtm|")}
    return header, {q: last[f"{UNIT}|vtm|{q}"] for q in (*CONTROLS, FAILING)}, sha


def executables(paths, header):
    pinned = header["codec_versions"]["vtm_gpac_sha256"]
    out = {}
    for key in ("encoder", "decoder", "cfg", "mp4box", "gpac"):
        now = sha256_file(paths[key]) if Path(paths[key]).is_file() else None
        out[key] = {"sha256": now, "journal": pinned.get(key), "equal": now is not None and now == pinned.get(key)}
    return out


def vtm_row(paths, image, qp, work):
    """B05 + B09 exactly as an E4 task; a module name so that tests can put a coder of their own in its place."""
    return e4_book.vtm_task(paths, UNIT, "f", image, qp, work, run_e4.TIMEOUT_S["f"])


def encode(paths, image, qp, work):
    coded = vtm.encode_decode(image, qp, max_pixels=image.size, level=e4_book.LEVEL_VTM,
                              frame_rate=e4_book.FRAME_RATE_VTM, timeout_s=run_e4.TIMEOUT_S["f"],
                              encoder=paths["encoder"], decoder=paths["decoder"], cfg=paths["cfg"], work_root=work)
    return coded.bitstream, coded.recon


def pack_b09(paths, bitstream, recon, work):
    return vvc_heif.pack(bitstream, recon, timeout_s=run_e4.TIMEOUT_S["f"], mp4box=paths["mp4box"],
                         gpac=paths["gpac"], decoder=paths["decoder"], work_root=work)


FIELDS = ("bitstream_sha256", "file_sha256", "recon_sha256")


def control(paths, image, qp, journal_row, work):
    row = vtm_row(paths, image, qp, work)
    return {"pcc": qp, "status": row["status"], **{f: row.get(f) for f in FIELDS},
            "journal": {f: journal_row.get(f) for f in FIELDS},
            "equal": row["status"] != "error" and all(row.get(f) == journal_row.get(f) for f in FIELDS)}


def pack_attempt(mp4box, bitstream, work, extra=()):
    """MP4Box with the command of B09 (`-add-image <stream>.266:primary -new <file>.heic`) in a fresh folder."""
    with tempfile.TemporaryDirectory(dir=work, prefix="pack_") as tmp:
        stream, heic = Path(tmp) / "stream.266", Path(tmp) / "image.heic"
        stream.write_bytes(bitstream)
        cmd = [*mp4box, "-add-image", f"{stream}:primary", "-new", str(heic), *extra]
        t0 = time.perf_counter()
        out = subprocess.run(cmd, capture_output=True, timeout=run_e4.TIMEOUT_S["f"])
        stdout, stderr = (b.decode("utf-8", errors="replace") for b in (out.stdout, out.stderr))
        written = heic.is_file()
        attempt = {"command": [str(c) for c in cmd], "exit_code": out.returncode,
                   "seconds": round(time.perf_counter() - t0, 3), "stdout_chars": len(stdout),
                   "stderr_chars": len(stderr), "stdout_tail": stdout[-TAIL:], "stderr_tail": stderr[-TAIL:],
                   "file_written": written, "file_bytes": heic.stat().st_size if written else None,
                   "file_sha256": hashlib.sha256(heic.read_bytes()).hexdigest() if written else None}
    return attempt, stdout, stderr


def gpac_profile(folder):
    """Names, sizes and times of the files of the default GPAC profile - never their content. A run of GPAC touches
    the time of the folder itself, so only the files count."""
    folder = Path(folder) if folder else None
    if folder is None or not folder.is_dir():
        return {"folder": "%APPDATA%/GPAC", "exists": False}
    files = []
    for p in sorted(folder.rglob("*")):
        st = p.stat()
        files.append({"name": p.relative_to(folder).as_posix(), "kind": "folder" if p.is_dir() else "file",
                      "bytes": st.st_size if p.is_file() else None, "modified_utc": utc(st.st_mtime).isoformat(timespec="seconds")})
    stamps = [(f, datetime.datetime.fromisoformat(f["modified_utc"])) for f in files if f["kind"] == "file"]
    return {"folder": "%APPDATA%/GPAC", "exists": True, "folder_modified_utc": utc(folder.stat().st_mtime).isoformat(timespec="seconds"),
            "files": files,
            "files_changed_after_2026_10_15": [f["name"] for f, t in stamps if t >= PROFILE_AFTER],
            "files_changed_within_1_h_of_2026_10_24_0944_utc": [f["name"] for f, t in stamps
                                                                 if abs((t - PROBE_UTC).total_seconds()) <= NEAR_S]}


def outcome(controls, attempts):
    """The rule declared before the numbers."""
    if not all(c["equal"] for c in controls):
        return "в", "контроли не равны журналу — среда изменилась, дальше не шли"
    codes = [a["exit_code"] for a in attempts]
    if codes and all(c != 0 for c in codes):
        return "а", "сбой воспроизводится, контроли равны журналу — дефект ограничен этим потоком; причина — по выводу"
    if codes and all(c == 0 for c in codes):
        return "б", "сбой не воспроизводится, контроли равны — сбой зависел от состояния машины в сессиях 6–7"
    return "не объявлен", f"попытки разошлись: коды выхода {codes}"


def diagnose(paths, image, rows, header, out_dir, mp4box, profile_dir, state, journal_sha, place, log=print):
    """The diagnosis by the declared steps; writes the file of the diagnosis once. Outcome (в) is a refusal in words
    after the file is written."""
    out_dir = Path(out_dir)
    run_path, store = out_dir / RUN_FILE, out_dir / STORE
    for path in (run_path, store):
        if path.exists():
            raise RunRefusal(f"{path} already exists: a diagnosis file is never overwritten")
    store.mkdir(parents=True)
    work = store / "work"
    work.mkdir()
    started = datetime.datetime.now(datetime.timezone.utc)
    log(f"{COMMAND}\n{QUESTION}\nDeclared before the numbers:")
    for key, text in DECLARED.items():
        log(f"  [{key}] {text}")
    record = {"run": RUN_FILE, "study": "P001", "cycle": CYCLE, "question": QUESTION, "command": COMMAND,
              "started_utc": started.isoformat(timespec="seconds"), **state, "declared_before_numbers": DECLARED,
              "unit": UNIT, "place": place, "journal_sha256": journal_sha,
              "journal_row_pcc46": {"status": rows[FAILING]["status"], "detail": rows[FAILING]["detail"]},
              "gpac_profile": gpac_profile(profile_dir), "executables": executables(paths, header),
              "controls": [], "stream": None, "attempts": [], "verbose_attempt": None, "round_trip": None}
    try:
        if all(e["equal"] for e in record["executables"].values()):
            record["controls"] = [control(paths, image, q, rows[q], work) for q in CONTROLS]
        else:
            record["controls"] = [{"pcc": None, "equal": False, "why": "sha256 of the executables differ from the journal"}]
        if all(c["equal"] for c in record["controls"]):
            bitstream, recon = encode(paths, image, FAILING, work)
            (store / f"stream_pcc{FAILING}.266").write_bytes(bitstream)
            record["stream"] = {"bytes": len(bitstream), "sha256": hashlib.sha256(bitstream).hexdigest(),
                                "file": f"{STORE}/stream_pcc{FAILING}.266"}
            record["attempts"] = [pack_attempt(mp4box, bitstream, work)[0] for _ in range(2)]
            if any(a["exit_code"] != 0 for a in record["attempts"]):
                attempt, stdout, stderr = pack_attempt(mp4box, bitstream, work, extra=(LOGS,))
                (store / f"mp4box_logs_pcc{FAILING}.txt").write_text(f"[stdout]\n{stdout}\n[stderr]\n{stderr}\n",
                                                                     encoding="utf-8", newline="\n")
                record["verbose_attempt"] = attempt | {"log_file": f"{STORE}/mp4box_logs_pcc{FAILING}.txt"}
            else:
                packed = pack_b09(paths, bitstream, recon, work)
                record["round_trip"] = {"file_bytes": len(packed.file),
                                        "file_sha256": hashlib.sha256(packed.file).hexdigest(),
                                        "result": packed.provenance.get("round_trip")}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    letter, words = outcome(record["controls"], record["attempts"])
    record["outcome"] = {"letter": letter, "words": words}
    record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with open(run_path, "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(e4_book.jsonable(record), ensure_ascii=False, indent=1, allow_nan=False))
    for c in record["controls"]:
        log(f"  control PCC {c['pcc']}: {'equal to the journal' if c['equal'] else 'DIFFERS from the journal'}")
    for k, a in enumerate(record["attempts"], start=1):
        log(f"  PCC {FAILING}, MP4Box attempt {k}: exit code {a['exit_code']}; stderr: {a['stderr_tail'][-300:]!r}")
    log(f"Outcome ({letter}): {words}\nDiagnosis file: {run_path} (sha256 {sha256_file(run_path)})")
    if letter == "в":
        raise RunRefusal(f"outcome (в): {words}; the file {RUN_FILE} holds what was found")
    return run_path


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace", line_buffering=True)
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    try:
        paths = run_e4.load_paths()
        cache = paths["cache_root"] / "cycles"
        header, rows, journal_sha = journal_rows(cache / "13" / run_e4.JOURNAL_FILE, run_e4.JOURNAL13_SHA256)
        image, place = load_fragment(paths["data_root"], cache / "9" / "p001_tiles.json")
        appdata = os.environ.get("APPDATA")
        diagnose(paths, image, rows, header, cache / str(CYCLE), [str(paths["mp4box"])],
                 Path(appdata) / "GPAC" if appdata else None, git_state(ROOT), journal_sha, place)
    except RunRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
