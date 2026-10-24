"""Lab tests of run_e4.py (study P001, E4 contract of STUDY.md section 7) on a synthetic copy of the set in tmp_path.

No pixel of P001 is read. The copy: a registry of two "images" 96 x 160 (D01, D02), a grid of 2 x 1 fragments
32 x 32 per image (sizes given by the test), PNG copies of random pixels; invented patient names lie in a private
table beside it, which the run must not read. VTM is a substitute coder with a given PSNR-HVS-M curve by QP
(D01 monotone, D02 not); JPEG, HEIF, AVIF are the real wrappers (lab_paths.json with the B12 environment needed).

- refinement: the crossing found (3 QP between the right nodes), no crossing both ways with the right reason,
  a non-monotone curve takes the first crossing from the high-quality side;
- the run: rows per image (19 grid + 3 refinement), whole images before fragments, slow first;
- resume after a stop between the grid and the refinement and after one in the middle of the refinement:
  every refinement runs once, rows equal an uninterrupted run;
- privacy: no invented name and no pixel array in the run file, the book, the journal and the session file;
- --plan equals a count by hand; missing registry, grid or copy refused in words;
- sessions (cycle 13; one coder process, so that the moment of a stop is exact): a stop by the deadline (a clock
  of the test) and by the flag file - after the signal no task starts, the started one is in the journal, the
  journal is copied, the flag renamed, the same command gives rows equal to an uninterrupted run; the progress
  line counted by hand; the session file holds the output; --status of a running, a stopped and a cut-off session
  (the journal unchanged by it); another commit refused; a failed grid node runs again once, right after the
  failure, and the image is refined in the same session; failed twice - the grid waits for the next session and
  the session writes no run file;
- assembly from the journal (cycle 14): a journal with a task failing at every attempt gives rows equal to an
  uninterrupted run except that task (its rows `error`, attempts counted), without calling a coder (the substitute
  coders fail when called); an existing file, another sha256 of the journal and an unfinished run are refused;
  the cause of a failed program carries the stderr of a substitute program.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.lab

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
NAMES = ["Шевченко Тарас Григорович 1814-03-09.bmp", "Українка Леся 1871-02-25.png"]  # invented, like the real ones


@pytest.fixture(scope="module")
def e4():
    spec = importlib.util.spec_from_file_location("run_e4", STUDY / "run_e4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GRID = list(range(-12, 61, 4))
PLAN = {"vtm": {"f": [30, 50], "m_grid": GRID, "refine_step": 4, "threshold_db": 42.0},
        "jpeg": [50, 100], "heif": [50, 100], "avif": [50, 100], "tile": 32, "pool": 2,
        "controls": {"fragment_qps": [30], "whole_qp": 40}}
STATE = {"core_commit": "test", "core_tree": "test"}


def curve(unit, qp):
    """PSNR-HVS-M of the substitute VTM: D01 falls steadily (crossing between 24 and 28); D02 dips below 42 dB at
    QP 4, comes back at 8 and falls again (first crossing from the high-quality side: between 0 and 4)."""
    if unit.startswith("D02") and len(unit) == 3:
        return {4: 41.0, 8: 43.0}.get(qp, 60.0 - 0.6 * (qp + 12) if qp <= 0 else 48.0 - 0.5 * qp)
    return 72.0 - 0.9 * (qp + 12)


def fake_vtm(paths, unit, level, image, qp, work):
    h, w = image.shape
    size = 2000 - 20 * qp
    return {"key": f"{unit}|vtm|{qp}", "unit": unit, "level": level, "codec": "vtm", "param": qp, "status": "valid",
            "file_bytes": size, "cr_file": h * w / size, "bpp": 8 * size / (h * w), "psnr_db": curve(unit, qp) - 5,
            "psnr_hvs_m_db": curve(unit, qp), "haarpsi": 0.9, "seconds": 0.001, "seconds_metrics": 0.0,
            "file_sha256": hashlib.sha256(f"{unit}{qp}".encode()).hexdigest(), "recon_sha256": "r", "detail": None}


def synthetic_copy(root):
    from PIL import Image

    rng = np.random.default_rng(7)
    (root / "data" / "P001").mkdir(parents=True)
    (root / "data" / "registry").mkdir()
    entries, rows = [], []
    for code in ("D01", "D02"):
        values = rng.integers(0, 256, (96, 160)).astype(np.uint8)
        path = root / "data" / "P001" / f"{code}.png"
        Image.fromarray(values, "L").save(path)
        entries.append({"code": code, "H": 96, "W": 160, "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "full_tiles": 2})
        rows += [{"tile": f"{code}_r{r:02d}_c00", "code": code, "r": r, "c": 0, "x": 64, "y": 16 + 32 * r} for r in (0, 1)]
    (root / "data" / "registry" / "P001.json").write_text(json.dumps({"contract": "B10 0.2.0", "images": entries}),
                                                          encoding="utf-8")
    (root / "cycles9").mkdir()
    (root / "cycles9" / "p001_tiles.json").write_text(json.dumps({"rows": rows, "parameters": {"tile": 32}}),
                                                       encoding="utf-8")
    (root / "private").mkdir()
    (root / "private" / "P001_codes.json").write_text(
        json.dumps({"codes": {"D01": [{"path": NAMES[0]}], "D02": [{"path": NAMES[1]}]}}, ensure_ascii=False),
        encoding="utf-8")
    return root


TIMING = {"f": {"vtm": [(30, 8.0), (50, 2.0)], "jpeg": [(50, 0.001), (100, 0.002)], "heif": [(50, 0.05), (100, 0.08)],
                "avif": [(50, 0.04), (100, 0.04)]},
          "m": {"vtm": [(q, 600.0 - 8 * (q + 12)) for q in GRID], "jpeg": [(50, 0.01), (100, 0.02)],
                "heif": [(50, 0.8), (100, 1.2)], "avif": [(50, 0.4), (100, 0.4)]},
          "metrics": {"f": 0.05, "m": 0.9}, "k": 3.5, "refine_qps": [25, 26, 27]}


@pytest.fixture()
def setup(e4, tmp_path, monkeypatch):
    if not (ROOT / "lab_paths.json").is_file():
        pytest.skip("lab_paths.json is absent: the laboratory is not configured on this machine")
    raw = json.loads((ROOT / "lab_paths.json").read_text(encoding="utf-8"))
    python = raw.get("avif_libheif_python")
    if not python or not Path(python).is_file():
        pytest.skip("the B12 environment is not configured on this machine")
    root = synthetic_copy(tmp_path)
    paths = {"data_root": root / "data", "cache_root": root / "cache", "avif_libheif_python": Path(python),
             "private_root": root / "private", **{k: root / f"no_{k}" for k in ("encoder", "decoder", "cfg", "mp4box", "gpac")}}
    registry, tile_rows, tile = e4.load_inputs(paths["data_root"], root / "cycles9" / "p001_tiles.json", pins=(None, None))
    units = e4.load_units(paths["data_root"], registry, tile_rows, tile)
    monkeypatch.setattr(e4, "vtm_task", fake_vtm)
    return root, paths, registry, tile_rows, units


def run(e4, setup, out, plan=PLAN, state=STATE, log=lambda *a: None, **kw):
    root, paths, _, _, units = setup
    return e4.run(paths, units, plan, out, state, TIMING, "reg", "tiles", log=log, **kw)


def test_refinement_rule(e4):
    ref = e4.e4_book.refinement
    grid = [(q, 72 - 0.9 * (q + 12)) for q in GRID]  # 43.2 dB at QP 20, 39.6 dB at QP 24
    a = max(q for q, v in grid if v >= 42)
    assert a == 20
    assert ref(grid) == (list(range(a + 1, a + 4)), None) and a in GRID and a + 4 in GRID
    assert ref([(q, 50.0) for q in GRID]) == ([], "no_crossing: every node is at or above 42 dB")
    assert ref([(q, 30.0) for q in GRID]) == ([], "no_crossing: every node is below 42 dB")
    assert ref([(q, "+inf" if q < 0 else 30.0) for q in GRID]) == ([-3, -2, -1], None)  # lossless counts as above
    bumpy = [(q, curve("D02", q)) for q in GRID]
    assert ref(bumpy) == ([1, 2, 3], None)  # the first crossing from the high-quality side, not the later ones
    assert ref([(q, 30.0 if q < 20 else 50.0) for q in GRID]) == (
        [], "no_crossing: no fall below 42 dB from the high-quality side")


def test_run_rows_order_and_refinement(e4, setup, tmp_path):
    run_path, book_path = run(e4, setup, tmp_path / "out")
    record = json.loads(run_path.read_text(encoding="utf-8"))
    vtm = {u: sorted(r["param"] for r in record["rows"] if r["unit"] == u and r["codec"] == "vtm") for u in ("D01", "D02")}
    a = max(q for q in GRID if curve("D01", q) >= 42)
    assert vtm["D01"] == sorted(GRID + [a + 1, a + 2, a + 3]) and vtm["D02"] == sorted(GRID + [1, 2, 3])
    assert record["summary"]["refinement"]["D02"] == {"qps": [1, 2, 3], "reason": None}
    assert len(record["rows"]) == 2 * (19 + 3 + 6) + 4 * (2 + 6)
    whole = [r for r in record["rows"] if r.get("phase") == "grid"]
    assert {r["level"] for r in whole} == {"m"}
    second = sorted((r for r in record["rows"] if r.get("phase") in ("refine", "fragment")), key=lambda r: r["order"])
    assert [r["phase"] for r in second[:6]] == ["refine"] * 6  # whole images before fragments
    frag = [r for r in second if r["phase"] == "fragment"]
    assert [r["param"] for r in frag[:4]] == [30] * 4  # slow first: QP 30 expected 8 s, QP 50 2 s
    assert all(d["equal"] for d in record["summary"]["determinism"]) and len(record["summary"]["determinism"]) == 2
    with pytest.raises(e4.RunRefusal, match=e4.RUN_FILE):
        run(e4, setup, tmp_path / "out")


def keyed(run_path):
    record = json.loads(Path(run_path).read_text(encoding="utf-8"))
    return record, {r["key"]: (r["status"], r["file_sha256"], r["psnr_hvs_m_db"]) for r in record["rows"]}


def test_resume_between_grid_and_refinement_refines_once(e4, setup, tmp_path):
    whole, _ = run(e4, setup, tmp_path / "whole")
    out = tmp_path / "stopped"
    with pytest.raises(e4.RunStopped):
        run(e4, setup, out, stop_after=2 * (19 + 3))  # the last task of the whole-image stage, before any refinement
    entries = [json.loads(line) for line in (out / e4.JOURNAL_FILE).read_text(encoding="utf-8").splitlines()[1:]]
    assert not any(r.get("phase") == "refine" for e in entries for r in e["rows"])
    with pytest.raises(e4.RunStopped):
        run(e4, setup, out, stop_after=3)  # in the middle of the refinement: 3 of its 6 tasks done
    resumed, _ = run(e4, setup, out)
    entries = [json.loads(line) for line in (out / e4.JOURNAL_FILE).read_text(encoding="utf-8").splitlines()[1:]]
    refine_tasks = [e["task"] for e in entries if any(r.get("phase") == "refine" for r in e["rows"])]
    assert len(refine_tasks) == len(set(refine_tasks)) == 6
    record_r, rows_r = keyed(resumed)
    assert rows_r == keyed(whole)[1] and record_r["resumed_tasks"] == 2 * (19 + 3) + 3
    assert len(record_r["rows"]) == len({r["key"] for r in record_r["rows"]})


def test_privacy_no_names_no_pixels(e4, setup, tmp_path):
    run_path, book_path = run(e4, setup, tmp_path / "out")
    texts = [run_path.read_text(encoding="utf-8"), (tmp_path / "out" / e4.JOURNAL_FILE).read_text(encoding="utf-8"),
             (tmp_path / "out" / e4.SESSION_LOG.format(k=1)).read_text(encoding="utf-8")]
    with zipfile.ZipFile(book_path) as z:
        texts += [z.read(n).decode("utf-8", errors="ignore") for n in z.namelist()]
    for name in NAMES:
        for part in (name, Path(name).stem, json.dumps(name)[1:-1]):
            assert not any(part in t for t in texts), part
    assert longest_int_list(json.loads(texts[0])) < 64  # no row of pixels anywhere


def longest_int_list(obj):
    if isinstance(obj, list):
        own = len(obj) if obj and all(isinstance(x, int) and not isinstance(x, bool) for x in obj) else 0
        return max([own] + [longest_int_list(x) for x in obj])
    if isinstance(obj, dict):
        return max([0] + [longest_int_list(x) for x in obj.values()])
    return 0


def test_plan_by_hand(e4, setup):
    _, _, registry, tile_rows, _ = setup
    got = e4.plan_counts(registry, tile_rows, PLAN, TIMING)
    assert [(i["code"], i["fragments"], i["vtm_whole"], i["vtm_fragments"], i["still_tasks"], i["tasks"])
            for i in got["images"]] == [("D01", 2, 22, 4, 9, 35), ("D02", 2, 22, 4, 9, 35)]
    assert got["totals"] == {"vtm": {"tasks": 52, "files": 52}, "jpeg": {"tasks": 6, "files": 12},
                             "heif": {"tasks": 6, "files": 12}, "avif": {"tasks": 6, "files": 12}}
    vtm_m = sum(600.0 - 8 * (q + 12) for q in GRID) + sum(600.0 - 8 * (q + 12) for q in (25, 26, 27))
    whole = vtm_m + 22 * 0.9 + (0.01 + 0.02 + 0.8 + 1.2 + 0.4 + 0.4) + 6 * 0.9
    frag = (8.0 + 2.0 + 2 * 0.05) + (0.001 + 0.002 + 0.05 + 0.08 + 0.04 + 0.04) + 6 * 0.05
    seconds = 2 * whole + 4 * frag
    assert got["hours_1"] == pytest.approx(seconds / 3600, rel=1e-12)
    assert got["hours_4"] == pytest.approx(seconds / 3.5 / 3600, rel=1e-12)


def test_missing_inputs_refused_in_words(e4, setup):
    root, paths, _, _, _ = setup
    tiles = root / "cycles9" / "p001_tiles.json"
    with pytest.raises(e4.RunRefusal, match="the grid cycles/9/p001_tiles.json does not exist"):
        e4.load_inputs(paths["data_root"], root / "none.json", pins=(None, None))
    (paths["data_root"] / "P001" / "D02.png").rename(root / "moved.png")
    with pytest.raises(e4.RunRefusal, match="D02: the copy data/P001/D02.png does not exist"):
        e4.load_inputs(paths["data_root"], tiles, pins=(None, None))
    with pytest.raises(e4.RunRefusal, match="registry data/registry/P001.json: sha256 differs from the pinned"):
        e4.load_inputs(paths["data_root"], tiles, pins=("0" * 64, None))
    (paths["data_root"] / "registry" / "P001.json").rename(root / "moved.json")
    with pytest.raises(e4.RunRefusal, match="the registry data/registry/P001.json does not exist"):
        e4.load_inputs(paths["data_root"], tiles, pins=(None, None))


# ---------------------------------------------------------------- sessions (cycle 13)

PLAN1 = PLAN | {"pool": 1}  # one coder process: the moment of a stop is exact
TOTAL = 2 * (19 + 3 + 3) + 4 * (2 + 3)  # tasks: per image grid, refinement, JPEG/HEIF/AVIF; per fragment VTM, three


class Fake:
    """Substitute coders that note every start; `fail` - task id: how many times it fails before it works;
    `trigger` - (n, action): when the n-th task starts, action() runs and that task waits (up to 5 s) until the run
    says that it stops by plan."""

    def __init__(self, e4, still, fail=None, trigger=None):
        self.e4, self.real_still, self.fail, self.trigger = e4, still, dict(fail or {}), trigger
        self.events, self.lines, self.lock, self.release, self.starts = [], [], threading.Lock(), threading.Event(), 0

    def started(self, task):
        with self.lock:
            self.starts += 1
            n = self.starts
            self.events.append(("start", task))
        if self.trigger and n == self.trigger[0]:
            self.trigger[1]()
            self.release.wait(5)

    def vtm(self, paths, unit, level, image, qp, work):
        task = f"{unit}|vtm|{qp}"
        self.started(task)
        with self.lock:
            failing = self.fail.get(task, 0) > 0
            self.fail[task] = self.fail.get(task, 0) - failing
        if failing:
            return self.e4.e4_book.error_row(unit, level, "vtm", qp, RuntimeError("substitute failure"))
        return fake_vtm(paths, unit, level, image, qp, work)

    def still(self, paths, unit, level, image, codec, values, work):
        self.started(f"{unit}|{codec}|batch")
        return self.real_still(paths, unit, level, image, codec, values, work)

    def log(self, text):
        with self.lock:
            self.events.append(("log", text))
            self.lines.append(text)
        if "planned stop" in text:
            self.release.set()


@pytest.fixture()
def fake(e4, setup, monkeypatch):
    real_still = e4.still_task

    def make(**kw):
        f = Fake(e4, real_still, **kw)
        monkeypatch.setattr(e4, "vtm_task", f.vtm)
        monkeypatch.setattr(e4, "still_task", f.still)
        return f
    return make


def reference(e4, setup, fake, tmp_path):
    """Rows of an uninterrupted run in one session."""
    run_path, _ = run(e4, setup, tmp_path / "reference", plan=PLAN1, log=fake().log)
    return keyed(run_path)[1]


def journal(e4, out):
    return [json.loads(line) for line in (out / e4.JOURNAL_FILE).read_text(encoding="utf-8").splitlines()[1:]]


def stopped_at(f, reason):
    """Tasks started before the run said it stops by plan; True if none started after."""
    stop = next(i for i, (what, text) in enumerate(f.events) if what == "log" and f"planned stop ({reason})" in text)
    return [t for what, t in f.events[:stop] if what == "start"], not any(what == "start" for what, _ in f.events[stop:])


def test_stop_by_deadline_finishes_the_started_task_and_the_same_command_continues(e4, setup, fake, tmp_path):
    expected = reference(e4, setup, fake, tmp_path)
    now = [0.0]
    f = fake(trigger=(5, lambda: now.__setitem__(0, 1000.0)))  # the 5th task starts at 1000 s: past 0.25 h
    out = tmp_path / "out"
    assert run(e4, setup, out, plan=PLAN1, log=f.log, hours=0.25, clock=lambda: now[0], poll_s=0.01) is None
    started, none_after = stopped_at(f, "deadline")
    assert len(started) == 5 and none_after
    entries = journal(e4, out)
    assert {e["task"] for e in entries if e4.entry_kind(e["task"]) == "task"} == set(started)  # the started one too
    end = [e for e in entries if e["task"] == "session|1|end"]
    assert end[0]["end"] == "stopped by plan (deadline)" and end[0]["finished_after_signal"] == 1
    assert (out / e4.SESSION_COPY.format(k=1)).read_bytes() == (out / e4.JOURNAL_FILE).read_bytes()
    assert f.lines[-1] == e4.STOPPED
    assert (out / e4.SESSION_LOG.format(k=1)).read_text(encoding="utf-8").splitlines() == "\n".join(f.lines).splitlines()
    # the progress line at the stop, by hand: done - the grid at QP -12, -12, -8, -8, -4 (slow first)
    vtm_m = lambda q: 600.0 - 8 * (q + 12) + 0.9  # noqa: E731
    rest = (2 * sum(vtm_m(q) for q in GRID) - sum(vtm_m(q) for q in (-12, -12, -8, -8, -4))
            + 2 * sum(vtm_m(q) for q in (25, 26, 27))  # refinement not known yet: the QPs of the dry run
            + 2 * (0.01 + 0.02 + 0.8 + 1.2 + 0.4 + 0.4 + 6 * 0.9)
            + 4 * (8.0 + 2.0 + 2 * 0.05 + 0.001 + 0.002 + 0.05 + 0.08 + 0.04 + 0.04 + 6 * 0.05))
    line = [x for x in f.lines if x.startswith("progress")][-1]
    assert (f"session 1, wall 0.28 h | done 5/{TOTAL} tasks: whole images VVC (VTM) 5/44, JPEG 0/2, HEIF 0/2, AVIF "
            "0/2; fragments VVC (VTM) 0/8, JPEG 0/4, HEIF 0/4, AVIF 0/4; controls 0/2 | error: 0 rows, 0 tasks to run "
            f"again | expected rest {rest / 3.5 / 3600:.2f} h") in line
    resumed, _ = run(e4, setup, out, plan=PLAN1, log=f.log)
    record, rows = keyed(resumed)
    assert rows == expected and record["resumed_tasks"] == 5


def test_stop_by_flag_and_status_of_a_running_and_a_stopped_session(e4, setup, fake, tmp_path):
    _, _, registry, tile_rows, _ = setup
    levels = e4.unit_levels(registry, tile_rows)
    expected = reference(e4, setup, fake, tmp_path)
    out, seen = tmp_path / "out", []

    def signal():
        seen.append(e4.status(out, levels, PLAN1, TIMING, log=lambda *a: None))  # while the session runs
        (out / e4.STOP_FLAG).write_bytes(b"")

    f = fake(trigger=(5, signal))
    assert run(e4, setup, out, plan=PLAN1, log=f.log, poll_s=0.01) is None
    assert seen == ["running"]
    started, none_after = stopped_at(f, "flag")
    assert len(started) == 5 and none_after
    assert {e["task"] for e in journal(e4, out) if e4.entry_kind(e["task"]) == "task"} == set(started)
    assert not (out / e4.STOP_FLAG).exists() and (out / f"{e4.STOP_FLAG}.done-1").is_file()
    assert (out / e4.SESSION_COPY.format(k=1)).read_bytes() == (out / e4.JOURNAL_FILE).read_bytes()
    assert f.lines[-1] == e4.STOPPED
    lines = []
    assert e4.status(out, levels, PLAN1, TIMING, log=lines.append) == "stopped by plan (flag)"
    at_stop = [x for x in f.lines if x.startswith("progress")][-1]
    assert lines[0].split(" | ")[2:5] == at_stop.split(" | ")[2:5]  # the same counts and rest, from the journal
    resumed, _ = run(e4, setup, out, plan=PLAN1, log=f.log)
    assert keyed(resumed)[1] == expected
    assert e4.status(out, levels, PLAN1, TIMING, log=lines.append).startswith("the run is finished")


def test_status_of_a_cut_session_reads_only_and_another_commit_is_refused(e4, setup, fake, tmp_path):
    _, _, registry, tile_rows, _ = setup
    levels = e4.unit_levels(registry, tile_rows)
    out = tmp_path / "out"
    f = fake()
    with pytest.raises(e4.RunStopped):
        run(e4, setup, out, plan=PLAN1, log=f.log, stop_after=3)  # as Ctrl+C: the session writes no end
    with open(out / e4.JOURNAL_FILE, "a", encoding="utf-8") as j:
        j.write('{"task": "D01|vtm|')  # a line torn by a killed process
    before = hashlib.sha256((out / e4.JOURNAL_FILE).read_bytes()).hexdigest()
    lines = []
    state = e4.status(out, levels, PLAN1, TIMING, now=time.time() + 3600, log=lines.append)
    assert state.startswith("cut off") and lines[-1].endswith("; torn lines 1") and f"done 3/{TOTAL} tasks" in lines[0]
    assert hashlib.sha256((out / e4.JOURNAL_FILE).read_bytes()).hexdigest() == before
    with pytest.raises(e4.RunRefusal, match="the journal was started at core commit test, the code is now at other"):
        run(e4, setup, out, plan=PLAN1, state={"core_commit": "other", "core_tree": "test"})


def test_a_failed_grid_node_runs_again_at_once_and_the_image_is_refined_in_the_same_session(e4, setup, fake, tmp_path):
    f = fake(fail={"D01|vtm|20": 1})
    run_path, _ = run(e4, setup, tmp_path / "out", plan=PLAN1, log=f.log)
    starts = [t for what, t in f.events if what == "start"]
    first = starts.index("D01|vtm|20")
    assert starts[first + 1] == "D01|vtm|20" and starts.count("D01|vtm|20") == 2  # right after the failure, once
    entries = journal(e4, tmp_path / "out")
    assert [e["rows"][0]["status"] for e in entries if e["task"] == "D01|vtm|20"] == ["error", "valid"]
    record = json.loads(run_path.read_text(encoding="utf-8"))
    assert record["summary"]["refinement"]["D01"] == {"qps": [21, 22, 23], "reason": None}
    assert record["retried_in_session"] == 1
    assert [s["end"] for s in record["summary"]["sessions"]] == ["finished the run"]


def test_a_node_failed_twice_leaves_the_grid_to_the_next_session(e4, setup, fake, tmp_path):
    expected = reference(e4, setup, fake, tmp_path)
    out = tmp_path / "out"
    f = fake(fail={"D01|vtm|20": 2})
    assert run(e4, setup, out, plan=PLAN1, log=f.log) is None and not (out / e4.RUN_FILE).exists()
    assert f.lines[-1] == ("сессия закончена, не готово задач: 4 (снимков без законченной сетки: 1) — та же команда "
                           "повторит их")
    entries = journal(e4, out)
    assert [e["rows"][0]["status"] for e in entries if e["task"] == "D01|vtm|20"] == ["error", "error"]
    assert {e["task"] for e in entries for r in e["rows"] if r.get("phase") == "refine"} == {
        "D02|vtm|1", "D02|vtm|2", "D02|vtm|3"}
    resumed, _ = run(e4, setup, out, plan=PLAN1, log=f.log)
    record, rows = keyed(resumed)
    assert rows == expected and record["retried_tasks"] == 1
    assert record["summary"]["refinement"]["D01"]["qps"] == [21, 22, 23]
    assert [s["end"] for s in record["summary"]["sessions"]] == ["ended with tasks left", "finished the run"]


# ---------------------------------------------------------------- assembly from the journal (cycle 14)

ALWAYS = "D02_r01_c00|vtm|50"  # a fragment task that fails at every attempt (not a control task)


def never_called(*args, **kwargs):
    raise AssertionError("the assembly called a coder")


def test_assembly_from_a_journal_with_a_task_failing_every_time(e4, setup, fake, tmp_path, monkeypatch):
    _, _, registry, tile_rows, _ = setup
    levels = e4.unit_levels(registry, tile_rows)
    expected = reference(e4, setup, fake, tmp_path)
    out = tmp_path / "out"
    f = fake(fail={ALWAYS: 99, "D01|vtm|20": 1})  # a grid node fails once and works on its retry
    for _ in range(2):
        assert run(e4, setup, out, plan=PLAN1, log=f.log) is None  # tasks left: no run file
    monkeypatch.setattr(e4, "vtm_task", never_called)
    monkeypatch.setattr(e4, "still_task", never_called)
    sha = hashlib.sha256((out / e4.JOURNAL_FILE).read_bytes()).hexdigest()
    run_path, book_path = e4.assemble(out, levels, TIMING, {"core_commit": "assembly", "core_tree": "tree"},
                                      journal_sha256=sha, log=lambda *a: None)
    record, rows = keyed(run_path)
    assert {k: v for k, v in rows.items() if k != ALWAYS} == {k: v for k, v in expected.items() if k != ALWAYS}
    assert len(record["rows"]) == len(expected) and rows[ALWAYS][0] == "error"
    row = next(r for r in record["rows"] if r["key"] == ALWAYS)
    assert row["attempts"] == 4 and row["detail"].startswith("RuntimeError: substitute failure")
    a = record["assembled"]
    assert a["line"] == "собран из журнала: незаконченных задач 1" and a["journal_sha256"] == sha
    assert (a["journal_commit"], a["assembly_commit"], record["core_commit"]) == ("test", "assembly", "test")
    assert a["unfinished"] == [{"task": ALWAYS, "attempts": 4, "detail": row["detail"]}]
    assert [s["end"] for s in record["summary"]["sessions"]] == ["ended with tasks left"] * 2
    assert record["summary"]["refinement"]["D01"]["qps"] == [21, 22, 23]  # the node failed once: its last entry counts
    assert len(record["summary"]["determinism"]) == 2 and all(d["equal"] for d in record["summary"]["determinism"])
    assert ("Собран из журнала", a["line"]) in [tuple(x) for x in record["summary_lines"]]
    from openpyxl import load_workbook
    strings = load_workbook(book_path)["Строки"]
    found = [r for r in strings.iter_rows(min_row=3, values_only=True) if r[0] == ALWAYS]
    assert len(found) == 1 and found[0][5] == "error" and found[0][15] == row["detail"]  # status and cause
    assert longest_int_list(record["rows"]) == 0  # privacy: no list of integers in the rows of tasks
    with pytest.raises(e4.RunRefusal, match="already exists"):
        e4.assemble(out, levels, TIMING, STATE, journal_sha256=sha)


def test_assembly_refuses_another_journal_and_an_unfinished_run(e4, setup, fake, tmp_path):
    _, _, registry, tile_rows, _ = setup
    levels = e4.unit_levels(registry, tile_rows)
    out = tmp_path / "out"
    with pytest.raises(e4.RunStopped):
        run(e4, setup, out, plan=PLAN1, log=fake().log, stop_after=3)
    sha = hashlib.sha256((out / e4.JOURNAL_FILE).read_bytes()).hexdigest()
    with pytest.raises(e4.RunRefusal, match="the assembly is declared for " + "0" * 64):
        e4.assemble(out, levels, TIMING, STATE, journal_sha256="0" * 64)
    with pytest.raises(e4.RunRefusal, match="the journal is not a finished run"):
        e4.assemble(out, levels, TIMING, STATE, journal_sha256=sha)
    assert not (out / e4.RUN_FILE).exists()


def test_the_cause_of_a_failed_program_carries_its_output(e4):
    from imstudies.lab.coders.process import run_tool

    program = [sys.executable, "-c", "import sys; print('stdout of the substitute'); "
               "sys.stderr.write('Error: bad VVC NAL in the substitute'); sys.exit(1)"]
    with pytest.raises(subprocess.CalledProcessError) as caught:
        run_tool(program, 60)
    row = e4.e4_book.error_row("D01", "m", "vtm", 46, caught.value)
    assert row["status"] == "error" and row["detail"].startswith("CalledProcessError: ")
    assert "\nstdout: stdout of the substitute" in row["detail"]
    assert row["detail"].endswith("\nstderr: Error: bad VVC NAL in the substitute")
