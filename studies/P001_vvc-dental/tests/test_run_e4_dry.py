"""Lab tests of run_e4_dry.py (study P001, cycle 11).

- slow tasks first: synthetic expected times, ties by QP then unit;
- the estimate written again by hand on rows whose median differs from the mean (fragments, metrics, whole
  images in variants A, B, C, the crossing of 42 dB, hours for 1 and for k processes);
- on one small configuration (frisco and diego, a mosaic 64 x 96 of them, VTM QP 50 and 60, two values of the
  other codecs; lab_paths.json with the coders needed): a run stopped twice by the test stop (in the path control
  and in the pool) and continued by the same call gives the rows of an uninterrupted run (file sha256 and metric
  values); another plan on the same journal is refused in words; the final file refuses the next run by name;
  pool rows are ordered slow first; a task that failed before a stop (an injected failure, as a Ctrl+C that
  reaches the coder) runs again on resume, and the run equals the uninterrupted one.
"""

import importlib.util
import json
import statistics
from pathlib import Path

import pytest

pytestmark = pytest.mark.lab

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]


@pytest.fixture(scope="module")
def dry():
    spec = importlib.util.spec_from_file_location("run_e4_dry", STUDY / "run_e4_dry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_slow_first(dry):
    expected = {10: 5.0, 20: 9.0, 30: 9.0}
    tasks = [("a", 10), ("b", 20), ("a", 20), ("b", 30), ("a", 30)]
    assert dry.slow_first(tasks, lambda t: expected[t[1]]) == [("a", 20), ("b", 20), ("a", 30), ("b", 30), ("a", 10)]


def hand_estimate(rows, k, n_values):
    frag = [r for r in rows if r["level"] == "f"]
    total = 0.0
    for codec, params in n_values.items():
        for p in params:
            total += 154 * statistics.median([r["seconds"] for r in frag if (r["codec"], r["param"]) == (codec, p)])
    units = sorted({r["unit"] for r in frag})
    per_unit = [statistics.median([r["seconds_metrics"] for r in frag if r["unit"] == u]) for u in units]
    n_files = sum(len(v) for v in n_values.values())
    total += 154 * n_files * statistics.median(per_unit)
    m = [r for r in rows if r["unit"] == "mosaic"]
    grid = sorted((r["param"], r["seconds"]) for r in m if r["codec"] == "vtm")
    t_metric = statistics.median([r["seconds_metrics"] for r in m])
    still = sum(r["seconds"] for r in m if r["codec"] != "vtm")
    n_still = sum(len(n_values[c]) for c in ("jpeg", "heif", "avif"))

    def t(q):  # linear between grid points, nearest outside
        if q <= grid[0][0]:
            return grid[0][1]
        if q >= grid[-1][0]:
            return grid[-1][1]
        for (a, ta), (b, tb) in zip(grid, grid[1:]):
            if a <= q <= b:
                return ta + (tb - ta) * (q - a) / (b - a)

    hvs = sorted((r["param"], r["psnr_hvs_m_db"]) for r in m if r["codec"] == "vtm")
    cross = next(a + (va - 42) * (b - a) / (va - vb) for (a, va), (b, vb) in zip(hvs, hvs[1:]) if va >= 42 > vb)
    whole_vtm = {"A": (16 * sum(t(q) for q in n_values["vtm"]), len(n_values["vtm"]))}
    lower = max(q for q, _ in grid if q <= cross)
    upper = min(q for q, _ in grid if q > lower)
    between = range(lower + 1, upper)
    whole_vtm["B"] = (16 * (sum(x for _, x in grid) + sum(t(q) for q in between)), len(grid) + len(between))
    whole_vtm["C"] = (16 * 8 * t(cross), 8)
    out = {}
    for name, (vtm_s, calls) in whole_vtm.items():
        hours = (total + vtm_s + 16 * still + 16 * (calls + n_still) * t_metric) / 3600
        out[name] = (hours, hours / k)
    return cross, out


def test_estimate_by_hand_with_median_not_mean(dry):
    n_values = {"vtm": [0, 10, 20], "jpeg": [50], "heif": [50], "avif": [50]}
    rows = []
    for u, unit in enumerate(("a", "b", "c")):
        for codec, params in n_values.items():
            for p in params:
                rows.append({"unit": unit, "level": "f", "codec": codec, "param": p, "status": "valid",
                             "seconds": (1, 2, 9)[u] * (1 + p / 10), "seconds_metrics": (0.1, 0.2, 0.9)[u] * (1 + p / 100)})
    for p, hvs, s in ((0, 50.0, 100.0), (8, 44.0, 60.0), (16, 40.0, 20.0)):
        rows.append({"unit": "mosaic", "level": "m", "codec": "vtm", "param": p, "status": "valid", "seconds": s,
                     "seconds_metrics": 1.0, "psnr_hvs_m_db": hvs})
    for codec, s in (("jpeg", 0.5), ("heif", 3.0), ("avif", 1.0)):
        rows.append({"unit": "mosaic", "level": "m", "codec": codec, "param": 50, "status": "valid", "seconds": s,
                     "seconds_metrics": 2.0, "psnr_hvs_m_db": 30.0})
    got = dry.estimate(rows, 3.5, n_values)
    cross, hand = hand_estimate(rows, 3.5, n_values)
    assert got["crossing_qp"] == pytest.approx(cross, rel=1e-12)  # 8 + (44 - 42) * 8 / 4 = 12
    for name in ("A", "B", "C"):
        assert got["variants"][name]["hours_1"] == pytest.approx(hand[name][0], rel=1e-12), name
        assert got["variants"][name]["hours_4"] == pytest.approx(hand[name][1], rel=1e-12), name
    assert got["variants"]["B"]["refine_qps"] == [9, 10, 11, 12, 13, 14, 15]


@pytest.fixture(scope="module")
def setup(dry):
    if not (ROOT / "lab_paths.json").is_file():
        pytest.skip("lab_paths.json is absent: the laboratory is not configured on this machine")
    try:
        paths = dry.load_paths()
    except dry.RunRefusal as refusal:
        pytest.skip(f"the coders of the dry run are not configured: {refusal}")
    cycle10_path = paths["cache_root"] / "cycles" / "10" / "e4_timing.json"
    if not cycle10_path.is_file():
        pytest.skip("the run file of cycle 10 is absent on this machine")
    cycle10 = json.loads(cycle10_path.read_text(encoding="utf-8"))
    units = dry.load_units(paths["images_root"], None, shape=(64, 96), names=["frisco.bmp", "DIEGO.BMP"])
    return paths, units, cycle10


PLAN = {"vtm": {"f": [50, 60], "m": [50, 60]}, "jpeg": [50, 100], "heif": [50, 100], "avif": [50, 100],
        "controls": [["frisco", "vtm", 50]], "mosaic_shape": [64, 96], "pool": 4}
STATE = {"core_commit": "test", "core_tree": "test"}


def rows_of(run_path):
    record = json.loads(Path(run_path).read_text(encoding="utf-8"))
    return record, {r["key"]: (r["status"], r["file_sha256"], r["psnr_db"], r["psnr_hvs_m_db"], r["haarpsi"],
                               r["cr_file"]) for r in record["rows"]}


def test_resume_after_stops_equals_an_uninterrupted_run(dry, setup, tmp_path):
    paths, units, cycle10 = setup
    whole, _ = dry.run(paths, units, PLAN, tmp_path / "whole", STATE, cycle10, log=lambda *a: None)
    out = tmp_path / "stopped"
    with pytest.raises(dry.RunStopped):
        dry.run(paths, units, PLAN, out, STATE, cycle10, stop_after=3, log=lambda *a: None)  # in the path control
    with pytest.raises(dry.RunStopped):
        dry.run(paths, units, PLAN, out, STATE, cycle10, stop_after=16, log=lambda *a: None)  # 14 + 2 in the pool
    assert not (out / dry.WORK_DIR).exists()
    with pytest.raises(dry.RunRefusal, match=r"another header: differs in \['plan'\]"):
        dry.run(paths, units, PLAN | {"vtm": {"f": [50, 61], "m": [50, 60]}}, out, STATE, cycle10, log=lambda *a: None)
    resumed, _ = dry.run(paths, units, PLAN, out, STATE, cycle10, log=lambda *a: None)
    record_w, rows_w = rows_of(whole)
    record_r, rows_r = rows_of(resumed)
    assert rows_r == rows_w and len(rows_w) == 3 * 6 + 2 * 2 + 2
    assert record_r["resumed_tasks"] == 3 + 16 and record_w["resumed_tasks"] == 0
    assert all(d["equal"] for d in record_r["summary"]["determinism"])
    assert any(r["status"] == "lossless" for r in record_r["rows"])  # AVIF 100 is lossless here
    with pytest.raises(dry.RunRefusal, match=dry.RUN_FILE):
        dry.run(paths, units, PLAN, out, STATE, cycle10, log=lambda *a: None)
    pool = sorted((r for r in record_w["rows"] if r.get("phase") == "pool f"), key=lambda r: r["order"])
    times = [dry.expected_seconds(cycle10, "f", r["param"]) for r in pool]
    assert times == sorted(times, reverse=True) and len(pool) == 3


def test_failed_task_before_a_stop_runs_again(dry, setup, tmp_path, monkeypatch):
    paths, units, cycle10 = setup
    whole, _ = dry.run(paths, units, PLAN, tmp_path / "whole", STATE, cycle10, log=lambda *a: None)
    real = dry.vtm_task

    def cut(paths_, unit, level, image, qp, work):  # the coder of this task is killed by the stop
        if (unit, qp) == ("frisco", 50):
            return dry.error_row(unit, level, "vtm", qp, RuntimeError("EncoderApp ended by the stop"))
        return real(paths_, unit, level, image, qp, work)

    monkeypatch.setattr(dry, "vtm_task", cut)
    out = tmp_path / "cut"
    with pytest.raises(dry.RunStopped):
        dry.run(paths, units, PLAN, out, STATE, cycle10, stop_after=17 + 3, log=lambda *a: None)  # path + pool (f)
    monkeypatch.setattr(dry, "vtm_task", real)
    resumed, _ = dry.run(paths, units, PLAN, out, STATE, cycle10, log=lambda *a: None)
    record_r, rows_r = rows_of(resumed)
    assert record_r["retried_tasks"] == 1 and rows_r["frisco|vtm|50"][0] == "valid"
    assert rows_r == rows_of(whole)[1]
