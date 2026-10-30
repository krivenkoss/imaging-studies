"""Lab tests of run_e5_fixed.py (study P001, STUDY.md section 7, «Дополнение E5»; cycle 26) on synthetic numbers and
synthetic sha256 - no pixel of P001; every expected value is written by hand.

- the fixed PCC by the direction of the scale: VVC - the smallest threshold PCC of the fragments, JPEG, HEIF, AVIF -
  the largest; whole images and another T do not count; a value other than the expected one, a fragment not judged
  or missing in E5 - a refusal in words;
- the points: a missing VVC point of a whole image is found, an error row counts as missing; a missing point of
  another codec or of a fragment is a refusal; the control - the first whole image that has the point in E4;
- per patient: the medians over the fragments (odd and even counts), the loss and r_fix; per whole image; the
  summary over the images - median, quartiles, min, max, n, r_fix > 1;
- the scale: different files per unit and the PCC equal to PCC - 1 on synthetic sha256 with a chain of three equal
  files and an error row, per level and over all units;
- end to end with a coder of the test's own: --plan encodes nothing and reads no pixel; the run encodes the control
  first and then the missing points with the arguments of E4 for whole images; a failed point stops it and the same
  command encodes only what is left; the run file and the journal hold no time of coding; the book is checked cell
  by cell; a second run is refused by name; a control that differs from E4 and a coder file of another sha256 are
  refused.
"""

import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.lab

STUDY = Path(__file__).resolve().parents[1]
EXPECTED = {"vtm": 25, "jpeg": 57, "heif": 43, "avif": 66}


@pytest.fixture(scope="module")
def fx():
    spec = importlib.util.spec_from_file_location("run_e5_fixed", STUDY / "run_e5_fixed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(data):
    return hashlib.sha256(data).hexdigest()


def e5_unit(unit, level, image, codec, pcc, t=42.0, judged=True, cr=10.0):
    return {"unit": unit, "level": level, "image": image, "codec": codec, "t_db": t, "judged": judged,
            "reason": None if judged else "не судится: пропуск у порога", "pcc": pcc if judged else None,
            "cr": cr if judged else None}


THRESHOLDS = {  # threshold PCC at 42 dB of the fragments: VVC 25 is the smallest, the others' largest 57, 43, 66
    "D01_r00_c00": {"vtm": 26, "jpeg": 50, "heif": 41, "avif": 60},
    "D01_r00_c01": {"vtm": 25, "jpeg": 57, "heif": 43, "avif": 66},
    "D02_r00_c00": {"vtm": 27, "jpeg": 45, "heif": 43, "avif": 58},
    "D02_r00_c01": {"vtm": 25, "jpeg": 40, "heif": 41, "avif": 57},
    "D02_r00_c02": {"vtm": 28, "jpeg": 52, "heif": 42, "avif": 61},
    "D03_r00_c00": {"vtm": 26, "jpeg": 38, "heif": 43, "avif": 63},
}
WHOLE = {"D01": {"vtm": 26, "jpeg": 45, "heif": 43, "avif": 60}, "D02": {"vtm": 28, "jpeg": 47, "heif": 41, "avif": 57},
         "D03": {"vtm": 27, "jpeg": 41, "heif": 43, "avif": 62}}
LEVELS = {"D01": "m", "D02": "m", "D03": "m", **{u: "f" for u in THRESHOLDS}}


def e5_units(extra=()):
    """E5 at 42 dB (whole images with values beyond the fragments' ones) and at 40 dB with other values."""
    out = [e5_unit(u, "f", u[:3], c, p, cr=8.0 + p / 10 + k) for k, (u, by) in enumerate(THRESHOLDS.items())
           for c, p in by.items()]
    out += [e5_unit(u, "m", u, c, p - 10 if c == "vtm" else p + 30, cr=12.0 + k) for k, (u, by) in enumerate(WHOLE.items())
            for c, p in by.items()]  # whole images: VVC 16-18 and quality 71-92 would win if they counted
    out += [e5_unit(u, "f", u[:3], c, p - 3 if c == "vtm" else p + 3, t=40.0) for u, by in THRESHOLDS.items()
            for c, p in by.items()]  # another T: VVC 22 and quality 60, 46, 69 would win if it counted
    return out + list(extra)


def test_fixed_pcc_by_the_direction_of_the_scale(fx):
    fragments = list(THRESHOLDS)
    assert fx.fixed_pcc(e5_units(), fragments, EXPECTED) == {"vtm": 25, "jpeg": 57, "heif": 43, "avif": 66}
    with pytest.raises(fx.RunRefusal, match=r"VVC \(VTM\): the fixed PCC by E5 is 25, 26 is expected"):
        fx.fixed_pcc(e5_units(), fragments, EXPECTED | {"vtm": 26})
    with pytest.raises(fx.RunRefusal, match="HEIF: the fixed PCC by E5 is 43, 44 is expected"):
        fx.fixed_pcc(e5_units(), fragments, EXPECTED | {"heif": 44})
    unjudged = [u if not (u["unit"] == "D02_r00_c02" and u["codec"] == "avif" and u["t_db"] == 42.0)
                else e5_unit(u["unit"], "f", "D02", "avif", None, judged=False) for u in e5_units()]
    with pytest.raises(fx.RunRefusal, match=r"AVIF: 1 fragments are not judged at 42 dB \(\['D02_r00_c02'\]"):
        fx.fixed_pcc(unjudged, fragments, EXPECTED)
    with pytest.raises(fx.RunRefusal, match="E5 has 6 fragments at 42 dB, E4 has 7"):
        fx.fixed_pcc(e5_units(), fragments + ["D04_r00_c00"], EXPECTED)


def e4_row(unit, codec, pcc, status="valid", file_sha=None, recon_sha=None, cr=10.0, psnr=45.0):
    return {"unit": unit, "codec": codec, "param": pcc, "status": status, "cr_file": cr, "psnr_hvs_m_db": psnr,
            "psnr_db": 40.0, "haarpsi": 0.95, "bpp": 8 / cr, "file_sha256": file_sha or sha(f"{unit}{codec}{pcc}".encode()),
            "recon_sha256": recon_sha or sha(f"r{unit}{codec}{pcc}".encode())}


def test_missing_points_and_the_control(fx):
    levels = {"W1": "m", "W2": "m", "F1": "f", "F2": "f"}
    rows = {(u, c, EXPECTED[c]): e4_row(u, c, EXPECTED[c]) for u in levels for c in fx.CODECS}
    del rows[("W2", "vtm", 25)]
    rows[("W2", "vtm", 24)], rows[("W2", "vtm", 28)] = e4_row("W2", "vtm", 24), e4_row("W2", "vtm", 28)
    assert fx.missing_points(levels, rows, EXPECTED) == [("W2", "vtm", 25)]
    assert fx.control_point(levels, rows, EXPECTED) == ("W1", "vtm", 25)
    failed = rows | {("W1", "vtm", 25): e4_row("W1", "vtm", 25, status="error")}
    assert fx.missing_points(levels, failed, EXPECTED) == [("W1", "vtm", 25), ("W2", "vtm", 25)]
    with pytest.raises(fx.RunRefusal, match="no whole image has VVC at PCC 25 in E4"):
        fx.control_point(levels, failed, EXPECTED)
    no_jpeg = {k: v for k, v in rows.items() if k != ("F1", "jpeg", 57)}
    with pytest.raises(fx.RunRefusal, match="1 points at the fixed PCC are missing in E4 outside VVC of whole images: "
                                            "F1 JPEG PCC 57"):
        fx.missing_points(levels, no_jpeg, EXPECTED)
    no_fragment = {k: v for k, v in rows.items() if k != ("F2", "vtm", 25)}
    with pytest.raises(fx.RunRefusal, match=r"F2 VVC \(VTM\) PCC 25"):
        fx.missing_points(levels, no_fragment, EXPECTED)
    lossless = rows | {("F1", "avif", 66): e4_row("F1", "avif", 66, status="lossless", psnr="+inf")}
    assert fx.missing_points(levels, lossless, EXPECTED) == [("W2", "vtm", 25)]  # a lossless point is a point


def point(image, codec, cr_fix, cr_own, level="f"):
    return {"level": level, "image": image, "codec": codec, "cr_fix": cr_fix, "cr_own": cr_own}


# per patient and codec: (CR at the fixed PCC, CR at the own threshold) of its fragments; medians by hand below
PATIENTS = {
    "P1": {"vtm": [(30, 40), (10, 25), (20, 30)], "jpeg": [(12, 15), (8, 10), (16, 20)],
           "heif": [(25, 25), (20, 20), (30, 30)], "avif": [(16, 20), (18, 20), (14, 20)]},
    "P2": {"vtm": [(10, 16), (14, 14)], "jpeg": [(6, 8), (10, 12)], "heif": [(10, 12), (14, 12)],
           "avif": [(9, 10), (11, 15)]},
    "P3": {"vtm": [(9, 10)], "jpeg": [(10, 10)], "heif": [(12, 12)], "avif": [(6, 8)]},
}


def test_patients_loss_and_r_fix_by_hand(fx):
    points = [point(p, c, float(a), float(b)) for p, by in PATIENTS.items() for c, pairs in by.items() for a, b in pairs]
    table = fx.per_image(points, "f", ["P1", "P2", "P3"])
    got = {(x["image"], x["codec"]): (x["n"], x["cr_fix"], x["cr_own"], x["loss"]) for x in table}
    assert got[("P1", "vtm")] == (3, 20.0, 30.0, pytest.approx(2 / 3))
    assert got[("P2", "vtm")] == (2, 12.0, 15.0, pytest.approx(0.8))  # an even count: the mean of the middle two
    assert got[("P2", "avif")] == (2, 10.0, 12.5, pytest.approx(0.8))
    assert got[("P3", "heif")] == (1, 12.0, 12.0, 1.0)
    assert {x["level"] for x in table} == {"пациенты"}
    r = {(x["image"], x["pair"]): x["r_fix"] for x in fx.ratios(table)}
    assert r == pytest.approx({("P1", "vtm/jpeg"): 20 / 12, ("P1", "vtm/heif"): 0.8, ("P1", "vtm/avif"): 1.25,
                               ("P2", "vtm/jpeg"): 1.5, ("P2", "vtm/heif"): 1.0, ("P2", "vtm/avif"): 1.2,
                               ("P3", "vtm/jpeg"): 0.9, ("P3", "vtm/heif"): 0.75, ("P3", "vtm/avif"): 1.5})
    s = {x["codec"]: x for x in fx.summary(table, fx.ratios(table))}
    assert (s["vtm"]["n"], s["vtm"]["median"], s["vtm"]["min"], s["vtm"]["max"]) == (3, pytest.approx(0.8),
                                                                                    pytest.approx(2 / 3), pytest.approx(0.9))
    assert s["vtm"]["above_1"] is None and s["avif"]["min"] == pytest.approx(0.75)
    assert (s["vtm/jpeg"]["median"], s["vtm/jpeg"]["min"], s["vtm/jpeg"]["max"], s["vtm/jpeg"]["above_1"]) == \
        (pytest.approx(1.5), pytest.approx(0.9), pytest.approx(20 / 12), 2)
    assert (s["vtm/jpeg"]["q25"], s["vtm/jpeg"]["q75"]) == (pytest.approx(1.2), pytest.approx(1.5 + (20 / 12 - 1.5) / 2))
    assert s["vtm/heif"]["above_1"] == 0 and s["vtm/avif"]["above_1"] == 3  # r_fix = 1.0 is not above 1
    whole = fx.per_image([point("W1", c, 20.0, 25.0 if c != "heif" else None, "m") for c in fx.CODECS], "m", ["W1"])
    w = {x["codec"]: x for x in whole}
    assert (w["vtm"]["level"], w["vtm"]["n"], w["vtm"]["loss"]) == ("снимки", 1, pytest.approx(0.8))
    assert (w["heif"]["own_n"], w["heif"]["cr_own"], w["heif"]["loss"]) == (0, None, None)  # no own threshold: no loss
    sw = {x["codec"]: x for x in fx.summary(whole, fx.ratios(whole))}
    assert (sw["heif"]["n"], sw["heif"]["median"], sw["vtm"]["n"], sw["vtm/heif"]["median"]) == (0, None, 1, 1.0)


def scale_rows(unit, codec, values, equal, errors=()):
    """Synthetic sha256 of a sweep: the file at q is the file at q - 1 for q in `equal` (a chain carries on)."""
    label = {q: q for q in values}
    for q in sorted(equal):
        label[q] = label[q - 1]
    return [{"unit": unit, "codec": codec, "param": q, "file_sha256": None if q in errors else
             sha(f"{unit}|{codec}|{label[q]}".encode())} for q in values]


def test_scale_counts_different_files_and_equal_neighbours(fx):
    rows = (scale_rows("F1", "heif", range(11), [2, 6, 9, 10]) + scale_rows("W1", "heif", range(11), [2, 6, 10])
            + scale_rows("F1", "jpeg", range(1, 6), []) + scale_rows("W1", "jpeg", range(1, 6), [])
            + scale_rows("F1", "avif", range(4), [3], errors=[1]) + scale_rows("W1", "avif", range(4), [3])
            + [{"unit": "F1", "codec": "vtm", "param": q, "file_sha256": "0" * 64} for q in range(5)])
    units, table = fx.scale(rows, {"F1": "f", "W1": "m"})
    got = {(u["unit"], u["codec"]): (u["values"], u["files"], u["equal_previous"]) for u in units}
    assert got == {("F1", "heif"): (11, 7, [2, 6, 9, 10]), ("W1", "heif"): (11, 8, [2, 6, 10]),
                   ("F1", "jpeg"): (5, 5, []), ("W1", "jpeg"): (5, 5, []),
                   ("F1", "avif"): (3, 2, [3]), ("W1", "avif"): (4, 3, [3])}  # VVC is not on the scale
    by = {(x["codec"], x["level"]): x for x in table}
    assert by[("heif", "фрагменты")]["equal_previous_all"] == [2, 6, 9, 10]
    assert by[("heif", "все")]["equal_previous_all"] == [2, 6, 10]
    assert (by[("heif", "все")]["files_per_unit"], by[("heif", "все")]["values_per_unit"], by[("heif", "все")]["n"]) == \
        ({"7": 1, "8": 1}, {"11": 2}, 2)
    assert by[("jpeg", "снимки")]["equal_previous_all"] == [] and by[("avif", "все")]["files_per_unit"] == {"2": 1, "3": 1}


# ---------------------------------------------------------------- end to end

def image_of(code):
    rng = np.random.default_rng(int(code[1:]))
    return rng.integers(0, 256, (16, 24), dtype=np.uint8)


def coded(unit, qp, image):
    """The test's own VVC point: a file and a reconstruction from the code, QP and pixels."""
    file = f"{unit}|vvc|{qp}|".encode() + image[:1, :4].tobytes()
    return file, (image // 2) * 2


def synthetic_run(tmp, control_sha=None):
    """E4 and E5 files, coder files and the pins; D02 and D03 have no VVC point at 25 (grid 24, 28)."""
    paths = {}
    for key in ("encoder", "decoder", "cfg", "mp4box", "gpac"):
        paths[key] = tmp / f"{key}.bin"
        paths[key].write_bytes(key.encode() * 3)
    rows = []
    for k, (u, level) in enumerate(LEVELS.items()):
        for c in ("vtm", "jpeg", "heif", "avif"):
            if c == "vtm" and level == "m" and u != "D01":
                rows += [e4_row(u, c, 24, cr=7.0), e4_row(u, c, 28, cr=9.0)]
                continue
            status, psnr = ("lossless", "+inf") if (u, c) == ("D02_r00_c02", "jpeg") else ("valid", 44.0 + k / 10)
            file_sha = recon_sha = None
            if (u, c) == ("D01", "vtm"):
                file, recon = coded("D01", 25, image_of("D01"))
                file_sha, recon_sha = control_sha or sha(file), sha(recon.tobytes())
            rows.append(e4_row(u, c, EXPECTED[c], status, file_sha, recon_sha, cr=6.0 + k + "vjha".index(c[0]), psnr=psnr))
    rows += [e4_row(u, "heif", 44, file_sha=sha(f"{u}heif43".encode())) for u in LEVELS]  # HEIF 44 = 43: on the scale
    e4 = {"header": {"codec_versions": {"vtm_gpac_sha256": {k: sha(p.read_bytes()) for k, p in paths.items()}}},
          "units": LEVELS, "rows": rows}
    (tmp / "e4.json").write_text(json.dumps(e4), encoding="utf-8")
    units = e5_units()
    patients = [{"image": im, "codec": c, "t_db": 42.0,
                 "median_cr": statistics.median(u["cr"] for u in units if u["level"] == "f" and u["image"] == im
                                                and u["codec"] == c and u["t_db"] == 42.0)}
                for im in WHOLE for c in ("vtm", "jpeg", "heif", "avif")]
    e5 = {"inputs": {"e4_run_sha256": sha((tmp / "e4.json").read_bytes())},
          "analysis": {"units": units, "patients": patients}}
    (tmp / "e5.json").write_text(json.dumps(e5, ensure_ascii=False), encoding="utf-8")
    pins = (sha((tmp / "e4.json").read_bytes()), sha((tmp / "e5.json").read_bytes()))
    return paths, pins, e4, e5


class Coder:
    """The test's own vtm_task of e4_book: the arguments of E4 for whole images, a failure on demand, the calls."""
    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def __call__(self, paths, unit, level, image, qp, work, timeout_s):
        assert (level, timeout_s, image.dtype, Path(work).is_dir()) == ("m", 3600, np.uint8, True)
        assert np.array_equal(image, image_of(unit))
        self.calls.append(unit)
        base = {"key": f"{unit}|vtm|{qp}", "unit": unit, "level": level, "codec": "vtm", "param": qp}
        if unit in self.fail:
            self.fail.discard(unit)
            return base | {"status": "error", "detail": "CalledProcessError: EncoderApp returned 1", "file_sha256": None,
                           "recon_sha256": None, "seconds": None, "seconds_metrics": None}
        file, recon = coded(unit, qp, image)
        return base | {"file_bytes": len(file), "cr_file": image.size / len(file), "bpp": 8 * len(file) / image.size,
                       "seconds": 9.5, "seconds_b05": 8.0, "seconds_b09": 1.5, "seconds_metrics": 0.25,
                       "file_sha256": sha(file), "recon_sha256": sha(recon.tobytes()), "detail": None,
                       "bitstream_sha256": sha(file[:5]), "psnr_db": 39.0, "psnr_hvs_m_db": 41.8 if unit == "D03" else 44.5,
                       "haarpsi": 0.97, "status": "valid"}


def loader(asked):
    def arrays_of(codes):
        asked.append(list(codes))
        return {c: image_of(c) for c in codes}
    return arrays_of


STATE = {"core_commit": "c" * 40, "core_tree": "d" * 40}


def test_end_to_end_plan_run_resume_book_and_refusal_by_name(fx, tmp_path, monkeypatch):
    from openpyxl import load_workbook

    paths, pins, e4, e5 = synthetic_run(tmp_path)
    out = tmp_path / "cycles" / "26"
    coder, asked, lines = Coder(fail={"D03"}), [], []
    monkeypatch.setattr(fx.e4_book, "vtm_task", coder)
    args = (tmp_path / "e4.json", tmp_path / "e5.json", out, STATE, paths, loader(asked))
    assert fx.run(*args, pins=pins, plan_only=True, log=lines.append) is None
    assert (coder.calls, asked) == ([], [])
    assert "Missing VVC points of whole images at PCC 25: 2 - D02, D03" in lines and "Journal: 0 of 3 points finished" in lines
    with pytest.raises(fx.RunRefusal, match="D03 VVC PCC 25: the coder failed .*the same command encodes it again"):
        fx.run(*args, pins=pins, log=lines.append)
    assert coder.calls == ["D01", "D02", "D03"] and asked == [["D01", "D02", "D03"]]  # the control first
    assert not (out / fx.RUN_FILE).exists()
    lines.clear()
    fx.run(*args, pins=pins, plan_only=True, log=lines.append)
    assert "Journal: 2 of 3 points finished" in lines
    run_path, book_path = fx.run(*args, pins=pins, log=lines.append)
    assert coder.calls == ["D01", "D02", "D03", "D03"] and asked[-1] == ["D03"]  # only what was left
    text, journal = run_path.read_text(encoding="utf-8"), (out / fx.JOURNAL_FILE).read_text(encoding="utf-8")
    assert '"seconds' not in text and '"seconds' not in journal  # no time of coding is kept
    record = json.loads(text)
    a = record["analysis"]
    assert a["fixed_pcc"] == EXPECTED
    assert [(p["role"], p["unit"], p["param"]) for p in a["new_points"]] == [("контроль", "D01", 25), ("новая", "D02", 25),
                                                                             ("новая", "D03", 25)]
    (control,) = [p for p in a["new_points"] if p["role"] == "контроль"]
    assert control["file_equal_e4"] is True and control["recon_equal_e4"] is True
    at = {(p["unit"], p["codec"]): p for p in a["points"]}
    assert len(a["points"]) == 9 * 4
    assert (at[("D02", "vtm")]["source"], at[("D01", "vtm")]["source"], at[("D02_r00_c00", "heif")]["source"]) == \
        ("цикл 26", "E4", "E4")
    file, _ = coded("D02", 25, image_of("D02"))
    assert at[("D02", "vtm")]["cr_fix"] == pytest.approx(16 * 24 / len(file))
    assert at[("D02_r00_c02", "jpeg")]["below_t"] is False and at[("D03", "vtm")]["below_t"] is True
    below = {(x["level"], x["codec"]): x for x in a["below"]}
    assert (below[("снимки", "vtm")]["below"], below[("снимки", "vtm")]["units"]) == (1, ["D03"])
    assert below[("фрагменты", "vtm")] == {"level": "фрагменты", "codec": "vtm", "n": 6, "below": 0, "units": [],
                                           "images": []}
    # the patient D02: three fragments; CR at the fixed PCC by e4_row of synthetic_run, the own CR by e5_units
    patients = {(x["image"], x["codec"]): x for x in a["patients"]}
    fix = [6.0 + k for k in (5, 6, 7)]  # VVC: the fragments of D02 are the 6th to 8th units of LEVELS
    own = [8.0 + THRESHOLDS[u]["vtm"] / 10 + k for k, u in enumerate(THRESHOLDS) if u.startswith("D02")]
    assert (patients[("D02", "vtm")]["cr_fix"], patients[("D02", "vtm")]["cr_own"]) == (fix[1], sorted(own)[1])
    r = {(x["level"], x["image"], x["pair"]): x["r_fix"] for x in a["ratios"]}
    assert r[("пациенты", "D02", "vtm/jpeg")] == pytest.approx(fix[1] / (fix[1] + 1))
    heif = {x["level"]: x for x in a["scale"]["summary"] if x["codec"] == "heif"}
    assert heif["все"]["equal_previous_all"] == [44] and heif["все"]["files_per_unit"] == {"1": 9}
    book = load_workbook(book_path)
    assert book.sheetnames == list(fx.SHEETS)
    assert book["Описание"]["B1"].value == sha(run_path.read_bytes())
    assert book["Описание"]["A2"].value == "Вопрос" and "Время кодирования не хранится" in json.dumps(
        [c.value for row in book["Описание"].iter_rows() for c in row], ensure_ascii=False)
    with pytest.raises(fx.RunRefusal, match="already exists: the run file and the book of cycle 26 are never overwritten"):
        fx.run(*args, pins=pins, log=lines.append)


def test_a_control_that_differs_from_e4_and_another_coder_are_refused(fx, tmp_path, monkeypatch):
    paths, pins, _, _ = synthetic_run(tmp_path, control_sha="0" * 64)
    coder = Coder()
    monkeypatch.setattr(fx.e4_book, "vtm_task", coder)
    args = (tmp_path / "e4.json", tmp_path / "e5.json", tmp_path / "out", STATE, paths, loader([]))
    with pytest.raises(fx.RunRefusal, match="control D01 VVC PCC 25: encoded again, the file differs from E4 by sha256"):
        fx.run(*args, pins=pins, log=lambda *a: None)
    assert coder.calls == ["D01"]  # the long part does not start
    paths["mp4box"].write_bytes(b"another build")
    with pytest.raises(fx.RunRefusal, match="the coder file 'mp4box' has sha256 .*, E4 was encoded with"):
        fx.run(*args, pins=pins, log=lambda *a: None)
    with pytest.raises(fx.RunRefusal, match="the run file of E5 has sha256 .*, the addendum is declared for"):
        fx.run(*args, pins=(pins[0], "f" * 64), log=lambda *a: None)
