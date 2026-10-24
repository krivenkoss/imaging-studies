"""Lab tests of run_e5.py (study P001, E5 declared in STUDY.md section 6) on synthetic numbers - no pixel of P001.

- the threshold walk on known curves: VVC upward, the others downward; lossless above; "порог не достигнут",
  "порог выше наилучшего качества", "пропуск у порога" (an error between the last point >= T and the first < T,
  and one before any point); an error after the first point < T changes nothing;
- whole images, VVC at 40 and 41 dB: CR interpolated in ln CR between two grid nodes, by hand;
- ratios and the test on given numbers: medians per patient, r, a patient out of the test with its reason, B14 and
  Holm; the verdict by the rule of section 6 (CI above 1 and p Holm < 0.05; otherwise not established);
- JPEG with optimize=True: the reconstruction is checked against the E4 row; a substitute coder with another
  reconstruction is refused in words;
- end to end: a synthetic copy of 8 "patients" (one image and two fragments each), an E4 run file with real JPEG
  rows and given VVC, HEIF, AVIF curves, a stand-in IrfanView file -> the run file and the book, checked cell by
  cell; VVC better than JPEG and HEIF by the rule, not established against AVIF; privacy - codes only, no pixels.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.lab

STUDY = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def e5():
    spec = importlib.util.spec_from_file_location("run_e5", STUDY / "run_e5.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def point(codec, param, psnr_hvs_m, cr=10.0, status="valid"):
    return {"unit": "U", "codec": codec, "param": param, "status": status, "psnr_hvs_m_db": psnr_hvs_m, "cr_file": cr,
            "psnr_db": psnr_hvs_m - 3 if isinstance(psnr_hvs_m, float) else psnr_hvs_m, "haarpsi": 0.9, "bpp": 8 / cr}


def test_threshold_walk_on_known_curves(e5):
    vvc = [point("vtm", q, 50.0 - 0.8 * (q - 20)) for q in range(20, 41)]  # >= 42 up to QP 30
    row, reason = e5.threshold("vtm", vvc, 42.0)
    assert (row["param"], reason) == (30, None)
    jpeg = [point("jpeg", q, 30.0 + 0.2 * q) for q in range(1, 101)]  # >= 42 down to quality 60
    assert e5.threshold("jpeg", jpeg, 42.0)[0]["param"] == 60
    assert e5.threshold("jpeg", list(reversed(jpeg)), 42.0)[0]["param"] == 60  # the order of the rows does not matter
    assert e5.threshold("vtm", [point("vtm", q, 45.0) for q in range(5)], 42.0) == (None, e5.NOT_REACHED)
    assert e5.threshold("vtm", [point("vtm", q, 40.0) for q in range(5)], 42.0) == (None, e5.ABOVE_BEST)
    lossless = [point("avif", 100, "+inf", status="lossless")] + [point("avif", q, 41.0) for q in range(99, 90, -1)]
    assert e5.threshold("avif", lossless, 42.0)[0]["param"] == 100  # lossless counts as above the threshold
    gap = [p if p["param"] != 31 else point("vtm", 31, None, status="error") for p in vvc]
    assert e5.threshold("vtm", gap, 42.0) == (None, e5.GAP)
    late = [p if p["param"] != 35 else point("vtm", 35, None, status="error") for p in vvc]
    assert e5.threshold("vtm", late, 42.0)[0]["param"] == 30  # an error after the first point < T changes nothing
    early = [point("vtm", 19, None, status="error")] + [point("vtm", q, 41.0 - q / 100) for q in range(20, 25)]
    assert e5.threshold("vtm", early, 42.0) == (None, e5.GAP)  # an error before any point: a gap at the threshold
    non_monotone = [point("vtm", q, {0: 45.0, 4: 41.0, 8: 43.0}.get(q, 30.0)) for q in (0, 4, 8, 12)]
    assert e5.threshold("vtm", non_monotone, 42.0)[0]["param"] == 0  # the first crossing from the high-quality side


def test_interpolation_between_grid_nodes_by_hand(e5):
    cr = lambda q: 10.0 * math.exp(0.08 * (q + 12))  # noqa: E731
    grid = [point("vtm", q, 60.0 - 0.6 * (q + 12), cr(q)) for q in range(-12, 61, 4)]
    node, _ = e5.threshold("vtm", grid, 40.0)
    assert node["param"] == 20  # 40.8 dB at QP 20, 38.4 at 24
    got, words = e5.interpolated("vtm", grid, node, 40.0)
    f = (40.0 - 40.8) / (38.4 - 40.8)
    assert got == pytest.approx(math.exp(math.log(cr(20)) + f * (math.log(cr(24)) - math.log(cr(20)))), rel=1e-12)
    assert words == "между PCC 20 и 24"
    refined = sorted(grid + [point("vtm", q, 60.0 - 0.6 * (q + 12), cr(q)) for q in (21, 22, 23)], key=lambda p: p["param"])
    node, _ = e5.threshold("vtm", refined, 40.0)
    assert node["param"] == 21 and e5.interpolated("vtm", refined, node, 40.0)[0] is None  # refinement: the node


def units_for(e5, crs, judged=None):
    """Synthetic «Единицы»: crs[image][codec] = (CR of fragment 1, CR of fragment 2) at every T."""
    out = []
    for image, by_codec in crs.items():
        for codec, values in by_codec.items():
            for k, cr in enumerate(values):
                for t in e5.THRESHOLDS:
                    ok = judged is None or judged(image, codec, k)
                    out.append({"unit": f"{image}_r0{k}_c00", "level": "f", "image": image, "codec": codec, "t_db": t,
                                "judged": ok, "reason": None if ok else e5.GAP, "cr": cr if ok else None,
                                "cr_opt": cr * 1.05 if ok and codec == "jpeg" else None})
    return out


def test_ratios_and_the_comparison_on_given_numbers(e5):
    images = [f"D0{k}" for k in range(1, 9)]
    crs = {im: {"vtm": (20.0 + k, 22.0 + k), "jpeg": (5.0, 6.0 + 0.1 * k), "heif": (10.0, 11.0 + 0.1 * k),
                "avif": ((20.0 + k) * (1.2 + 0.01 * k if k % 2 else 0.8 - 0.01 * k),
                         (22.0 + k) * (1.2 + 0.01 * k if k % 2 else 0.8 - 0.01 * k))}
           for k, im in enumerate(images)}
    units = units_for(e5, crs, judged=lambda image, codec, k: not (image == "D01" and codec == "jpeg"))
    patients = e5.patients_table(units, images)
    by = {(p["image"], p["codec"], p["t_db"]): p for p in patients}
    assert by[("D03", "vtm", 42.0)] == {"image": "D03", "codec": "vtm", "t_db": 42.0, "judged": 2, "of": 2,
                                        "median_cr": (22.0 + 24.0) / 2}
    assert by[("D01", "jpeg", 42.0)]["judged"] == 0 and by[("D01", "jpeg", 42.0)]["median_cr"] is None
    medians = {k: p["median_cr"] for k, p in by.items()}
    ratios = e5.ratios_table(medians, images, "пациенты")
    r = {(x["image"], x["pair"], x["t_db"]): x for x in ratios}
    assert r[("D01", "vtm/jpeg", 42.0)]["in_test"] is False and r[("D01", "vtm/jpeg", 42.0)]["reason"] == "нет судимых единиц JPEG"
    assert r[("D03", "vtm/heif", 42.0)]["r"] == pytest.approx(23.0 / ((10.0 + 11.2) / 2), rel=1e-12)
    comparisons = {(c["pair"], c["t_db"]): c for c in e5.compare(ratios, "пациенты")}
    jpeg, heif, avif = (comparisons[(f"vtm/{c}", 42.0)] for c in ("jpeg", "heif", "avif"))
    assert jpeg["n"] == 7 and heif["n"] == 8 and avif["n"] == 8
    d = np.array([x["ln_r"] for x in ratios if x["pair"] == "vtm/heif" and x["t_db"] == 42.0 and x["in_test"]])
    own = e5.paired.wilcoxon_hl(d)
    assert (heif["w_plus"], heif["p"], heif["ratio"], heif["ratio_low"]) == (36.0, own.p_value, own.ratio, own.ratio_ci95[0])
    assert (jpeg["p_holm"], heif["p_holm"], avif["p_holm"]) == e5.paired.holm([jpeg["p"], heif["p"], avif["p"]])
    assert heif["p"] == pytest.approx(2 / 256) and heif["p_holm"] < 0.05 and heif["ratio_low"] > 1
    assert heif["verdict"] == "VVC выгоднее HEIF" and heif["role"] == "основное"
    assert jpeg["p"] == pytest.approx(2 / 128) and jpeg["p_holm"] < 0.05 and jpeg["verdict"] == "VVC выгоднее JPEG"
    assert avif["verdict"].startswith("«VVC выгоднее AVIF» не установлено") and avif["ratio_low"] < 1
    assert comparisons[("vtm/heif", 40.0)]["verdict"] is None and comparisons[("vtm/heif", 40.0)]["role"] == "описательно"
    assert comparisons[("vtm/jpeg_opt", 42.0)]["p_holm"] is None  # outside the family of the three comparisons


def test_huffman_reencoding_checks_the_reconstruction(e5, monkeypatch):
    from imstudies.lab.coders import jpeg

    image = np.random.default_rng(5).integers(0, 256, (24, 40)).astype(np.uint8)
    rows = {}
    for q in (10, 50, 90):
        c = jpeg.encode_decode(image, q, max_pixels=image.size)
        rows[("U", q)] = {"file_bytes": len(c.file), "cr_file": image.size / len(c.file), "recon_sha256": e5.recon_sha256(c.recon)}
    out = e5.huffman({"U": image}, rows)
    assert [o["pcc"] for o in out] == [10, 50, 90] and all(o["recon_equal"] for o in out)
    assert all(o["file_bytes_opt"] <= o["file_bytes"] and o["ratio"] == o["cr_opt"] / o["cr"] for o in out)

    real = e5.encode_opt

    def unequal(img, quality):
        file, recon = real(img, quality)
        recon = recon.copy()
        recon[0, 0] ^= 1
        return file, recon
    monkeypatch.setattr(e5, "encode_opt", unequal)
    with pytest.raises(e5.RunRefusal, match=r"U, JPEG PCC 10: the reconstruction with optimize=True has sha256"):
        e5.huffman({"U": image}, rows)


# ---------------------------------------------------------------- end to end

N_IMAGES = 8


def curve_rows(unit, level, codec, k, arrays):
    """VVC, HEIF, AVIF rows with given curves; JPEG rows encoded for real (the run re-encodes them)."""
    from imstudies.lab.coders import jpeg

    rows = []
    if codec == "vtm":
        qps = range(-12, 61, 4) if level == "m" else range(20, 41)
        for q in qps:
            hvs = 60.0 - 0.6 * (q + 12) if level == "m" else 50.0 - 0.8 * (q - 20)
            if unit == "D01_r00_c00" and q == 31:
                rows.append({"codec": codec, "param": q, "status": "error", "psnr_hvs_m_db": None, "cr_file": None,
                             "psnr_db": None, "haarpsi": None, "bpp": None})
                continue
            rows.append({"codec": codec, "param": q, "status": "valid", "psnr_hvs_m_db": hvs, "psnr_db": hvs - 4,
                         "haarpsi": 0.8, "cr_file": (20.0 + k) * math.exp(0.08 * (q - 20)), "bpp": 0.5})
    elif codec == "jpeg":
        for q in range(1, 101):
            c = jpeg.encode_decode(arrays[unit], q, max_pixels=arrays[unit].size)
            rows.append({"codec": codec, "param": q, "status": "valid", "psnr_hvs_m_db": 30.0 + 0.2 * q,
                         "psnr_db": 27.0 + 0.2 * q, "haarpsi": 0.5 + q / 400, "cr_file": arrays[unit].size / len(c.file),
                         "bpp": 8 * len(c.file) / arrays[unit].size, "file_bytes": len(c.file),
                         "recon_sha256": hashlib.sha256(np.ascontiguousarray(c.recon).tobytes()).hexdigest()})
    else:
        for q in range(0, 101):
            lossless = codec == "avif" and q == 100
            factor = (1.2 + 0.01 * k if k % 2 else 0.8 - 0.01 * k) if codec == "avif" else 0.25 + 0.01 * k
            vvc_cr = (20.0 + k) * math.exp(0.08 * 10)  # the CR of VVC at its threshold (QP 30 on fragments)
            rows.append({"codec": codec, "param": q, "status": "lossless" if lossless else "valid",
                         "psnr_hvs_m_db": "+inf" if lossless else 30.0 + 0.2 * q, "psnr_db": "+inf" if lossless else 27.0,
                         "haarpsi": 1.0 if lossless else 0.7, "cr_file": vvc_cr * factor * (1 + (100 - q) / 1000),
                         "bpp": 0.4})
    return [r | {"unit": unit, "level": level, "key": f"{unit}|{codec}|{r['param']}"} for r in rows]


@pytest.fixture()
def world(tmp_path):
    """A synthetic copy (registry, PNGs, grid), an E4 run file, images_root with frisco.bmp, a stand-in IrfanView file."""
    from PIL import Image

    from imstudies.lab.coders import jpeg

    rng = np.random.default_rng(15)
    (tmp_path / "data" / "P001").mkdir(parents=True)
    (tmp_path / "data" / "registry").mkdir()
    entries, tiles, levels, arrays = [], [], {}, {}
    for k in range(N_IMAGES):
        code = f"D{k + 1:02d}"
        values = rng.integers(0, 256, (64, 96)).astype(np.uint8)
        path = tmp_path / "data" / "P001" / f"{code}.png"
        Image.fromarray(values, "L").save(path)
        entries.append({"code": code, "H": 64, "W": 96, "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "full_tiles": 2})
        levels[code], arrays[code] = "m", values
        for r in (0, 1):
            tile = f"{code}_r0{r}_c00"
            tiles.append({"tile": tile, "code": code, "r": r, "c": 0, "x": 32, "y": 32 * r})
            levels[tile], arrays[tile] = "f", values[32 * r:32 * r + 32, 32:64]
    registry = tmp_path / "data" / "registry" / "P001.json"
    registry.write_text(json.dumps({"contract": "B10 0.2.0", "images": entries}), encoding="utf-8")
    grid = tmp_path / "p001_tiles.json"
    grid.write_text(json.dumps({"rows": tiles, "parameters": {"tile": 32}}), encoding="utf-8")
    rows = []
    for unit, level in levels.items():
        k = int(unit[1:3]) - 1
        for codec in ("vtm", "jpeg", "heif", "avif"):
            rows += curve_rows(unit, level, codec, k, arrays)
    e4_path = tmp_path / "e4_main.json"
    e4_path.write_text(json.dumps({"rows": rows, "units": levels}), encoding="utf-8")
    (tmp_path / "images" / "BMP").mkdir(parents=True)
    frisco = rng.integers(0, 256, (512, 512)).astype(np.uint8)
    Image.fromarray(frisco, "L").save(tmp_path / "images" / "BMP" / "frisco.bmp")
    (tmp_path / "fidler").mkdir()
    import io
    buf = io.BytesIO()
    Image.fromarray(frisco, "L").save(buf, "JPEG", quality=50, optimize=True, progressive=False)
    (tmp_path / "fidler" / "frisco_irfanview-9.99_q50.jpg").write_bytes(buf.getvalue())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    return {"e4": e4_path, "data": tmp_path / "data", "grid": grid, "images": tmp_path / "images",
            "fidler": tmp_path / "fidler", "out": tmp_path / "cycles15", "pins": (sha(registry), sha(grid)),
            "e4_sha": sha(e4_path), "ours50": jpeg.encode_decode(frisco, 50, max_pixels=frisco.size).file}


def run_world(e5, w, **kw):
    return e5.run(w["e4"], w["data"], w["grid"], w["images"], w["fidler"], w["out"],
                  {"core_commit": "test", "core_tree": "test"}, e4_pin=w["e4_sha"], registry_pins=w["pins"],
                  log=lambda *a: None, **kw)


def longest_int_list(obj):
    if isinstance(obj, list):
        own = len(obj) if obj and all(isinstance(x, int) and not isinstance(x, bool) for x in obj) else 0
        return max([own] + [longest_int_list(x) for x in obj])
    if isinstance(obj, dict):
        return max([0] + [longest_int_list(x) for x in obj.values()])
    return 0


def test_end_to_end_run_file_and_book(e5, world):
    run_path, book_path = run_world(e5, world)
    record = json.loads(run_path.read_text(encoding="utf-8"))
    a = record["analysis"]
    verdict = {v["pair"]: v["verdict"] for v in a["verdict"]["pairs"]}
    assert verdict["vtm/jpeg"] == "VVC выгоднее JPEG" and verdict["vtm/heif"] == "VVC выгоднее HEIF"
    assert verdict["vtm/avif"].startswith("«VVC выгоднее AVIF» не установлено")
    assert a["verdict"]["all"] == "«VVC выгоднее всех» не установлено"
    gap = next(u for u in a["units"] if u["unit"] == "D01_r00_c00" and u["codec"] == "vtm" and u["t_db"] == 42.0)
    assert (gap["judged"], gap["reason"]) == (False, e5.GAP)
    d01 = next(p for p in a["patients"] if p["image"] == "D01" and p["codec"] == "vtm" and p["t_db"] == 42.0)
    assert (d01["judged"], d01["of"]) == (1, 2)
    whole = next(u for u in a["units"] if u["unit"] == "D02" and u["codec"] == "vtm" and u["t_db"] == 40.0)
    assert whole["pcc"] == 20 and whole["cr_interp"] is not None and whole["interp_note"] == "между PCC 20 и 24"
    assert a["huffman"]["recon_equal"] == a["huffman"]["of"] == 3 * N_IMAGES * 100
    scale = a["fidler_scale"]
    assert scale["equal"] and scale["equals_pillow_optimize"] and scale["version"] == "9.99" and scale["mode"] == "L"
    assert scale["ours_sha256"] == hashlib.sha256(world["ours50"]).hexdigest()
    assert a["jpeg50"]["name"] == "уровень 50 по Fidler et al."
    share = next(s for s in a["jpeg50"]["shares"] if s["level"] == "фрагменты" and s["t_db"] == 42.0)
    assert (share["n"], share["count"], share["share_pct"]) == (2 * N_IMAGES, 0, 0.0)  # 40 dB at quality 50
    rho = {(r["level"], r["key"]): r for r in a["complexity"]["rho"]}
    assert rho[("фрагменты", "ln_cr_vtm")]["n"] == 2 * N_IMAGES - 1  # the fragment with the gap is not judged
    assert a["complexity"]["expectation"]["words"].startswith("ожидание")
    assert record["inputs"]["e4_run_sha256"] == world["e4_sha"] and record["environment"]["imstudies"]
    # privacy: codes only - the units are D.. and D.._r.._c..; no list of integers but the two quantisation tables
    assert {u["unit"][:1] for u in a["units"]} == {"D"}
    assert longest_int_list({k: v for k, v in a.items() if k != "fidler_scale"}) == 0
    with pytest.raises(e5.RunRefusal, match="already exists"):
        run_world(e5, world)


def test_book_cells_checked_and_without_the_irfanview_file(e5, world):
    for f in world["fidler"].iterdir():
        f.unlink()
    run_path, book_path = run_world(e5, world)
    record = json.loads(run_path.read_text(encoding="utf-8"))
    assert record["analysis"]["fidler_scale"] == {"present": False, "equal": False,
                                                  "words": "файла IrfanView нет к запуску E5"}
    assert record["analysis"]["jpeg50"]["name"] == "JPEG PCC 50" and "оговорка" in record["analysis"]["jpeg50"]["note"]
    from openpyxl import load_workbook
    book = load_workbook(book_path)
    assert book.sheetnames == list(e5.SHEETS) and len(book["Сложность"]._charts) == 4
    assert book["Сравнение"]["B1"].value == e5.RULE
    book["Единицы"]["A3"] = "changed"
    book.save(book_path)
    with pytest.raises(e5.RunRefusal, match="book cell Единицы!R3C1"):
        e5.check_book(run_path, book_path)


def test_another_e4_run_file_is_refused(e5, world):
    with pytest.raises(e5.RunRefusal, match="E5 is declared for " + "0" * 64):
        e5.run(world["e4"], world["data"], world["grid"], world["images"], world["fidler"], world["out"],
               {"core_commit": "test", "core_tree": "test"}, e4_pin="0" * 64, registry_pins=world["pins"])
