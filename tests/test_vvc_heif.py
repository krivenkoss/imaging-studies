"""Lab tests of the VVC-in-HEIF packing (B09) imstudies.lab.coders.vvc_heif and of the E0 recipe module.

Need lab_paths.json with coders.vtm and coders.gpac (VTM build and GPAC installed).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from imstudies.core.metrics import Refusal
from imstudies.lab import synthetic
from imstudies.lab.coders import isobmff, vtm, vvc_heif

pytestmark = pytest.mark.lab

ROOT = Path(__file__).resolve().parents[1]
NOWHERE = {"mp4box": "no/such/mp4box.exe", "gpac": "no/such/gpac.exe", "decoder": "no/such/DecoderApp.exe"}


@pytest.fixture(scope="module")
def lab():
    paths_file = ROOT / "lab_paths.json"
    if not paths_file.is_file():
        pytest.skip("lab_paths.json is absent: the laboratory is not configured on this machine")
    raw = json.loads(paths_file.read_text(encoding="utf-8"))
    coders = {k: {n: ROOT / p for n, p in raw["coders"][k].items()} for k in ("vtm", "gpac")}
    missing = [str(p) for c in coders.values() for p in c.values() if not p.is_file()]
    if missing:
        pytest.skip(f"VTM or GPAC is missing: {missing}")
    return {"vtm": coders["vtm"], "gpac": coders["gpac"], "cache_root": (ROOT / raw["cache_root"]).resolve()}


@pytest.fixture(scope="module")
def coded(lab, tmp_path_factory):
    image = synthetic.e0_crop(21, 13)
    return vtm.encode_decode(image, 32, max_pixels=image.size, level="5", frame_rate=1, timeout_s=600,
                             work_root=tmp_path_factory.mktemp("vtm"), **lab["vtm"])


def _paths(lab):
    return {"mp4box": lab["gpac"]["mp4box"], "gpac": lab["gpac"]["gpac"], "decoder": lab["vtm"]["decoder"]}


def test_timeout_refusals_before_any_file(tmp_path):
    with pytest.raises(Refusal) as info:
        vvc_heif.pack(b"x", np.zeros((8, 8), np.uint8), timeout_s=None, work_root=tmp_path, **NOWHERE)
    assert info.value.reason == "parameter_not_assigned"
    for bad in (0, -1, True, "5", float("inf"), float("nan")):
        with pytest.raises(Refusal) as info:
            vvc_heif.pack(b"x", np.zeros((8, 8), np.uint8), timeout_s=bad, work_root=tmp_path, **NOWHERE)
        assert info.value.reason == "parameter_out_of_range"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bitstream", [b"", bytearray(b"x"), "x"])
def test_broken_bitstream_is_an_error(tmp_path, bitstream):
    with pytest.raises(RuntimeError, match="bitstream is empty or not bytes"):
        vvc_heif.pack(bitstream, np.zeros((8, 8), np.uint8), timeout_s=60, work_root=tmp_path, **NOWHERE)


def test_missing_executable_is_refused_in_words(tmp_path):
    with pytest.raises(FileNotFoundError, match="MP4Box not found"):
        vvc_heif.pack(b"x", np.zeros((8, 8), np.uint8), timeout_s=60, work_root=tmp_path, **NOWHERE)


def test_pack_round_trip_and_provenance(lab, coded, tmp_path):
    first = vvc_heif.pack(coded.bitstream, coded.recon, timeout_s=60, work_root=tmp_path, **_paths(lab))
    second = vvc_heif.pack(coded.bitstream, coded.recon, timeout_s=60, work_root=tmp_path, **_paths(lab))
    assert first.file == second.file  # nothing time-dependent in the file
    assert isobmff.ispe(first.file) == (13, 21)
    p = first.provenance
    assert p["round_trip"] == "recon equal" and p["file_bytes"] == len(first.file)
    assert p["gpac_version"] == "26.08-DEV-rev171-gf9f54240-master"
    assert p["mp4box_sha256"] == hashlib.sha256(Path(lab["gpac"]["mp4box"]).read_bytes()).hexdigest()
    assert p["pack_command"][1:3] == ["-add-image", p["pack_command"][2]] and p["pack_command"][2].endswith(":primary")
    assert p["mdat_bytes"] > 0 and sum(p["box_bytes"].values()) == len(first.file)
    assert list(tmp_path.iterdir()) == []


def test_wrong_recon_fails_the_round_trip(lab, coded, tmp_path):
    other = coded.recon.copy()
    other[0, 0] ^= 1
    with pytest.raises(RuntimeError, match="round trip failed") as info:
        vvc_heif.pack(coded.bitstream, other, timeout_s=60, work_root=tmp_path, **_paths(lab))
    assert not isinstance(info.value, Refusal)


def test_ispe_must_match_the_recon(lab, coded, tmp_path):
    with pytest.raises(RuntimeError, match=r"ispe of the HEIF file is \(13, 21\), expected"):
        vvc_heif.pack(coded.bitstream, coded.recon.T.copy(), timeout_s=60, work_root=tmp_path, **_paths(lab))


def test_corrupted_file_is_an_error(lab, coded, tmp_path, monkeypatch):
    original = vvc_heif._pack

    def corrupting(stream, heic, mp4box, timeout_s):
        cmd = original(stream, heic, mp4box, timeout_s)
        data = bytearray(heic.read_bytes())
        data[-1 - (isobmff.top_level_sizes(bytes(data)).get("free", 0))] ^= 0xFF  # last byte of mdat
        heic.write_bytes(bytes(data))
        return cmd

    monkeypatch.setattr(vvc_heif, "_pack", corrupting)
    with pytest.raises(Exception) as info:
        vvc_heif.pack(coded.bitstream, coded.recon, timeout_s=60, work_root=tmp_path, **_paths(lab))
    assert not isinstance(info.value, Refusal)


def test_e0_recipe_parity_with_cycle_3(lab):
    run3 = json.loads((lab["cache_root"] / "cycles" / "3" / "v2a_vtm.json").read_text(encoding="utf-8"))
    recorded = {i["code"]: i["pixels_sha256"] for i in run3["images"] if i["class"] in ("control", "synthetic")}
    ours = {"e0": synthetic.e0_frame()} | {f"syn{h}x{w}": synthetic.e0_crop(h, w) for h, w in ((8, 8), (21, 13), (517, 389))}
    assert {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in ours.items()} == recorded
    with pytest.raises(ValueError):
        synthetic.e0_crop(518, 8)


@pytest.mark.parametrize("code", ["e0", "syn517x389"])
def test_regression_against_cycles_3_and_5(lab, tmp_path, code):
    """B05 + B09 after moving run_tool/read_plane (cycle 6): every byte as in the run files of cycles 3 and 5."""
    image = synthetic.e0_frame() if code == "e0" else synthetic.e0_crop(517, 389)
    cycles = lab["cache_root"] / "cycles"
    then3 = next(r for r in json.loads((cycles / "3" / "v2a_vtm.json").read_text(encoding="utf-8"))["rows"]
                 if (r["image_code"], r["qp"]) == (code, 32))
    then5 = next(r for r in json.loads((cycles / "5" / "v2e_heif.json").read_text(encoding="utf-8"))["rows"]
                 if (r["image_code"], r["qp"]) == (code, 32))
    coded = vtm.encode_decode(image, 32, max_pixels=image.size, level="5", frame_rate=1, timeout_s=600,
                              work_root=tmp_path, **lab["vtm"])
    packed = vvc_heif.pack(coded.bitstream, coded.recon, timeout_s=600, work_root=tmp_path, **_paths(lab))
    assert hashlib.sha256(coded.bitstream).hexdigest() == then3["bitstream_sha256"]
    assert hashlib.sha256(coded.recon.tobytes()).hexdigest() == then3["recon_sha256"]
    assert hashlib.sha256(packed.file).hexdigest() == then5["heif_sha256"]
