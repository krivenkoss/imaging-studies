"""Lab tests of the VTM wrapper imstudies.lab.coders.vtm (contract ../REFERENCE.md#b05).

Need lab_paths.json with coders.vtm and the build of tools/build_coders.ps1 vtm. Refusals
are checked with executables that do not exist: a refusal raised after the existence check
or after the call would show up as FileNotFoundError instead of the named reason.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from imstudies.core.metrics import Refusal
from imstudies.lab.coders import vtm

pytestmark = pytest.mark.lab

ROOT = Path(__file__).resolve().parents[1]
LEVEL, FRAME_RATE = "5", 1  # B05, approved by the owner 2026-10-14
TIMEOUT_S = 120  # per coder run in these tests; the longest call of cycle 3 took 43 s
NOWHERE = {"encoder": "no/such/EncoderApp.exe", "decoder": "no/such/DecoderApp.exe", "cfg": "no/such/intra.cfg"}


@pytest.fixture(scope="module")
def coder():
    paths_file = ROOT / "lab_paths.json"
    if not paths_file.is_file():
        pytest.skip("lab_paths.json is absent: the laboratory is not configured on this machine")
    paths = {k: ROOT / v for k, v in json.loads(paths_file.read_text(encoding="utf-8"))["coders"]["vtm"].items()}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        pytest.skip(f"VTM is not built (tools/build_coders.ps1 vtm): {missing}")
    return paths


def _frame(h, w):
    """Recipe of frame E0 (seed 1) on (h, w)."""
    y, x = np.mgrid[0:h, 0:w]
    return (x / 2 + y / 4 + np.random.default_rng(1).normal(0, 8, (h, w))).clip(0, 255).astype(np.uint8)


def _call(image, qp=32, *, max_pixels=10**6, level=LEVEL, frame_rate=FRAME_RATE, timeout_s=TIMEOUT_S, paths=NOWHERE,
          work_root):
    return vtm.encode_decode(image, qp, max_pixels=max_pixels, level=level, frame_rate=frame_rate,
                             timeout_s=timeout_s, work_root=work_root, **paths)


def _reason(*args, **kwargs):
    with pytest.raises(Refusal) as info:
        _call(*args, **kwargs)
    return info.value.reason


def test_refusals_before_the_coder_and_their_order(tmp_path):
    good = _frame(16, 16)
    assert _reason(good[None], work_root=tmp_path) == "not_2d"
    assert _reason(good.ravel(), work_root=tmp_path) == "not_2d"
    assert _reason(good.astype(np.float64)[None], qp=None, work_root=tmp_path) == "not_2d"
    assert _reason(good.astype(np.uint16), work_root=tmp_path) == "unsupported_dtype"
    assert _reason(good[:7].astype(np.float64), work_root=tmp_path) == "unsupported_dtype"
    for small in (good[:7], good[:, :7]):
        assert _reason(small, qp=None, work_root=tmp_path) == "image_too_small"
    for name in ("qp", "max_pixels", "level", "frame_rate", "timeout_s"):
        assert _reason(good, **{name: None}, work_root=tmp_path) == "parameter_not_assigned"
    for qp in (-13, 64, 32.0, "32", True):
        assert _reason(good, qp=qp, max_pixels=1, work_root=tmp_path) == "parameter_out_of_range"
    assert _reason(good, max_pixels=good.size - 1, work_root=tmp_path) == "image_too_large"
    huge = np.broadcast_to(np.uint8(7), (100_000, 100_000))  # refused before any file is written
    assert _reason(huge, max_pixels=10**6, work_root=tmp_path) == "image_too_large"
    assert list(tmp_path.iterdir()) == []


def test_missing_executable_is_refused_in_words(tmp_path):
    with pytest.raises(FileNotFoundError, match="EncoderApp not found"):
        _call(_frame(16, 16), work_root=tmp_path)


@pytest.mark.parametrize("qp, shape, max_pixels", [(-12, (8, 8), 64), (63, (8, 8), 64), (np.int64(32), (8, 9), 72)])
def test_good_side_of_the_boundaries(coder, tmp_path, qp, shape, max_pixels):
    image = _frame(*shape)
    result = _call(image, qp=qp, max_pixels=max_pixels, paths=coder, work_root=tmp_path)
    assert result.recon.shape == shape and result.recon.dtype == np.uint8
    assert np.array_equal(result.recon, result.encoder_recon) and len(result.bitstream) > 0
    assert list(tmp_path.iterdir()) == []  # the call directory is removed


@pytest.mark.parametrize("shape", [(21, 13), (13, 21)])
def test_size_not_a_multiple_of_8(coder, tmp_path, shape):
    image = _frame(*shape)
    result = _call(image, paths=coder, work_root=tmp_path)
    assert result.recon.shape == image.shape and result.encoder_recon.shape == image.shape
    assert np.array_equal(result.recon, result.encoder_recon)


def test_provenance_and_determinism(coder, tmp_path):
    image = _frame(24, 40)
    first = _call(image, qp=27, paths=coder, work_root=tmp_path)
    second = _call(image, qp=27, paths=coder, work_root=tmp_path)
    assert first.bitstream == second.bitstream and np.array_equal(first.recon, second.recon)
    prov = first.provenance
    sha = {k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in coder.items()}
    assert (prov["encoder_sha256"], prov["decoder_sha256"], prov["cfg_sha256"]) == (sha["encoder"], sha["decoder"], sha["cfg"])
    assert prov["encoder_version"].startswith("VTM Encoder Version 23.0")
    assert prov["decoder_version"].startswith("VTM Decoder Version 23.0")
    for option in ("--InputBitDepth=8", "--InternalBitDepth=10", "--OutputBitDepth=8", "--InputChromaFormat=400",
                   "--ChromaFormatIDC=400", "--SourceWidth=40", "--SourceHeight=24", "--FramesToBeEncoded=1",
                   "--QP=27", f"--Level={LEVEL}", f"--FrameRate={FRAME_RATE}"):
        assert option in prov["encoder_command"]
    assert prov["decoder_command"][-2:] == ["-d", "8"]
    assert isinstance(first.log_psnr_y, float)


def test_coder_failure_is_an_error_not_a_refusal(coder, tmp_path):
    fake = tmp_path / "failing_encoder.bat"
    fake.write_text("@echo fake encoder failing\r\n@exit /b 3\r\n", encoding="ascii")
    paths = dict(coder, encoder=fake)
    work = tmp_path / "work"
    with pytest.raises(subprocess.CalledProcessError) as info:
        _call(_frame(16, 16), paths=paths, work_root=work)
    assert info.value.returncode == 3 and not isinstance(info.value, Refusal)
    assert "fake encoder failing" in info.value.output


def _fake(path, lines):
    """A batch file standing for a coder executable; %6 is the -b bitstream path of the encoder call."""
    path.write_text("".join(f"@{line}\r\n" for line in lines), encoding="ascii")
    return path


def test_time_limit_is_an_error_not_a_refusal(coder, tmp_path):
    sleeper = _fake(tmp_path / "sleeping_encoder.bat", ["ping -n 4 127.0.0.1 >nul", "exit /b 0"])
    with pytest.raises(TimeoutError, match=r"sleeping_encoder\.bat exceeded timeout_s=1: .* s elapsed; command") as info:
        _call(_frame(16, 16), timeout_s=1, paths=dict(coder, encoder=sleeper), work_root=tmp_path / "work")
    assert not isinstance(info.value, Refusal)
    result = _call(_frame(16, 16), timeout_s=TIMEOUT_S, paths=coder, work_root=tmp_path / "work")  # the other side
    assert np.array_equal(result.recon, result.encoder_recon) and result.provenance["timeout_s"] == TIMEOUT_S


def test_missing_version_line_is_an_error_with_the_output(coder, tmp_path):
    silent = _fake(tmp_path / "unversioned_encoder.bat", ["echo encoder without a banner", "echo data> %6", "exit /b 0"])
    with pytest.raises(RuntimeError, match="printed no 'VTM Encoder Version' line.*encoder without a banner"):
        _call(_frame(16, 16), paths=dict(coder, encoder=silent), work_root=tmp_path / "work")


@pytest.mark.parametrize("stream_line", [None, "type nul > %6"], ids=["no-stream", "empty-stream"])
def test_missing_bitstream_is_reported_before_the_decoder(coder, tmp_path, stream_line):
    marker = tmp_path / "decoder_was_called.txt"
    encoder = _fake(tmp_path / "streamless_encoder.bat",
                    ["echo VTM Encoder Version 23.0 [fake]", *([stream_line] if stream_line else []), "exit /b 0"])
    decoder = _fake(tmp_path / "marking_decoder.bat", [f"echo called> {marker}", "exit /b 1"])
    with pytest.raises(RuntimeError, match="EncoderApp wrote no bitstream or an empty one"):
        _call(_frame(16, 16), paths=dict(coder, encoder=encoder, decoder=decoder), work_root=tmp_path / "work")
    assert not marker.exists()
