"""Lab tests of imstudies.lab.coders.avif_libheif (contract ../REFERENCE.md#b12): the separate environment of
lab_paths.json key avif_libheif_python is needed for the coding tests (skipped without it).

Both sides of every decision: synthetic sizes 8 x 8, 21 x 13, 517 x 389, 512 x 512 are coded monochrome with
the shape kept; a repeated call gives the same sha256; a batch call equals single calls bit for bit; the six
named refusals come before any call; no key or no file of the environment is refused in words; an executor
reporting other versions is refused in words; an executor returning a colour AVIF (made by Pillow's own AVIF)
is a failure. The substitute executors run under this test's python.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from imstudies.core.metrics import Refusal
from imstudies.lab.synthetic import e0_crop, e0_frame

pytestmark = pytest.mark.lab

CORE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def b12():
    import importlib

    return importlib.import_module("imstudies.lab.coders.avif_libheif")


@pytest.fixture(scope="module")
def env():
    path = CORE / "lab_paths.json"
    if not path.is_file():
        pytest.skip("lab_paths.json is absent: the laboratory is not configured on this machine")
    raw = json.loads(path.read_text(encoding="utf-8"))
    python = raw.get("avif_libheif_python")
    if not python or python.startswith("<") or not Path(python).is_file():
        pytest.skip("the B12 environment (avif_libheif_python) is not configured on this machine")
    return {"python": python, "work_root": (CORE / raw["cache_root"]).resolve() / "work_tests_b12"}


def call(b12, env, image, quality, **kw):
    return b12.encode_decode(image, quality, max_pixels=image.size, timeout_s=120, python=env["python"],
                             work_root=env["work_root"], **kw)


@pytest.mark.parametrize("shape", [(8, 8), (21, 13), (517, 389), (512, 512)])
def test_monochrome_and_shape_kept(b12, env, shape):
    image = e0_frame() if shape == (512, 512) else e0_crop(*shape)
    coded = call(b12, env, image, 50)
    assert coded.recon.shape == image.shape and coded.recon.dtype == np.uint8
    assert coded.provenance["av1C_monochrome"] == 1 and coded.provenance["enc_params"] == b12.ENC_PARAMS
    assert coded.provenance["versions"] == b12.EXPECTED_VERSIONS
    assert list(Path(env["work_root"]).iterdir()) == []  # the temporary folder is removed after the call


def test_repeat_and_batch_equal_single_calls(b12, env):
    image = e0_crop(517, 389)
    first, second = call(b12, env, image, 37), call(b12, env, image, 37)
    assert hashlib.sha256(first.file).hexdigest() == hashlib.sha256(second.file).hexdigest()
    batch = call(b12, env, image, [0, 37, 100])
    singles = [call(b12, env, image, q) for q in (0, 37, 100)]
    assert [c.file for c in batch] == [c.file for c in singles] and batch[1].file == first.file
    assert all(np.array_equal(b.recon, s.recon) for b, s in zip(batch, singles))
    assert [c.provenance["encode"]["quality"] for c in batch] == [0, 37, 100]


@pytest.mark.parametrize("image, quality, kw, reason", [
    (np.zeros((3, 8, 8), np.uint8), 50, {}, "not_2d"),
    (np.zeros((8, 8), np.uint16), 50, {}, "unsupported_dtype"),
    (np.zeros((7, 8), np.uint8), 50, {}, "image_too_small"),
    (np.zeros((8, 8), np.uint8), None, {}, "parameter_not_assigned"),
    (np.zeros((8, 8), np.uint8), 50, {"timeout_s": None}, "parameter_not_assigned"),
    (np.zeros((8, 8), np.uint8), 101, {}, "parameter_out_of_range"),
    (np.zeros((8, 8), np.uint8), True, {}, "parameter_out_of_range"),
    (np.zeros((8, 8), np.uint8), [], {}, "parameter_out_of_range"),
    (np.zeros((8, 8), np.uint8), [50, -1], {}, "parameter_out_of_range"),
    (np.zeros((8, 8), np.uint8), 50, {"timeout_s": 0}, "parameter_out_of_range"),
    (np.zeros((8, 8), np.uint8), 50, {"max_pixels": 63}, "image_too_large"),
])
def test_named_refusals_before_any_call(b12, image, quality, kw, reason):
    args = {"max_pixels": 64, "timeout_s": 60, "python": None, "work_root": "unused"} | kw
    with pytest.raises(Refusal) as info:
        b12.encode_decode(image, quality, **args)
    assert info.value.reason == reason


def test_environment_missing_is_refused_in_words(b12, tmp_path):
    image = e0_crop(8, 8)
    with pytest.raises(FileNotFoundError, match="avif_libheif_python is not set"):
        b12.encode_decode(image, 50, max_pixels=64, timeout_s=60, python=None, work_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="python of the B12 environment is not found"):
        b12.encode_decode(image, 50, max_pixels=64, timeout_s=60, python=tmp_path / "python.exe", work_root=tmp_path)


FAKE = '''
import io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
work = Path(sys.argv[1])
request = json.loads((work / "request.json").read_text(encoding="utf-8"))
image = np.load(work / "image.npy")
versions = json.loads({versions!r})
items = []
for q in request["qualities"]:
    buf = io.BytesIO()
    Image.fromarray(image, "L").convert("RGB").save(buf, "AVIF", quality=q)  # Pillow's own AVIF, colour
    (work / f"q{{q:03d}}.avif").write_bytes(buf.getvalue())
    np.save(work / f"q{{q:03d}}_recon.npy", image)
    items.append({{"quality": q, "mode": "L", "shape": list(image.shape), "dtype": "uint8"}})
(work / "result.json").write_text(json.dumps({{"versions": versions, "libheif_info": {{}}, "options": {{}},
                                              "items": items}}), encoding="utf-8")
'''


def fake(b12, monkeypatch, tmp_path, versions):
    script = tmp_path / "fake_exec.py"
    script.write_text(FAKE.format(versions=json.dumps(versions)), encoding="utf-8")
    monkeypatch.setattr(b12, "EXEC", script)


def test_other_versions_refused_in_words(b12, monkeypatch, tmp_path):
    fake(b12, monkeypatch, tmp_path, b12.EXPECTED_VERSIONS | {"aom_encoder": "AOMedia Project AV1 Encoder v3.6.1"})
    with pytest.raises(RuntimeError, match=r"AVIF \(libheif\): library versions differ from the declared ones"):
        b12.encode_decode(e0_crop(21, 13), 50, max_pixels=273, timeout_s=60, python=sys.executable, work_root=tmp_path)


def test_colour_file_is_a_failure(b12, monkeypatch, tmp_path):
    fake(b12, monkeypatch, tmp_path, dict(b12.EXPECTED_VERSIONS))
    with pytest.raises(RuntimeError, match=r"q 50: av1C monochrome = 0, not 1"):
        b12.encode_decode(e0_crop(21, 13), 50, max_pixels=273, timeout_s=60, python=sys.executable, work_root=tmp_path)
