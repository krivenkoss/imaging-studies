"""Lab tests of blocks B03-B04 against the authors' files (not in this repository, not in CI).

The authors' sources are read from `metric_sources` of lab_paths.json; they are never
copied here. Checks: the files are the ones named in studies/P001_vvc-dental/STUDY.md
section 6 (sha256); the CSF and MaskCof tables equal the ones in psnrhvsm.m; the stored
haarPsi.py values of tests/vectors/metrics_reference.json are reproduced by haarPsi.py.
"""

import base64
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from imstudies.core.metrics import CSF, MASK_COF

pytestmark = pytest.mark.lab

ROOT = Path(__file__).resolve().parents[1]
VECTORS = json.loads((ROOT / "tests" / "vectors" / "metrics_reference.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sources():
    paths = ROOT / "lab_paths.json"
    if not paths.is_file():
        pytest.skip("lab_paths.json is absent: the laboratory is not configured on this machine")
    src = (ROOT / json.loads(paths.read_text(encoding="utf-8"))["metric_sources"]).resolve()
    assert src.is_dir(), f"metric_sources from lab_paths.json does not exist: {src}"
    return src


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sources_are_the_declared_files(sources):
    prov = VECTORS["provenance"]
    assert _sha256(sources / "psnrhvsm" / "psnrhvsm.m") == prov["psnrhvsm_m_sha256"]
    assert _sha256(sources / "haarpsi" / "HaarPSI.m") == prov["haarpsi_m_sha256"]
    assert _sha256(sources / "haarpsi" / "haarPsi.py") == prov["haarpsi_py_sha256"]


def _matrices(text, name):
    """Every `name = [ ... ];` matrix literal of a MATLAB file, as float arrays."""
    found = []
    for body in re.findall(rf"\b{name}\s*=\s*\[(.*?)\];", text, flags=re.S):
        rows = [r for r in (line.split("%")[0].strip(" ;\t") for line in body.splitlines()) if r]
        found.append(np.array([[float(v) for v in r.split(",")] for r in rows]))
    return found


def test_tables_equal_psnrhvsm_m(sources):
    text = (sources / "psnrhvsm" / "psnrhvsm.m").read_text(encoding="latin-1")
    csf = _matrices(text, "CSFCof")
    masks = _matrices(text, "MaskCof")
    assert len(csf) == 1 and len(masks) == 2  # MaskCof is written twice: main body and maskeff
    assert csf[0].shape == (8, 8) and np.array_equal(csf[0], CSF)
    for mask in masks:
        assert mask.shape == (8, 8) and np.array_equal(mask, MASK_COF)
    # spot values typed from the file: a parse and a table transposed alike would still be equal
    assert (CSF[0, 0], CSF[0, 2], CSF[7, 7]) == (1.608443, 2.573509, 0.259950)
    assert (MASK_COF[0, 2], MASK_COF[4, 5], MASK_COF[7, 7]) == (1.0, 0.008417, 0.010203)


def test_vectors_reproduced_by_haarpsi_py(sources):
    spec = importlib.util.spec_from_file_location("haarpsi_reference", sources / "haarpsi" / "haarPsi.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for case in VECTORS["cases"]:
        ref = np.frombuffer(base64.b64decode(case["ref"]), np.uint8).reshape(case["shape"])
        dist = np.frombuffer(base64.b64decode(case["dist"]), np.uint8).reshape(case["shape"])
        assert float(module.haar_psi(ref, dist)[0]) == case["haarpsi_py"], case["id"]
