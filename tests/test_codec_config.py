"""Lab test: versions and parameter ranges of the codec wrappers against ../config.json `coders`.

Every string of EXPECTED_VERSIONS of jpeg, heif, avif must occur in coders.<codec>.codec, QUALITY_RANGE
must equal coders.<codec>.range, and the QP range of the VTM wrapper must equal coders.vtm.range.
config.json is one level above core/; a missing file or key fails the test by name, no default
(../CLAUDE.md R09). Wrappers are imported inside the tests: without the [lab] group, -m core
deselects these tests instead of failing at collection.
"""

import importlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.lab

CONFIG = Path(__file__).resolve().parents[2] / "config.json"


def _coder(codec):
    if not CONFIG.is_file():
        pytest.fail(f"config.json not found: {CONFIG}")
    coders = json.loads(CONFIG.read_text(encoding="utf-8")).get("coders")
    if coders is None:
        pytest.fail(f"{CONFIG}: no key 'coders'")
    if codec not in coders:
        pytest.fail(f"{CONFIG}: no key 'coders.{codec}'")
    for key in ("codec", "range"):
        if key not in coders[codec]:
            pytest.fail(f"{CONFIG}: no key 'coders.{codec}.{key}'")
    return coders[codec]


@pytest.mark.parametrize("codec", ["jpeg", "heif", "avif"])
def test_expected_versions_occur_in_config(codec):
    module, entry = importlib.import_module(f"imstudies.lab.coders.{codec}"), _coder(codec)
    missing = {name: version for name, version in module.EXPECTED_VERSIONS.items() if version not in entry["codec"]}
    assert not missing, f"coders.{codec}.codec = {entry['codec']!r} lacks {missing}"


@pytest.mark.parametrize("codec", ["jpeg", "heif", "avif"])
def test_quality_range_equals_config(codec):
    module, entry = importlib.import_module(f"imstudies.lab.coders.{codec}"), _coder(codec)
    assert list(module.QUALITY_RANGE) == entry["range"], f"coders.{codec}.range = {entry['range']}"


def test_vtm_qp_range_equals_config():
    vtm = importlib.import_module("imstudies.lab.coders.vtm")
    assert [vtm.QP_MIN, vtm.QP_MAX] == _coder("vtm")["range"]
