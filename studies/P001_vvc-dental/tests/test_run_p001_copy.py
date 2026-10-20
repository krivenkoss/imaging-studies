"""Lab tests of run_p001_copy.py (study P001, cycle 7): the paths of the run, both sides.

private_root inside the project folder or left as the example placeholder is refused; a folder
outside the project is taken. The copy itself is tested on synthetic sets in tests/test_p001_set.py.
"""

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.lab

STUDY = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def run_copy():
    spec = importlib.util.spec_from_file_location("run_p001_copy", STUDY / "run_p001_copy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_private_root_must_lie_outside_the_project(run_copy, tmp_path, monkeypatch):
    project, outside = tmp_path / "project", tmp_path / "outside"
    core = project / "core"
    core.mkdir(parents=True)
    monkeypatch.setattr(run_copy, "ROOT", core)
    monkeypatch.setattr(run_copy, "PROJECT", project)
    base = {"images_root": "../Images", "data_root": "../data", "cache_root": "../tmp/lab_cache"}
    paths = core / "lab_paths.json"
    paths.write_text(json.dumps(base | {"private_root": "../P001_private"}), encoding="utf-8")
    with pytest.raises(run_copy.RunRefusal, match="private_root lies inside the project folder"):
        run_copy.load_paths()
    paths.write_text(json.dumps(base | {"private_root": "<folder outside the project>"}), encoding="utf-8")
    with pytest.raises(run_copy.RunRefusal, match="'private_root' or it is the example placeholder"):
        run_copy.load_paths()
    paths.write_text(json.dumps(base | {"private_root": str(outside)}), encoding="utf-8")
    assert run_copy.load_paths()["private_root"] == outside.resolve()
