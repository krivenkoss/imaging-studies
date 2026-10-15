"""Lab tests of the shared run pieces imstudies.lab.runs and imstudies.lab.images (moved from run_v1.py, cycle 4).

- git_state: a clean tree gives commit and tree; tracked changes and untracked files under
  src/ or studies/ refuse; an untracked file elsewhere does not (git needed);
- load_test_images: the same pixels as run_v1.test_set and as the run file of cycle 1
  (local test images and cycles/1/v1_metrics.json needed).
"""

import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from imstudies.lab.images import load_test_images
from imstudies.lab.runs import RunRefusal, git_state, json_number, sha256_file

pytestmark = pytest.mark.lab

ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES = ["BABOON.BMP", "BARBARA.BMP", "goldhill.bmp", "lenna.bmp", "frisco.bmp", "DIEGO.BMP"]


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.org", "-c", "core.autocrlf=false",
                    *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "studies").mkdir()
    (tmp_path / "src" / "module.py").write_text("x = 1\n", encoding="ascii")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "first")
    return tmp_path


def test_git_state_clean_and_untracked_elsewhere(repo):
    state = git_state(repo)
    assert len(state["core_commit"]) == 40 and len(state["core_tree"]) == 40
    (repo / "notes.txt").write_text("not code\n", encoding="ascii")
    assert git_state(repo) == state


@pytest.mark.parametrize("where", ["src/extra.py", "studies/P999/run_x.py"])
def test_git_state_refuses_untracked_under_src_and_studies(repo, where):
    path = repo / where
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("y = 2\n", encoding="ascii")
    with pytest.raises(RunRefusal, match="untracked files under src/ or studies/"):
        git_state(repo)


def test_git_state_refuses_tracked_changes(repo):
    (repo / "src" / "module.py").write_text("x = 2\n", encoding="ascii")
    with pytest.raises(RunRefusal, match="uncommitted changes"):
        git_state(repo)


def test_json_number_and_sha256(tmp_path):
    assert [json_number(v) for v in (1.5, None, math.inf, -math.inf, math.nan)] == [1.5, None, "+inf", "-inf", "nan"]
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc")
    assert sha256_file(f) == hashlib.sha256(b"abc").hexdigest()


def test_test_images_parity_with_run_v1_and_cycle_1():
    paths_file = ROOT / "lab_paths.json"
    if not paths_file.is_file():
        pytest.skip("lab_paths.json is absent: the laboratory is not configured on this machine")
    paths = json.loads(paths_file.read_text(encoding="utf-8"))
    images_root, cache_root = (ROOT / paths["images_root"]).resolve(), (ROOT / paths["cache_root"]).resolve()
    new = load_test_images(images_root, TEST_IMAGES, (512, 512))

    spec = importlib.util.spec_from_file_location("run_v1", ROOT / "studies" / "P001_vvc-dental" / "run_v1.py")
    run_v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_v1)
    old = run_v1.test_set(images_root)
    assert [i["code"] for i in new] == [i["code"] for i in old]
    for a, b in zip(new, old, strict=True):
        assert a["array"].dtype == b["array"].dtype == np.uint8 and np.array_equal(a["array"], b["array"])
        assert {k: v for k, v in a.items() if k != "array"} == {k: v for k, v in b.items() if k != "array"}

    cycle1 = json.loads((cache_root / "cycles" / "1" / "v1_metrics.json").read_text(encoding="utf-8"))
    recorded = {i["code"]: i["pixels_sha256"] for i in cycle1["images"] if i["class"] == "test"}
    assert {i["code"]: hashlib.sha256(i["array"].tobytes()).hexdigest() for i in new} == recorded
