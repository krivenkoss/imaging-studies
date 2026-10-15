"""Core tests of tools/api_registry.py, the API-REGISTRY generator (OPEN-BACKLOG Q-REGISTRY-GEN)."""

import ast
import datetime
import importlib.util
from pathlib import Path

import pytest

from imstudies.core.metrics import REFUSAL_REASONS

pytestmark = pytest.mark.core

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "api_registry.py"
_spec = importlib.util.spec_from_file_location("api_registry", TOOL)
api_registry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(api_registry)

SYNTHETIC = '''"""A synthetic core module."""
from typing import NamedTuple

REFUSAL_REASONS = ("first_reason", "second_reason")
_HIDDEN_CONSTANT = 1


class Refusal(Exception):
    def __init__(self, reason, detail=""):
        self.reason = reason


class Output(NamedTuple):
    value: float
    note: str | None = None


class _PrivateClass:
    pass


def public_fn(x, y=2, *, z) -> float:
    return _hidden_fn(x) + y + z


def _hidden_fn(q):
    return q
'''


def _synthetic_root(tmp_path):
    core = tmp_path / "src" / "imstudies" / "core"
    core.mkdir(parents=True)
    (core / "__init__.py").write_text('"""Package docstring only."""\n', encoding="utf-8")
    (core / "synthetic.py").write_text(SYNTHETIC, encoding="utf-8")
    return tmp_path


def test_synthetic_module_public_names_only(tmp_path):
    now = datetime.datetime(2026, 10, 14, 15, 0, tzinfo=datetime.timezone.utc)
    out = api_registry.registry(_synthetic_root(tmp_path), commit="abc123", now=now)
    assert out.startswith("Snapshot 2026-10-14 15:00 UTC, core commit `abc123`")
    assert "### `imstudies.core.synthetic`" in out
    assert "### `imstudies.core`" not in out  # nothing public in __init__.py
    assert "- `public_fn(x, y=2, *, z) -> float`" in out
    assert "- class `Refusal(Exception)`: `Refusal(reason, detail='')`" in out
    assert "- class `Output(NamedTuple)`: fields `value: float`, `note: str | None = None`" in out
    assert "- `REFUSAL_REASONS`: `first_reason`, `second_reason`" in out
    for hidden in ("_hidden_fn", "_PrivateClass", "_HIDDEN_CONSTANT"):
        assert hidden not in out


@pytest.mark.parametrize("layout", ["loose", "packed", "detached"])
def test_commit_read_from_git_files(tmp_path, layout):
    git = _synthetic_root(tmp_path) / ".git"
    git.mkdir()
    (git / "HEAD").write_text("0123abcd\n" if layout == "detached" else "ref: refs/heads/main\n", encoding="ascii")
    if layout == "loose":
        (git / "refs" / "heads").mkdir(parents=True)
        (git / "refs" / "heads" / "main").write_text("0123abcd\n", encoding="ascii")
    elif layout == "packed":
        (git / "packed-refs").write_text("# pack-refs with: peeled\n0123abcd refs/heads/main\n", encoding="ascii")
    assert "core commit `0123abcd`" in api_registry.registry(tmp_path)


def test_missing_ref_refuses_in_words(tmp_path):
    git = _synthetic_root(tmp_path) / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")
    with pytest.raises(SystemExit, match="pass --commit"):
        api_registry.registry(tmp_path)


def test_real_core():
    out = api_registry.registry(ROOT, commit="test")
    assert "### `imstudies.core.metrics`" in out
    for name in ("psnr", "psnr_hvs_m", "haarpsi"):
        assert f"- `{name}(ref, dist, *, max_pixels)`" in out
    line = next(s for s in out.splitlines() if s.startswith("- `REFUSAL_REASONS`: "))
    printed = tuple(r.strip(" `") for r in line.removeprefix("- `REFUSAL_REASONS`: ").split(","))
    assert printed == REFUSAL_REASONS


def test_generator_does_not_import_the_package():
    imported = set()
    for node in ast.walk(ast.parse(TOOL.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert imported == {"argparse", "ast", "copy", "datetime", "pathlib"}


def test_lab_modules_are_listed_with_nested_packages(tmp_path):
    root = _synthetic_root(tmp_path)
    nested = root / "src" / "imstudies" / "lab" / "coders"
    nested.mkdir(parents=True)
    (nested / "__init__.py").write_text('"""Only a docstring."""\n', encoding="utf-8")
    (nested / "codec.py").write_text("def encode(image, *, quality):\n    return image\n\n\n"
                                     "def _private(x):\n    return x\n", encoding="utf-8")
    out = api_registry.registry(root, commit="abc123")
    assert "### `imstudies.lab.coders.codec`" in out and "- `encode(image, *, quality)`" in out
    assert "_private" not in out and "### `imstudies.lab.coders`" not in out
    assert out.index("### `imstudies.core.synthetic`") < out.index("### `imstudies.lab.coders.codec`")


def test_real_lab():
    out = api_registry.registry(ROOT, commit="test")
    assert ("- `encode_decode(image, qp, *, max_pixels, level, frame_rate, timeout_s, encoder, decoder, cfg, "
            "work_root)`") in out
    assert "- `pack(bitstream, recon, *, timeout_s, mp4box, gpac, decoder, work_root)`" in out
    for module in ("jpeg", "heif", "avif"):
        assert f"### `imstudies.lab.coders.{module}`" in out
    assert "### `imstudies.lab.runs`" in out and "- `git_state(root)`" in out
