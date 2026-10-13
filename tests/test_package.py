"""Smoke test of the package skeleton: it imports and its version matches pyproject.toml."""

import tomllib
from pathlib import Path

import pytest

import imstudies

pytestmark = pytest.mark.core

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert imstudies.__version__ == project["version"]


def test_layers_import():
    import imstudies.core
    import imstudies.lab
    import imstudies.oracle

    assert imstudies.core.__doc__ and imstudies.lab.__doc__ and imstudies.oracle.__doc__
