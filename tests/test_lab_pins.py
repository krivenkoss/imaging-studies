"""The versions pinned in the [lab] group of pyproject.toml are the versions installed (cycle 22): the figures and tables
of P001 are the same bytes only under the same matplotlib and openpyxl. The pins are read from the file, not copied."""

import importlib.metadata
import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.lab

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
MUST_BE_PINNED = ("openpyxl", "matplotlib")


def lab_pins():
    lab = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"]["lab"]
    found = (re.fullmatch(r"([A-Za-z0-9_.\-]+)==(\S+)", requirement) for requirement in lab)
    return {m.group(1): m.group(2) for m in found if m}


def test_the_lab_versions_pinned_in_pyproject_are_installed():
    pinned = lab_pins()
    assert set(MUST_BE_PINNED) <= set(pinned), pinned
    installed = {name: importlib.metadata.version(name) for name in pinned}
    assert installed == pinned
