"""Layout check of study P001: the files every study must carry are present.

Real tests of the study (metrics, codec wrappers, run files) are added with its blocks.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.core

STUDY = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["README.md", "STUDY.md"])
def test_study_file_present(name):
    assert (STUDY / name).is_file()


def test_readme_has_badges():
    text = (STUDY / "README.md").read_text(encoding="utf-8")
    for badge in ("img.shields.io/badge/python", "img.shields.io/badge/license", "actions/workflows/P001.yml/badge.svg"):
        assert badge in text
