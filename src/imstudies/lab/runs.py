"""Shared pieces of study runs: the run-level refusal, the git state of the tree, file hashes, JSON numbers.

Moved from studies/P001_vvc-dental/run_v1.py (cycle 4); run_v1.py keeps its own copies as
the record of cycle 1.
"""

import hashlib
import math
import subprocess
from pathlib import Path


class RunRefusal(Exception):
    """The run cannot go on; the message says why in words."""


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_number(x):
    """A float for JSON: finite as is, infinities and NaN as the strings '+inf', '-inf', 'nan'."""
    if x is None or math.isfinite(x):
        return x
    return "+inf" if x > 0 else ("-inf" if x < 0 else "nan")


def git_state(root):
    """Commit and tree of HEAD in `root`; refuses a tree with tracked changes or untracked files under src/, studies/."""
    def git(*args):
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        if out.returncode != 0:
            raise RunRefusal(f"git {' '.join(args)} failed: {out.stderr.strip()}")
        return out.stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RunRefusal("the working tree has uncommitted changes: the run goes from a committed tree")
    untracked = git("status", "--porcelain", "--untracked-files=all", "--", "src", "studies")
    if untracked:
        raise RunRefusal("untracked files under src/ or studies/ can change the run: "
                         + "; ".join(untracked.splitlines()[:5]))
    return {"core_commit": git("rev-parse", "HEAD"), "core_tree": git("rev-parse", "HEAD^{tree}")}
