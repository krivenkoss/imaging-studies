"""Running external coder programs and reading their raw outputs (B05 VTM, B09 GPAC).

Moved from imstudies.lab.coders.vtm (_run, _plane) in cycle 6, behaviour unchanged
(../REFERENCE.md#open-backlog Q-LAB-RUNNER): a non-zero exit code, the time limit and an output
of the wrong size are failures (status `error`), never refusals.
"""

import subprocess
import time
from pathlib import Path

import numpy as np


def run_tool(cmd, timeout_s):
    """Run an executable; a non-zero exit or the time limit is a failure (status `error`), never a refusal."""
    started = time.perf_counter()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"{Path(cmd[0]).name} exceeded timeout_s={timeout_s}: "
                           f"{time.perf_counter() - started:.1f} s elapsed; command: {cmd}") from exc
    if out.returncode != 0:
        raise subprocess.CalledProcessError(out.returncode, cmd, output=out.stdout[-2000:], stderr=out.stderr[-2000:])
    return out.stdout


def read_plane(path, h, w, what):
    """A raw 8-bit (h, w) plane written by a coder; another size or no file is a failure."""
    path = Path(path)
    size = path.stat().st_size if path.is_file() else None
    if size != h * w:
        raise RuntimeError(f"{what}: expected {h * w} bytes (8-bit {h}x{w} plane), found {size}")
    return np.fromfile(path, dtype=np.uint8).reshape(h, w)
