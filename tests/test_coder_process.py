"""Lab tests of imstudies.lab.coders.process (run_tool, read_plane), moved from the VTM wrapper in cycle 6.

Fake programs are Python one-liners run by sys.executable, so the tests are the same on Windows
and Linux; both sides of every decision are checked.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from imstudies.core.metrics import Refusal
from imstudies.lab.coders.process import read_plane, run_tool

pytestmark = pytest.mark.lab


def _python(code):
    return [sys.executable, "-c", code]


def test_zero_exit_returns_the_output():
    assert "all good" in run_tool(_python("print('all good')"), 30)


def test_non_zero_exit_is_a_failure_with_code_and_output():
    cmd = _python("import sys; print('something broke'); sys.exit(7)") + ["--arg"]
    with pytest.raises(subprocess.CalledProcessError) as info:
        run_tool(cmd, 30)
    assert info.value.returncode == 7 and "something broke" in info.value.output and info.value.cmd == cmd
    assert not isinstance(info.value, Refusal)


def test_time_limit_is_a_failure_with_program_and_limit():
    cmd = _python("import time; time.sleep(3)")
    program = re.escape(Path(sys.executable).name)
    with pytest.raises(TimeoutError, match=rf"{program} exceeded timeout_s=1: .* s elapsed; command"):
        run_tool(cmd, 1)
    assert run_tool(cmd, 30) == ""  # the other side: the same program within its limit


def test_read_plane_both_sides(tmp_path):
    good = tmp_path / "good.yuv"
    good.write_bytes(bytes(range(6)))
    plane = read_plane(good, 2, 3, "test plane")
    assert plane.shape == (2, 3) and plane.dtype == np.uint8 and plane.tolist() == [[0, 1, 2], [3, 4, 5]]
    short = tmp_path / "short.yuv"
    short.write_bytes(bytes(5))
    with pytest.raises(RuntimeError, match=r"test plane: expected 6 bytes \(8-bit 2x3 plane\), found 5"):
        read_plane(short, 2, 3, "test plane")
    with pytest.raises(RuntimeError, match="found None"):
        read_plane(tmp_path / "missing.yuv", 2, 3, "test plane")
