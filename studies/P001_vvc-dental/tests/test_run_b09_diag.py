"""Lab tests of run_b09_diag.py (study P001, cycle 14) with substitute programs - no pixel of P001 is read.

The substitute MP4Box is a Python script: failing (exit code 1, a line on stdout, a message on stderr) or working
(writes the file named after -new). B05 + B09 of the controls, the encoder of PCC 46 and B09 of the round trip are
substitutes too; the executables are small files whose sha256 the journal header of the test holds.
- MP4Box fails: both attempts keep their exit code and output, the command is that of B09, the verbose attempt
  carries the logging key and its whole output lies beside the stream; outcome (а); a second run is refused;
- MP4Box works: B09 packs once with its round trip; no verbose attempt; outcome (б);
- a control differs from the journal: refusal in words, outcome (в) in the file, PCC 46 never encoded;
- the GPAC profile: names and sizes of its files, never their content.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.lab

STUDY = Path(__file__).resolve().parents[1]
STREAM = b"\x00\x00\x00\x01substitute stream"
PROFILE_TEXT = "content of the profile that must not leave it"
FAKE_MP4BOX = """
import sys
from pathlib import Path
args = sys.argv[1:]
if MODE == "fail":
    print("Importing VVC stream")
    sys.stderr.write("Error: substitute VVC NAL unit parsing failed"
                     + (" [debug log]" if any(a.startswith("-logs=") for a in args) else ""))
    sys.exit(1)
Path(args[args.index("-new") + 1]).write_bytes(b"substitute heif")
"""
ROWS = {**{q: {"status": "valid", "bitstream_sha256": f"b{q}", "file_sha256": f"f{q}", "recon_sha256": f"r{q}",
               "detail": None} for q in (45, 47)},
        46: {"status": "error", "detail": "CalledProcessError: Command [...] returned non-zero exit status 1."}}


@pytest.fixture(scope="module")
def diag():
    spec = importlib.util.spec_from_file_location("run_b09_diag", STUDY / "run_b09_diag.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def make(diag, tmp_path, monkeypatch):
    def build(mode, changed=None):
        exes = {}
        for key in ("encoder", "decoder", "cfg", "mp4box", "gpac"):
            exes[key] = tmp_path / "bin" / key
            exes[key].parent.mkdir(exist_ok=True)
            exes[key].write_bytes(key.encode())
        header = {"codec_versions": {"vtm_gpac_sha256": {k: hashlib.sha256(k.encode()).hexdigest() for k in exes}}}
        script = tmp_path / "mp4box_substitute.py"
        script.write_text(f"MODE = {mode!r}\n{FAKE_MP4BOX}", encoding="utf-8")
        calls = {"encoded": [], "packed": 0}

        def row(paths, image, qp, work):
            return {"status": "valid", "bitstream_sha256": f"b{qp}", "recon_sha256": f"r{qp}",
                    "file_sha256": "another" if qp == changed else f"f{qp}"}

        def encode(paths, image, qp, work):
            calls["encoded"].append(qp)
            return STREAM, image

        def pack_b09(paths, bitstream, recon, work):
            calls["packed"] += 1
            return SimpleNamespace(file=b"substitute heif", provenance={"round_trip": "recon equal"})

        monkeypatch.setattr(diag, "vtm_row", row)
        monkeypatch.setattr(diag, "encode", encode)
        monkeypatch.setattr(diag, "pack_b09", pack_b09)
        profile = tmp_path / "GPAC"
        profile.mkdir()
        (profile / "GPAC.cfg").write_text(PROFILE_TEXT, encoding="utf-8")
        out = tmp_path / "cycles14"

        def call():
            return diag.diagnose(exes, np.zeros((32, 32), np.uint8), ROWS, header, out, [sys.executable, str(script)],
                                 profile, {"core_commit": "test", "core_tree": "test"}, "journal sha256",
                                 {"x": 0, "y": 0, "tile": 32}, log=lambda *a: None)
        return out, call, calls
    return build


def test_a_failing_mp4box_is_kept_with_its_output(diag, make):
    out, call, calls = make("fail")
    record = json.loads(call().read_text(encoding="utf-8"))
    assert [c["equal"] for c in record["controls"]] == [True, True] and calls == {"encoded": [46], "packed": 0}
    assert [a["exit_code"] for a in record["attempts"]] == [1, 1]
    for a in record["attempts"]:
        assert a["stdout_tail"].strip() == "Importing VVC stream"
        assert a["stderr_tail"] == "Error: substitute VVC NAL unit parsing failed" and not a["file_written"]
        cmd = a["command"][2:]  # after the substitute's interpreter and script: the command of B09
        assert (cmd[0], cmd[2], len(cmd)) == ("-add-image", "-new", 4) and cmd[1].endswith("stream.266:primary")
    verbose = record["verbose_attempt"]
    assert verbose["exit_code"] == 1 and verbose["command"][-1] == "-logs=all@debug:ncl"
    assert verbose["stderr_tail"].endswith("[debug log]")
    log = (out / verbose["log_file"]).read_text(encoding="utf-8")
    assert "Importing VVC stream" in log and "[debug log]" in log
    assert (out / record["stream"]["file"]).read_bytes() == STREAM
    assert record["stream"] == {"bytes": len(STREAM), "sha256": hashlib.sha256(STREAM).hexdigest(),
                                "file": "b09_diag/stream_pcc46.266"}
    assert record["outcome"]["letter"] == "а" and record["round_trip"] is None
    assert [f["name"] for f in record["gpac_profile"]["files"]] == ["GPAC.cfg"]
    assert PROFILE_TEXT not in (out / diag.RUN_FILE).read_text(encoding="utf-8")
    with pytest.raises(diag.RunRefusal, match="already exists"):
        call()


def test_a_working_mp4box_goes_to_the_round_trip_of_b09(make):
    out, call, calls = make("work")
    record = json.loads(call().read_text(encoding="utf-8"))
    assert [a["exit_code"] for a in record["attempts"]] == [0, 0] and all(a["file_written"] for a in record["attempts"])
    assert record["verbose_attempt"] is None and calls == {"encoded": [46], "packed": 1}
    assert record["round_trip"] == {"file_bytes": 15, "file_sha256": hashlib.sha256(b"substitute heif").hexdigest(),
                                    "result": "recon equal"}
    assert record["outcome"]["letter"] == "б"


def test_a_control_that_differs_from_the_journal_stops_in_words(diag, make):
    out, call, calls = make("fail", changed=47)
    with pytest.raises(diag.RunRefusal, match=r"outcome \(в\): контроли не равны журналу"):
        call()
    record = json.loads((out / diag.RUN_FILE).read_text(encoding="utf-8"))
    assert [c["equal"] for c in record["controls"]] == [True, False] and record["controls"][1]["file_sha256"] == "another"
    assert record["outcome"]["letter"] == "в" and record["attempts"] == [] and calls == {"encoded": [], "packed": 0}
