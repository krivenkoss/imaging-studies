"""P001 set (B10): are the dental sources copied once under codes D01..D16 with a registry and a private code table? Run: python studies/P001_vvc-dental/run_p001_copy.py

Study P001, cycle 7, stage E3: contract ../REFERENCE.md#b10 through imstudies.lab.p001. Reads the
sources under images_root, writes <data_root>/P001/DNN.png, <data_root>/registry/P001.json and
<private_root>/P001_codes.json (paths from lab_paths.json; private_root lies outside the project).
Prints codes, sizes and counts only - never a source name, a path under the dental folders or a
pixel value - and the leak check: occurrences of source names in the outputs, expected 0.
Refuses existing outputs instead of overwriting them.
"""

import json
import sys
from pathlib import Path

from imstudies.lab import p001
from imstudies.lab.runs import RunRefusal, git_state, sha256_file

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
PROJECT = ROOT.parent
COMMAND = "python studies/P001_vvc-dental/run_p001_copy.py"
QUESTION = ("P001, B10: скопированы ли исходные снимки один раз под кодами D01…D16 с реестром набора и таблицей "
            "«код → файл» вне проекта, без имён в выходах?")
DECLARED = {
    "data": "Исходники B10; роль check у всех снимков; отладочных нет. Значения пикселей не печатаются.",
    "expected": f"{p001.N_UNIQUE} уникальных снимков; иначе стоп с числами, без имён.",
    "leaks": "Проверка утечек — имена и основы имён исходных файлов в PNG, реестре и выводе консоли; ожидается 0.",
}


def load_paths():
    path = ROOT / "lab_paths.json"
    if not path.is_file():
        raise RunRefusal(f"{path} not found: copy lab_paths.example.json to lab_paths.json and set the paths")
    raw = json.loads(path.read_text(encoding="utf-8"))
    paths = {}
    for key in ("images_root", "data_root", "cache_root", "private_root"):
        if not raw.get(key) or raw[key].startswith("<"):
            raise RunRefusal(f"lab_paths.json has no key '{key}' or it is the example placeholder")
        paths[key] = (ROOT / raw[key]).resolve()
    if paths["private_root"] == PROJECT or PROJECT in paths["private_root"].parents:
        raise RunRefusal("private_root lies inside the project folder: the code table must live outside it (B10)")
    return paths


def v1_inputs(cache_root):
    """Codes, shapes and crop corners of the V1 dental images (cycle 1) and the crop sha256 of its run file."""
    folder = Path(cache_root) / "cycles" / "1"
    codes_path, run_path = folder / "v1_dental_codes.json", folder / "v1_metrics.json"
    for path in (codes_path, run_path):
        if not path.is_file():
            raise RunRefusal(f"cycle 1 file not found: {path.name} under cache_root/cycles/1")
    codes = json.loads(codes_path.read_text(encoding="utf-8"))["codes"]
    crops = {i["code"]: i for i in json.loads(run_path.read_text(encoding="utf-8"))["images"] if i["class"] == "dental"}
    entries = [{"code": c["code"], "full_shape": c["full_shape"], "crop_top_left": c["crop_top_left"],
                "file_sha256": c["file_sha256"]} for c in codes]
    return entries, crops, sha256_file(codes_path)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    console = []

    def say(line=""):
        console.append(line)
        print(line)

    try:
        paths = load_paths()
        state = git_state(ROOT)
        say(f"{COMMAND}\n{QUESTION}\nDeclared before the numbers:")
        for key, text in DECLARED.items():
            say(f"  [{key}] {text}")
        entries, crops, codes_sha = v1_inputs(paths["cache_root"])
        registry, leaks = p001.build_copy(paths["images_root"], paths["data_root"], paths["private_root"], entries,
                                          crops, state, log=say)
        say(f"\nSource files: {registry['n_source_files']}; duplicates: {registry['n_duplicates']}; "
            f"unique: {registry['n_unique']} (expected {p001.N_UNIQUE})")
        say("V1 link: " + ", ".join(f"{k} -> {v}" for k, v in sorted(registry["v1_link"].items()))
            + f"; confirmed by file sha256: {sum(registry['v1_link_confirmed_by_file_sha256'].values())} of "
              f"{len(registry['v1_link'])}")
        registry_path = paths["data_root"] / "registry" / "P001.json"
        say(f"Registry: data/registry/P001.json, sha256 {sha256_file(registry_path)}")
        say(f"Copy: data/P001/, {len(registry['images'])} PNG; code table: P001_codes.json in private_root")
        names = [Path(rel).name for rel in p001.source_paths(paths["images_root"])]
        console_leaks = p001.leak_count(["\n".join(console).encode("utf-8")], names)
        say(f"Leak check (source names in PNG, registry, console): {leaks + console_leaks} matches (expected 0)")
        if leaks + console_leaks:
            raise RunRefusal("source names found in the outputs: the copy is not accepted")
    except RunRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
