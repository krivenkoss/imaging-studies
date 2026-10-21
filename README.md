# imaging-studies

Reproducible studies in lossy image compression and image denoising: one study, one paper.

![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![tests](https://github.com/krivenkoss/imaging-studies/actions/workflows/tests.yml/badge.svg)

**S. S. Kryvenko** · [Department of Information and Communication Technologies](https://dict.khai.edu/), National Aerospace University «Kharkiv Aviation Institute»

---

## What it is

`imaging-studies` holds the code behind our papers on lossy image compression and image denoising. Every study lives in its own folder under `studies/`, answers one question, and backs one paper: the numbers, tables and figures of the paper are rebuilt from the study's run files by a single command.

The shared package `imstudies` provides what the studies reuse: image quality metrics, noise models, compression-ratio bookkeeping and wrappers around codec executables. A study uses the package but does not change it; a change to the package goes through its own tests, which every study's tests then run against.

---

## Studies

| Study | Question | Status | Paper | Tests |
|---|---|---|---|---|
| [P001 — VVC for dental radiographs](studies/P001_vvc-dental/README.md) | At which QP does VVC (VTM, all-intra) reach the distortion-invisibility threshold of PSNR-HVS-M on dental panoramic radiographs, and is its compression ratio then larger than that of JPEG, HEIF and AVIF? | planned | in preparation, *Journal of Electrical Engineering* | ![P001](https://github.com/krivenkoss/imaging-studies/actions/workflows/P001.yml/badge.svg) |

---

## Project structure

```
imaging-studies/
├── README.md
├── LICENSE                     MIT
├── CITATION.cff                Citation metadata
├── pyproject.toml              Package, dependencies, pytest settings
├── .github/workflows/
│   ├── tests.yml               Tests of the shared package
│   └── P001.yml                Tests of study P001 (one workflow per study)
├── src/imstudies/
│   ├── core/                   Pure functions: metrics, noise models, CR
│   ├── lab/                    File layer: image readers, codec wrappers
│   └── oracle/                 Independent reference implementations for tests
├── tests/                      Tests of the shared package
├── third_party/                Address of each codec's tag archive and its SHA-256 (archives outside Git)
├── tools/                      Build scripts for codec executables
└── studies/
    └── P001_vvc-dental/
        ├── README.md           What the study does and how to rerun it
        ├── STUDY.md            Plan and progress log (in Russian)
        ├── run_*.py            Runs; each writes its own run file
        ├── figures.py          Paper figures and tables from run files
        ├── figures_style.toml  Fonts, sizes, colours, legends of the figures
        └── tests/
```

---

## Installation

Python 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS
pip install -e ".[lab,dev]"
```

The core tests need only `pip install -e ".[dev]"`.

---

## Codec executables

Codecs are built from their sources pinned to release tags, so every number in a paper can be traced to an exact codec version. Executables are not stored in the repository; `tools/` holds the build scripts and `third_party/` the address of each tag archive and its SHA-256 (the archives themselves are kept outside Git). The version, tag and SHA-256 of each executable are written into the provenance of every result.

| Codec | Implementation |
|---|---|
| VVC | VTM (VVC Test Model), reference implementation |
| JPEG | libjpeg-turbo |
| HEIF | libheif with x265 |
| AVIF | libavif with libaom |

---

## Data

Test images are not part of the repository. Medical images belong to the authors' institution and are never committed; papers show them without any patient data. Paths to local data are set in `lab_paths.json` (copy `lab_paths.example.json`).

---

## Tests

```bash
pytest -m core
```

`core` tests run without data, codecs or laboratory packages; they are what the badges report.

---

## Citation

```bibtex
@software{kryvenko2026imagingstudies,
  author      = {Kryvenko, S. S.},
  title       = {imaging-studies: reproducible studies in lossy image compression and denoising},
  year        = {2026},
  version     = {0.7.1},
  license     = {MIT},
  url         = {https://github.com/krivenkoss/imaging-studies},
  institution = {National Aerospace University, Department of Information and Communication Technologies}
}
```

See `CITATION.cff` for full metadata.

---

## License

MIT — see [LICENSE](LICENSE). Codec sources (archives listed in `third_party/`, kept outside Git) keep their own licenses.
