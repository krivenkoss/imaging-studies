# P001 — VVC for visually lossless compression of dental radiographs

![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![tests](https://github.com/krivenkoss/imaging-studies/actions/workflows/P001.yml/badge.svg)

Part of [imaging-studies](../../README.md) · Status: **planned** · Paper: in preparation, *Journal of Electrical Engineering*

---

## Question

At which quantization parameter (QP) does the VVC codec (VTM, all-intra) reach PSNR-HVS-M ≈ 42 dB — the threshold of distortion invisibility — on dental panoramic radiographs, and is its compression ratio (CR) at that point larger than that of JPEG, HEIF and AVIF at the same PSNR-HVS-M?

The refutation is declared before the numbers: if, at the threshold, VVC does not give a larger CR than the best of the other codecs for most patients, the paper reports a negative result.

---

## Design

| Item | Setting |
|---|---|
| Images | Dental panoramic radiographs of the authors' institution, 8-bit grayscale; one image per patient |
| Units of analysis | 512 × 512 fragments (threshold and codec comparison) and whole images (stored file size) |
| Independent unit | patient; fragment results are summarised per image first |
| Codecs | VTM (all-intra, 4:0:0, 10-bit internal, QP −12…63) · JPEG (libjpeg-turbo, QF 1–100) · HEIF (libheif + x265, q 0–100) · AVIF (libavif + libaom, 4:0:0, q 0–100) |
| Metrics | PSNR, PSNR-HVS-M, HaarPSI — reference implementations |
| Thresholds | PSNR-HVS-M = 42 dB (main); 40 and 41 dB (sensitivity) |
| CR | 8·W·H / file size in bits, container included; bpp reported alongside |

The full plan, the rules declared before the numbers and the progress log are in [STUDY.md](STUDY.md) (in Russian).

---

## Validation before the study

No number of the study is produced until the implementations pass these checks:

| Check | Against | Criterion |
|---|---|---|
| PSNR | analytic cases | exact |
| PSNR-HVS-M | authors' reference code | ≤ 0.01 dB |
| HaarPSI | authors' reference implementations | ≤ 1e-4 |
| VTM | its own decoder and VVdeC; earlier VTM-23.0 runs on Kodak | bit-exact; same bitstream size and PSNR-Y |
| JPEG | IJG libjpeg with the same quantization tables | bit-exact files |
| HEIF, AVIF | earlier results of the group obtained with libheif | within a tolerance declared before the run |

---

## How to rerun

> The runs are added as the study proceeds; this section lists the exact commands once they exist.

```bash
pytest -m core studies/P001_vvc-dental/tests
```

---

## Results

None yet.

---

## Data availability

The radiographs are not part of the repository. They belong to the authors' institution; the paper shows them without patient data.
