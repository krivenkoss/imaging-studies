# Rules for code work in this repository

Project documents live one level above this repository and are **read-only from
here**. They are written in Russian; code, comments, docstrings and commit messages
are English (`../CLAUDE.md` R09); `prompts/REPORT.md` is written in
{{REPORT LANGUAGE — the owner's choice, stated once here; ../REFERENCE.md Q-REPORT-LANG}}.
The root repository above stays Russian; the two histories are not mixed.

The subject is image processing research: lossy compression of noisy images,
prediction of distortions and visual quality metrics, denoising efficiency,
classification of compressed remote sensing images. Results go into papers
(`../papers/`).

## Read before writing code

| Address | Why |
|:---|:---|
| `../CLAUDE.md` R04 | Pareto, industrial simplicity, robustness-on-doubt and its second half |
| `../CLAUDE.md` R05 | the contract of a block: every field named, three links checked before code |
| `../CLAUDE.md` R06 | images: dtype, bit depth, range, channel order, colour space; noise seed, coder version and metric implementation are provenance; accumulation order is contract |
| `../CLAUDE.md` R07 | the oracle does not import the core; both sides of every decision are tested; fragments of one image never split across data roles |
| `../CLAUDE.md` R08 | every study is a runnable file the owner runs without reading code; paths from `lab_paths.json`; a number in a paper has a `NUMBERS.md` row |
| `../PROJECT.md` P03 | the core is a pure function: no files, no external coder calls, no network, no cache, no `lab`/`oracle` imports |
| `../PROJECT.md` P04 | the result is three things — value, uncertainty, coverage — and none stands in for another |
| `../REFERENCE.md#spec-blocks` | the contract of the block being implemented |
| `../REFERENCE.md#spec-result` | the result record and the flat list of refusal reasons |
| `../REFERENCE.md#ref-verify` | what is mandatory in a test; out-of-sample evaluation of predictors; fixtures stored as bytes |
| `../REFERENCE.md#result-audit` | the one home of the test count — report the number with its origin |
| `../REFERENCE.md#lab-experiment` | an experiment is a run → its own run file with rows → an Excel book → pages for the owner's eyes |
| `../REFERENCE.md#workflow` | the slot, the task's boundaries, what the report says in its first line |
| `../REFERENCE.md#data` | image sets, data roles, adapters, coder wrappers, medical images as codes |
| `../REFERENCE.md#papers` | paper figures and tables are built by a script from run files |
| `../REFERENCE.md#studies` | one study per paper: folder, STUDY.md, figures style file, own codec builds |
| `../REFERENCE.md#result-delivery` | the reproducibility archive from `git archive`, checked from the archive itself |

## Standing rules

**The contract outranks the prompt.** If a task contradicts a section of
`../REFERENCE.md` or an owner decision, stop and report the conflict instead of
resolving it silently.

**Never invent a value.** A missing number is not guessed and no default is
substituted: the block refuses and names what is missing. **Where a value comes
from — a function argument or `../config.json` — is stated in the block's own
contract.** The ban is on inventing, not on arguments. `null` in the config is
NOT ASSIGNED; a loader that reads it as zero or as "rule off" where the contract
did not say so disarms a protection silently.

**Documents above are read-only from here.** A needed config key is reported,
not added. A contract that does not match reality is reported, not edited.
`../papers/` is written only by the paper scripts into `figures/` and `tables/`;
manuscripts and `NUMBERS.md` are not edited from here — proposed `NUMBERS.md`
rows go into the report.

**Task in, report out, two overwritten files.** The task is `prompts/PROMPT.md`;
the answer is `prompts/REPORT.md`. Both are **overwritten, never multiplied** —
one live task and one live report at a time. **`prompts/` is ignored by git**
(owner decision 2026-10-13): no history of tasks and reports is kept. The work
itself is committed, and the report names its commits.

**Declare before the numbers what would count as failure.** Every report opens
with the criterion written before the run; a criterion written after the result
is a description, not a test.

**Report per the template in `prompts/REPORT.md`.** Verdict and addresses, not a
retelling: what was checked, with what, how it ended, the commit hash and the
composition of the diff. Every report ends with two sections: *where the contract
did not match what the tools actually do* and *what is not done*. Target 12 KB.
The test count is reported with its origin (command, commit) — it goes into
`../REFERENCE.md#result-audit` and nowhere else.

**A core test imports nothing of `lab` and no package of the `[lab]` group** — so
`-m core` runs without data and without external coders; the full suite runs in a
paper submission cycle or on a direct task (`../CLAUDE.md` R09). Report the
`-m core` count and the targeted tests.

**Nothing "for the future".** No dependency injection, no plugin systems, no
abstract base classes, no exception hierarchies. A flat set of named refusal
reasons is required and is not part of that list.

**The oracle is written without the implementation in context.** Otherwise it
mirrors the implementation's mistakes, and parity becomes a comparison of the
code with itself. Golden expectations from our own run test reproducibility,
not correctness; a `--record` flag that rewrites them is never used in a check.

**External coders and metric implementations are pinned.** A coder wrapper
records the executable, its version and the exact command line in provenance;
a metric records its implementation and version. Two implementations of the same
metric are never mixed in one table without a declared rule.

**Noise is seeded.** Every generated noise realisation has its seed in the run
file; a run without a seed is not reproducible and is refused.

**Medical images do not enter this repository.** Cases built from dental images
are ignored by name pattern (`.gitignore`); whether they may travel is the
owner's decision. `git add -A` must not take them. File names carrying patient
numbers or dates live only in `../Images/`, `../data/` and the cache; code,
tests, run files, logs, the report and commit messages carry codes (`D01`…).
Every report touching medical data prints the leak check.

**A run writes its own file and builds its own book.** The run file holds the
rows of the unit of analysis (image or fragment × condition), not only totals,
under a name with the cycle number; a later run refuses by name instead of
overwriting it. The Excel book is built by the run from that file and every cell
is checked against it. Pages for the eyes are chosen by a rule declared before
the numbers; the verdict column is left empty for the owner.

**One study, one folder, one paper.** `studies/P<NNN>_<topic>/` holds `STUDY.md`,
the runs, `figures.py` and `figures_style.toml` (`../REFERENCE.md#studies`). A
study uses the shared `src/` but does not edit it; a needed core change is reported.
Codec executables are our own builds from pinned tags (`tools/build_coders.ps1`).

**Paper figures are built, not drawn.** `studies/P<NNN>_<topic>/figures.py`
reads run files and the style file, writes CSV of each figure's data beside it, and writes
`../papers/P<NNN>/figures/` and `tables/` to the venue's format stated in
`PAPER.md`. All text in figures and tables — axes,
legends, captions, headers — is English; the manuscript body is Russian and is
translated by the owner. A number is never typed into a figure or a table by
hand.

**Notebooks call, they do not compute.** Formulas live in `lab/`; a cell chooses
and draws. A number from a notebook counts only after a top-to-bottom run on a
clean kernel; an edited cell that was not re-run has no output. A notebook is an
interactive bench, never the carrier of a published experiment.
