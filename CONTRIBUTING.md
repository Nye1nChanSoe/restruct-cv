# Contributing

Contributions are warmly welcome. Thank you for helping improve this project.

## Ways to Contribute

You can contribute by:

- Fixing bugs or improving extraction reliability.
- Improving OCR (`tesseract`) handling for scanned resumes.
- Testing the extractor against different resume formats and reporting edge cases.
- Improving documentation or developer setup instructions.
- Adding support for Burmese/English mixed-language resumes.

## Setup

```bash
uv sync
```

Model weights are local-only and gitignored; see [README.md](README.md#model-weights) for the two
directories and where they are looked for. Tesseract is only needed for the scanned fixtures. If
either is missing, the tests skip rather than fail, so a fresh clone stays green and you can start
on the model-free parts right away.

## The dev CLI

The commands a contributor needs live in `tools/dev.py` rather than in the shipped `restruct`
command. Everything in there reads or writes a directory that only exists in a checkout, so
shipping it as a console script would hand every PyPI user a command that cannot work.

```bash
uv run tools/dev.py batch [--truths | --unsupported] [--stages 1-5] [--reconstruct]
uv run tools/dev.py examples                        # refresh examples/ from results/
uv run tools/dev.py scorecard [--update-baseline]
uv run --group export tools/dev.py export-onnx      # re-export models/*/model.onnx
uv run tools/dev.py --version
```

`batch` writes every stage by default. Its job is to regenerate the committed corpus, and writing
less would let a stale artifact survive a run and quietly look correct.

The shipped `restruct` stays small on purpose: `-o`, `--ats`, `--reconstruct`, `--install-models`
and `--version`. There is one more, `--stages`, kept out of `--help` for the per-document case
described below.

## The two guards

There are two checks, and it is worth running both, because they answer different questions.

```bash
uv run pytest                                  # full suite (~50s; loads both models once)
uv run pytest tests/test_patterns.py           # fast, model-free unit tests
uv run pytest -k "golden and 7.anomaly"        # one fixture

uv run tools/dev.py scorecard                  # per-field precision/recall/F1
```

- **`tests/golden/`** holds byte-for-byte snapshots of `resume.json` for every synthetic fixture.
  They tell you that something *changed*. The pipeline is deterministic, so an empty diff really
  does mean nothing moved.
- **`tests/labels/` + `tests/scorecard.py`** hold ground truth written by hand, read off each
  resume rather than copied from pipeline output. They tell you whether the output got *worse* —
  which a snapshot on its own cannot distinguish from a fix. `tests/baseline_scores.json` is the
  per-field F1 floor the suite enforces.

### Change budget

The one rule worth stating flatly: **don't re-baseline a golden snapshot just to make a test
pass.** If a refactor produces a diff, the refactor changed behavior, and the code is what needs
fixing.

Beyond that it comes down to which kind of change you are making:

- **A refactor** — moving code around, removing duplication — should leave the output
  byte-identical. If it doesn't, something real changed.
- **A behavior change** is expected to produce diffs. Read through them, re-baseline in the same
  commit with `uv run pytest --update-golden`, and put the before/after scorecard in the commit
  message so the next person can see what the change cost or bought. F1 shouldn't drop on any
  field.

One caveat: the scorecard doesn't score bullets or paragraphs. If your change touches OCR or
layout, read the golden diff too — an unchanged scorecard isn't evidence on its own that nothing
got worse.

### Verifying what the snapshots do not cover

Debug artifacts aren't in the golden set. `results/` is untracked entirely — every run rewrites it
and the overlays are large. What is committed instead is `examples/`: three resumes, one per
ingestion track, copied out of `results/` by a script rather than curated by hand. If your change
touches rendering or section parsing, regenerate them and see whether anything moved:

```bash
uv run tools/dev.py batch && uv run tools/dev.py examples
git status --short examples/    # clean == byte-identical
```

Passes 1-4 render images and no JSON, because what they produce is geometry — a count can be
perfectly right while every box sits ten points too low. **Look at them whenever you change
anything geometric.**

```bash
uv run restruct <resume.pdf> -o out.json --stages 1-3   # one resume, hidden flag
uv run tools/dev.py batch --stages 1-3                  # the whole corpus
```

And to see what was *understood* rather than where the boxes landed, draw the result back out and
read it:

```bash
uv run restruct <resume.pdf> -o out.json --reconstruct
uv run restruct examples/11/resume.json --reconstruct   # or one already extracted
```

## Adding a field or a section

`resume.schema.json` is the published contract. It's hand-written rather than generated, and every
fixture is validated against it, so a new field means editing it **in the same commit**. The field
also has to be plain data: no bounding boxes, fonts, geometry, model names, confidences or
detection methods make it into `resume.json`. That evidence belongs in the `raw/` track.

You'll also want to teach `debug/reconstruct.py` how to draw the field. If you forget, it isn't
dropped silently — it's drawn in red under UNPLACED, and a test fails on any committed result that
reports one, so the reminder finds you.

Every synthetic fixture needs hand-written labels; a test enforces that too.

## Conventions

- **No abbreviated identifiers.** `rectangle` not `rect`, `document` not `doc`, `previous_box`.
- Comments explain *why*, never *what*. Public functions carry docstrings.
- New modules open with `from __future__ import annotations` and carry full type hints.
- Commit subjects: `feat:` `refactor:` `test:` `chore:` `perf:` `update:`.

## Privacy and Test Data

Please don't submit real resumes containing PII, or their labels — `resumes-truths/` is gitignored
for exactly that reason. Anonymized data is fine for examples, test files, bug reports and
reproduction cases.

Thank you for contributing!
