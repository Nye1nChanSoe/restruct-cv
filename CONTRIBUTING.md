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
directories and where they are looked for. Tesseract is needed only for the scanned fixtures.
Tests **skip** rather than fail when either is absent, so a fresh clone stays green.

## The two guards

Both must pass on every commit, because they answer different questions.

```bash
uv run pytest                                  # full suite (~50s; loads both models once)
uv run pytest tests/test_patterns.py           # fast, model-free unit tests
uv run pytest -k "golden and 7.anomaly"        # one fixture

uv run python -m tests.scorecard               # per-field precision/recall/F1
```

- **`tests/golden/`** holds byte-for-byte snapshots of `resume.json` for every synthetic fixture.
  They catch any *change*. The pipeline is deterministic, so an empty diff is a real signal.
- **`tests/labels/` + `tests/scorecard.py`** hold hand-written ground truth, read off each resume
  rather than taken from pipeline output. They catch output getting *worse*, which a snapshot
  cannot distinguish from a fix. `tests/baseline_scores.json` is the enforced per-field F1 floor.

### Change budget

**Never re-baseline a golden snapshot to make a test pass.** A diff during a refactor means the
refactor changed behavior — fix the code.

- **Refactor** (moving code, deduplicating): output must be **byte-identical**. Hard gate.
- **Behavior change**: diffs are expected. Review every line, re-baseline *in the same commit*
  with `uv run pytest --update-golden`, and put the before/after scorecard in the commit message.
  Scorecard F1 must not drop on any field.

The scorecard does not score bullets or paragraphs, so a change that touches OCR or layout must be
reviewed against the golden diff as well — an identical scorecard is not by itself evidence that
nothing got worse.

### Verifying what the snapshots do not cover

Debug artifacts are not in the golden set. To check a change that touches rendering or section
parsing, regenerate the committed corpus and confirm nothing moved:

```bash
uv run restruct && git status --short results/    # clean == byte-identical
```

Passes 1-4 render images and no JSON, because their output is geometry: a count can be right while
every box sits ten points too low. **Look at them when changing anything geometric.**

```bash
uv run restruct <resume.pdf> -o out.json --stages 1-3
```

## Adding a field or a section

`resume.schema.json` is the published contract, hand-written rather than generated, and every
fixture is validated against it. A new field means editing it **in the same commit**, and the field
must be plain data: no bounding boxes, fonts, geometry, model names, confidences or detection
methods reach `resume.json`. That evidence belongs in the `raw/` track.

Every synthetic fixture must have hand-written labels; a test enforces it.

## Conventions

- **No abbreviated identifiers.** `rectangle` not `rect`, `document` not `doc`, `previous_box`.
- Comments explain *why*, never *what*. Public functions carry docstrings.
- New modules open with `from __future__ import annotations` and carry full type hints.
- Commit subjects: `feat:` `refactor:` `test:` `chore:` `perf:` `update:`.

## Privacy and Test Data

Please do not submit real resumes containing PII, or their labels. `resumes-truths/` is gitignored
for exactly this reason. When providing examples, test files, bug reports, or reproduction cases,
use anonymized data.

## Major Changes

For major changes, please open an issue first to discuss your idea before submitting a pull
request. This helps ensure that proposed changes align with the project's direction and avoids
unnecessary work.

## Pull Requests

When submitting a pull request:

- Keep the change focused on a single issue or improvement.
- Clearly describe what was changed and why.
- Include reproduction steps when fixing a bug.
- Add or update tests when appropriate.
- Update documentation if your change affects setup, usage, or behavior.

Thank you for contributing!
