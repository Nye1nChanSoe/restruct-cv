# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run pytest                                  # full suite (~50s; loads both models once)
uv run pytest tests/test_patterns.py           # fast, model-free unit tests
uv run pytest -k "golden and 7.anomaly"        # one fixture
uv run pytest --update-golden                  # re-baseline snapshots (see Change budget)

uv run python -m tests.scorecard               # per-field precision/recall/F1
uv run python -m tests.scorecard --update-baseline   # re-freeze the accuracy floor

uv run restruct                                # batch: parse every PDF in resumes-synthetic/
uv run restruct --truths                       # batch: only resumes-truths/ (local, gitignored)
```

`restruct <path> -o <out>` does not exist yet; the CLI is still batch-over-directory and is
rebuilt in Milestone 4.

### Setup expectations

Model weights are **local only** and gitignored — `AutoTokenizer`/`SentenceTransformer` are
loaded with `local_files_only=True` and never hit the network. Both directories must exist:

```
models/all-MiniLM-L6-v2/     models/distilbert-NER/
```

Tesseract is a system dependency (`brew install tesseract`), needed only for the scanned
fixtures. Tests **skip** rather than fail when models or Tesseract are absent, so a fresh clone
stays green.

## The test harness is the point

The repository had no tests before this refactor. Two guards were added because they answer
different questions, and **both must pass on every commit**:

- **`tests/golden/`** — byte-for-byte snapshots of `resume.json` for the six synthetic fixtures.
  Catches any *change*. The pipeline is deterministic, so an empty diff is a real signal.
- **`tests/labels/` + `tests/scorecard.py`** — hand-written ground truth, derived by reading each
  resume (the scanned ones from rendered page images, never from OCR output). Catches output
  getting *worse*, which a snapshot cannot distinguish from a fix.
  `tests/baseline_scores.json` is the enforced per-field F1 floor.

### Change budget

**Never re-baseline a golden snapshot to make a test pass.** A diff during a refactor means the
refactor changed behavior — fix the code.

- Refactor commits (moving code, deduplicating): output must be **byte-identical**. Hard gate.
- Behavior commits (Pass 3 onward): diffs are expected. Review every line, re-baseline *in the
  same commit*, and record the before/after scorecard in the commit message. Scorecard F1 must
  not drop on any field.

### Verifying work the snapshots do not cover

Debug images are not in the golden set. `results/` has **121 committed debug artifacts**
(68 PNGs, 53 JSON) covering passes 4-5. To verify a change that touches rendering or
section parsing:

```bash
uv run restruct && git status --short results/    # clean == byte-identical
```

**Passes 1-3 render images and no JSON**, because their output is geometry: a count can be
right while every box sits ten points too low. They are written to
`results/<name>/debug/pass-{1-physical,2-words,3-lines}/` and are gitignored — large,
regenerated every run, and opt-in by design. **Look at them when changing anything
geometric.** Each carries a legend naming its layers and their counts.

This is not optional diligence. Rendering the pass-1 overlay is what found that five Year
cells in `7.anomaly`'s certification table were being classified as running footers; every
numeric test passed while that was true.

## Architecture

Five ordered passes over one shared in-memory document. The current code implements passes 4-5
fully; passes 1-3 are being built (see *Refactor in flight*).

```
ingestion/   physical extraction — native PDF text, per-page OCR fallback
document/    shared types (ExtractedLine, DetectedHeading, HeaderEntityMatch)
layout/      row clustering, paragraph/bullet block accumulation
structure/   heading detection, section routing, key-value pairs, metadata splitting
parsers/     one module per section shape (header, experience, education, skills, grouped, urls)
models/      DistilBERT NER and MiniLM adapters  (currently still model.py)
patterns/    deterministic regex evidence, grouped by what it describes
debug/       artifacts (JSON) and render (Pillow overlays), one colour registry
schema.py    the lean, versioned clean output
pipeline.py  orchestration only — the only module that knows the stage order
cli.py       argparse and filesystem layout only
```

Where new code goes: shared by two parsers → `layout/` or `structure/`; a regex → `patterns/`;
box arithmetic → `geometry.py`; anything drawn or dumped → `debug/`.

### Two output tracks, deliberately separate

`resume.json` is **lean and metadata-free**: no bboxes, fonts, geometry, model names,
confidences, or detection methods. All of that evidence lives only in the debug artifacts.
`test_output_carries_no_debug_metadata` enforces the separation — if you add a field to the
clean schema, it must be plain data.

### Extraction precedence

Deterministic regex → context-sensitive deterministic → NER → MiniLM → geometry →
`other`/unresolved. The mechanism is `overlaps_existing()`: each stage skips character spans an
earlier, stronger stage already claimed. This is why `HeaderEntityMatch` carries `start`/`end`
offsets into the source line — a later parser must always be able to see, and reverse, a split.

**Never classify content merely because it follows a heading.** Ambiguous content stays `other`.

### Fixed destinations

Sixteen, in `schema.V1_SECTION_ORDER`, always present in the same order. `others` is the
conservative fallback and preserves the original heading text. Compound headings
(`CERTIFICATIONS & LANGUAGES`) are **not yet split** — that is Milestone 3.

### OCR converges on the native types

`ingestion/ocr.py` rebuilds Tesseract TSV into the same line geometry the native path produces,
so nothing downstream needs OCR-specific handling. Preserve this when touching ingestion.

## Conventions

- **No abbreviated identifiers.** `rectangle` not `rect`, `document` not `doc`, `previous_box`,
  `line_index`. The codebase is consistent on this; match it.
- Comments explain *why*, never *what*. Public functions carry docstrings.
- New modules open with `from __future__ import annotations` and carry full type hints.
  The `configs/` modules and the re-export `__init__.py` files predate this and do not.
- Commit subjects: `feat:` `refactor:` `test:` `chore:` `perf:` `update:`.

### Gotchas that have already caused bugs

- **Invisible characters in regexes must be written as escapes.** `patterns/bullets.py` names
  U+F0B7 (the Private Use Area bullet Word exports) and U+200B / U+FEFF as `\uXXXX` escapes.
  They render as nothing, and one was silently lost when a character class was retyped.
  Refer to them by codepoint in prose too -- pasting the literal into a file reintroduces
  exactly the problem.
- **Comparing regex source text is not a valid equivalence check.** A pattern written with
  a `\uXXXX` escape and one written with the literal character compile to the same regex but
  differ as strings. Compare *behavior* over a corpus before deleting any original.
- **Two different routed-heading lists exist.** `build_sections` routes from the first header
  boundary; the section parsers route from line 0. They are not interchangeable.
- `_routed_section_headings` is O(n²) and currently recomputed ~15× per document. Known; queued.

## Refactor in flight

An approved 21-commit plan lives at
`~/.claude/plans/read-prompt-txt-first-before-robust-forest.md`; the design brief it implements
is `prompt.txt`. Milestones 0-1 (safety net, module extraction) are complete — 9 commits, all
byte-identical. Milestone 2 (passes 1-3) is next and is where output starts changing.

`README.md`'s architecture section is stale: it describes the `model.py` / `routing.py` /
`__init__.py` split that no longer exists. Scheduled for the final packaging commit.

## Test data

Never commit real resumes or their labels. `resumes-truths/` is gitignored for exactly this
reason, including any `resumes-truths/labels/` the scorecard picks up. Fixtures in
`resumes-synthetic/` are synthetic and safe to commit.
