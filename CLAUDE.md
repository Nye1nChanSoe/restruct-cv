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

uv run restruct <path> -o <out.json>           # one resume; quiet, writes nothing else
uv run restruct <path> -o <out.json> --debug   # + stage 4-5 artifacts in <out>/
uv run restruct <path> -o <out.json> --stages 1-3   # + those stages (implies --debug)

uv run restruct                                # batch: every PDF in resumes-synthetic/
uv run restruct --truths                       # batch: only resumes-truths/ (local, gitignored)
uv run restruct --unsupported                  # batch: resumes-unsupported/ (see below)
```

`--stages` selects **debug artifacts, never whether a pass runs** — every pass feeds the next,
so a flag that skipped one would quietly produce a different resume. `--stages` implies
`--debug`. The batch form writes every stage: its purpose is to regenerate the committed corpus,
and anything less would let a stale artifact survive a run and make `git status results/` read
as clean.

`errors.py` names every failure; `cli.py` is the only place that maps one to an exit code, so
the Python API raises and a caller embedding restruct keeps its process. Codes are grouped by
decade — 1x input, 2x environment, 3x extraction, 4x output — and `2` is left to argparse.

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

Debug output is not in the golden set. Each resume writes three things, and the split is on
disk as well as in shape:

```
results/<name>/resume.json           the lean, metadata-free output
results/<name>/raw/*.json            the evidence: boxes, fonts, confidences, methods
results/<name>/debug/page-N.png      the combined overlay, one per source page
results/<name>/debug/pass-*/         passes 1-4, gitignored
```

`results/` holds **86 committed artifacts** (64 raw JSON, 15 overlays, 7 `resume.json`). There
are no per-section overlays: the combined image draws every one of their boxes with the same
colours and labels, and shows the boundaries between sections as well, so 65 PNGs were dropped
for 25MB and no information. The per-section renderers went with them — `git log` has them if a
`--stages` flag ever wants one back.

To verify a change that touches rendering or section parsing:

```bash
uv run restruct && git status --short results/    # clean == byte-identical
```

**Passes 1-4 render images and no JSON**, because their output is geometry: a count can be
right while every box sits ten points too low. They are written to
`results/<name>/debug/pass-{1-physical,2-words,3-lines,4-sections}/` and are gitignored —
large, regenerated every run, and opt-in by design. **Look at them when changing anything
geometric.** Each carries a legend naming its layers and their counts.

- **pass 1** also draws unsupported-layout warnings, so a detected column gutter is visible
  as a box in the corridor it claims to have found.
- **pass 4** draws which destination each heading became. A compound heading splits into
  several sections from one line, so each is drawn inset and labelled
  `languages <- CERTIFICATIONS & LANGUAGES` — the split is right or wrong at a glance.
- **pass 5** is the one committed overlay, `debug/page-N.png`, carrying every section at once.

Every overlay goes through `debug/canvas.py`: one page canvas, one label placer, one legend.
**A model-backed box is drawn more heavily than a deterministic one**, so a reader can tell
whether a box is something the document said or something a model concluded. That question is
asked of `resolver.is_model_backed()`, not of how the method name is spelled — the old prefix
test for `distilbert`/`minilm` silently missed `semantic_similarity`,
`ner_minilm_reconciliation` and `geometry_ner_reconstruction`.

The unsupported fixtures are not in the batch run, since their parse is untrustworthy by
definition. Render their overlays with:

```bash
uv run restruct --unsupported      # resumes-unsupported/ -> results/1-unsupported/ (gitignored)
```

This is not optional diligence. Rendering the pass-1 overlay is what found that five Year
cells in `7.anomaly`'s certification table were being classified as running footers; every
numeric test passed while that was true.

## Architecture

Five ordered passes over one shared in-memory document. The current code implements passes 4-5
fully; passes 1-3 are being built (see *Refactor in flight*).

```
ingestion/   physical extraction — native PDF text, per-page OCR fallback
document/    shared types (ExtractedLine, DetectedHeading, HeaderEntityMatch)
layout/      row clustering, paragraph/bullet accumulation, unsupported-layout detection
structure/   heading detection, routing, compound headings, precedence resolver, separators
parsers/     one module per section shape (header, experience, education, skills, grouped, urls)
models/      DistilBERT NER and MiniLM adapters  (currently still model.py)
patterns/    deterministic regex evidence, grouped by what it describes
debug/       artifacts (JSON) and render (Pillow overlays), one canvas, one colour registry
schema.py    the lean, versioned clean output (contract: resume.schema.json)
errors.py    the failure taxonomy; only cli.py turns one into an exit code
pipeline.py  orchestration only — the only module that knows the stage order
cli.py       argparse and filesystem layout only
```

Where new code goes: shared by two parsers → `layout/` or `structure/`; a regex → `patterns/`;
box arithmetic → `geometry.py`; anything drawn or dumped → `debug/`.

### Two output tracks, deliberately separate

`resume.json` is **lean and metadata-free**: no bboxes, fonts, geometry, model names,
confidences, or detection methods. All of that evidence lives only under `raw/`.
`test_output_carries_no_debug_metadata` enforces the separation — if you add a field to the
clean schema, it must be plain data.

`resume.schema.json` in the repository root is the published contract, hand-written rather than
generated. `test_output_matches_the_published_schema` validates every fixture against it, so it
cannot drift into a stale description of output it no longer describes. A new field means
editing it in the same commit.

### Extraction precedence

Deterministic → context-sensitive deterministic → NER → MiniLM → geometry → `other`/unresolved.

`structure/resolver.py` is the one place this is written down, and it is **enforced, not
documented**: `SpanResolver.open(tier)` raises `PrecedenceError` if a stage runs after a weaker
one has already opened. That check exists because the order had silently drifted — the header's
MiniLM attribute stage was running before both the contact regexes and NER, and no test could
see it.

Two different questions, deliberately separate:

- `is_claimed()` asks about **characters** — a weaker stage cannot take spans a stronger one
  already read. This is why `HeaderEntityMatch` carries `start`/`end` offsets into the source
  line: a later parser must always be able to see, and reverse, a split.
- `has_kind()` asks about **fields** — the geometry name guess and the nationality fallback run
  only while their field is still missing. A stage that mixes the two will look right and behave
  wrongly.

Within the deterministic tier, order still matters and is not arbitrary: an explicit label is
the document naming the field itself, so `Date of Birth: 12/05/1995` is claimed before the broad
contact shapes — otherwise the digits are consumed as a phone number.

### A separator is evidence, not an instruction

`structure/separators.py` answers what a separator means *given its surroundings*, because the
same character does several jobs: a colon labels a field (`Languages:`) or sits inside a time
(`09:30`); a dash joins two dates into a range (`2019 - 2022`) or separates two fields
(`Senior Analyst - Logistics`); an `@` separates a role from an employer. A colon is a label
when the left side is short, **or** when the rows around it are labelled the same way — a
labelled block is a layout the document is committing to. Every split keeps the original text
and the offsets of both parts.

**An unclosed bracket outranks the geometry.** A line ending `... (non-licensed` has not
finished saying what it names, so it is not separator-parsed as it stands and the line under it
completes it — `continues_block(previous_text=...)` overrides the paragraph-gap test. The
override is bounded to the same page, downward, and three line heights, so one stray `(` cannot
join a whole section. A separator found *inside* brackets is part of the phrase, never a field
boundary, so splits are taken at depth zero.

**Never classify content merely because it follows a heading.** Ambiguous content stays `other`.

### Unsupported layouts are recorded, never repaired

v1 targets single-column resumes. `layout/unsupported.py` detects the shapes whose reading order
cannot be recovered — column gutters, vertical text, overlapping text boxes, text inside a
graphic, a table nested in a table — and writes them to `debug/layout-warnings.json`, which is
written even when empty so an absent finding is distinguishable from an absent check. The only
behavioral consequence is that row grouping refuses to join cells across a gutter. Nothing here
reconstructs a multi-column reading order; that is a later version.

The fixtures are `resumes-unsupported/3.cols.pdf` and `4.cols.pdf`. **The six synthetic resumes
must keep producing no warning at all** — a detector that fires on supported input teaches
readers to ignore it.

### Fixed destinations

Sixteen, in `schema.V1_SECTION_ORDER`, always present in the same order. `others` is the
conservative fallback and preserves the original heading text.

### Compound headings are split on evidence, never on shape

`structure/compound.py` turns one physical section into the logical sections it contains.
Components are classified by **exact** match against the reference lists and nothing else — a
near match is not evidence, which is why `SAFETY & TRAINING` stays in `others` rather than
being guessed into certifications. A heading that is itself a reference name
(`Education and Training`) is never split.

A component that names a destination does not yet own anything. Content is claimed by an
explicit label (`Certifications: …`), by a local subheading, or by deterministic evidence for
one destination; a key-value label claims only its own line, while a subheading owns the run
beneath it. Then:

- nothing claimed anything → the first component naming a destination takes the whole section
- something claimed and something did not → the remainder is genuinely unowned, so it goes to
  `others` under the heading as written
- a component that ends up owning no lines produces **no section at all** — an empty one
  registers a destination that owns nothing, and the real section of that type is then the
  second occurrence and goes unread

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
is `prompt.txt`. **Milestone 3 is complete** (C14 compound headings, C15 ordered precedence,
C16 unified renderer), as is C13. Milestone 4 — the product surface, starting with C17's real
`restruct <path> -o <out>` CLI — is next.

C15 left one item of its plan undone on purpose: `_experience_line_entities` still reconciles
titles and companies across segments with its own logic rather than through `SpanResolver`. The
reconciliation is a comparison *between* segments, not a first-come claim, so it does not fit
the resolver's shape without a redesign that would move output. The `@` split and the date
spans in that function do run deterministically ahead of the model.

`README.md`'s architecture section is stale: it describes the `model.py` / `routing.py` /
`__init__.py` split that no longer exists. Scheduled for the final packaging commit.

## Test data

Never commit real resumes or their labels. `resumes-truths/` is gitignored for exactly this
reason, including any `resumes-truths/labels/` the scorecard picks up. Fixtures in
`resumes-synthetic/` are synthetic and safe to commit.
