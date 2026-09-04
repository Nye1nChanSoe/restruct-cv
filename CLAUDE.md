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
uv run restruct <path> -o .                    # a directory: writes <resume>.json into it
uv run restruct <path> -o <out.json> --debug   # + stage 4-5 artifacts in <out>/
uv run restruct <path> -o <out.json> --stages 1-3   # + those stages (implies --debug)

uv run restruct                                # batch: every PDF in resumes-synthetic/
uv run restruct --truths                       # batch: only resumes-truths/ (local, gitignored)
uv run restruct --unsupported                  # batch: resumes-unsupported/ (see below)
```

`-o` takes a file or a directory. A directory — `.`, `out/`, an existing path — writes
`<resume>.json` inside it, named from the input rather than a fixed `output.json` so several
extractions into one directory do not overwrite each other. Detection uses the raw argument,
because `Path("out/")` normalises away the trailing separator that says "directory" about one
that does not exist yet.

`--stages` selects **debug artifacts, never whether a pass runs** — every pass feeds the next,
so a flag that skipped one would quietly produce a different resume. `--stages` implies
`--debug`. The batch form writes every stage: its purpose is to regenerate the committed corpus,
and anything less would let a stale artifact survive a run and make `git status examples/` read
as clean after a refresh.

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

It is looked for by `ingestion/ocr.find_tesseract()` on PATH first and then where installers put
it (Program Files on Windows, both Homebrew prefixes on macOS), and **only on a page that has
too little native text to parse** — a native PDF and a DOCX must run on a machine with no OCR
engine at all, which `tests/test_ocr.py` enforces. `SETTINGS.ocr.dpi` stays at 300: the 200 the
design asks for was measured and loses bullet markers, and the comment there records the
numbers.

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

**`results/` is untracked in full.** Every run rewrites it, and its overlays were 113MB of this
repository's history before that history was rewritten to drop them. Three resumes are kept as
committed evidence instead, one per ingestion track:

```
examples/7.anomaly/   native PDF: geometry, layout warnings, three overlays
examples/9.ocr/       scanned PDF: OCR rebuilt into the same line geometry
examples/11/          DOCX: no overlays, because there is no geometry to draw
```

They are **copied out of `results/`, never edited by hand** — `tools/refresh_examples.py` mirrors
each one, minus the pass overlays and reconstructions. That is what keeps them evidence rather
than illustration, and it is how a change that touches rendering or section parsing is verified:

```bash
uv run restruct && uv run python tools/refresh_examples.py
git status --short examples/    # clean == byte-identical
```

A file that stops being produced stops being committed, because the refresh empties each example
before copying. `EXAMPLES` in that script is the list; `test_no_committed_result_reports_unplaced_content`
reads it, so adding an example needs no second edit.

There are no per-section overlays: the combined image draws every one of their boxes with the same
colours and labels, and shows the boundaries between sections as well, so 65 PNGs were dropped
for 25MB and no information. The per-section renderers went with them — `git log` has them if a
`--stages` flag ever wants one back.

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
ingestion/   physical extraction — native PDF text, per-page OCR fallback, DOCX
document/    shared types (ExtractedLine, DetectedHeading, HeaderEntityMatch)
layout/      row clustering, paragraph/bullet accumulation, unsupported-layout detection
structure/   heading detection, routing, compound headings, precedence resolver, separators
parsers/     one module per section shape (header, experience, education, skills, grouped, urls)
models/      DistilBERT NER and MiniLM adapters  (currently still model.py)
stages.py    which debug artifacts each stage owns; importable without the models
patterns/    deterministic regex evidence, grouped by what it describes
debug/       artifacts (JSON), render (Pillow overlays), reconstruct (the result as a page)
schema.py    the lean, versioned clean output (contract: resume.schema.json)
errors.py    the failure taxonomy; only cli.py turns one into an exit code
pipeline.py  orchestration only — the only module that knows the stage order
cli.py       argparse and filesystem layout only
```

Where new code goes: shared by two parsers → `layout/` or `structure/`; a regex → `patterns/`;
box arithmetic → `geometry.py`; anything drawn or dumped → `debug/`.

### Two output tracks, deliberately separate

`resume.json` opens with `schema_version` (`schema.SCHEMA_VERSION`, currently `"1.0"`) and is
otherwise **lean and metadata-free**: no bboxes, fonts, geometry, model names,
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

### A reconstruction is read instead of the JSON, not beside it

`debug/reconstruct.py` draws `resume.json` back out as a readable page (`--reconstruct`). It
answers the question the overlays cannot: an overlay draws on the document, so the document keeps
making sense whatever was understood, and a bullet filed under education looks perfectly correct.
Throwing the page away and drawing only what was understood is what makes that visible.

Three properties hold it up, and each is a test:

- **It renders from the written `resume.json`**, never from the parser's types — so it is a
  consumer of the published contract, and a field the contract cannot express is one it cannot
  draw. `render_resume_file` reads the file back for exactly this reason.
- **Absent and empty values are skipped.** A page of "none" rows is a page nobody proof-reads.
- **Content it cannot place is drawn in red under UNPLACED.** Adding a schema field without
  teaching the renderer would otherwise make the field invisible, and invisible reads as an
  extraction failure. `_DRAWN_KEYS` is what that check is made of; a new schema key belongs there
  in the same commit.

It is **not** a facsimile. Imitating the source layout would hide the errors it exists to reveal,
and for a DOCX it would mean inventing the geometry ingestion refuses to invent.

The font is chosen against the characters of the document rather than by name: the corpus already
carries en dashes, curly quotes and Thai, and PDF's base-14 faces draw every one of those as a
middle dot — which a reader would take for an extraction bug. When nothing installed covers the
text, the page says which characters are missing. Weight is drawn by stroking the outline, because
the one face that covers a document may have no bold, and losing a glyph to gain a heavier heading
is a bad trade in a tool for reading.

Reconstructions are gitignored: they are drawn from a committed `resume.json`, so committing them
would store the same information twice and let the two disagree.

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

### DOCX has no geometry, and must never pretend to

`ingestion/docx.py` reads the XML with `python-docx`. **Do not switch it to MuPDF**, which will
open a DOCX and return convincing bounding boxes: it gets them by re-laying the document out
with substituted fonts into an invented page box, losing the paragraph styles, the list markers
and the table structure. Those boxes would then feed every document-relative statistic in the
package, which would measure a layout nobody laid out and produce plausible output from fiction.

`Document.has_geometry` is `False` for a DOCX, and every rule that compares points asks it
first: `is_paragraph_gap` never joins (paragraph boundaries are stated), `cell_gap_threshold` is
zero (cells are stated), column-gutter detection is skipped (there is no page), and `_visual_rows`
groups by the stated table row instead of by baseline. `bbox` carries reading order and indent
depth only.

The pass-1 dump is named after its reader — `raw-pymupdf.json` for a PDF, `raw-docx.json` for a
DOCX — because the two readers state different things. The DOCX dump holds the paragraphs
python-docx read: style name, indent steps, table cell, resolved run fonts, whether a run stated
a property itself or inherited it, and whether the bullet marker was in the document or put back
by the reader. It deliberately omits the ordinal boxes, which are the one thing in that reader
invented rather than read.

Losing geometry costs less than it sounds, because the DOCX states outright what the geometric
heuristics were reconstructing — a heading by style name, a list item by style, line boundaries
by paragraph, cells by table. Where it states a bullet only in the numbering part, the reader
puts the marker back into the text, so nine downstream bullet rules need no DOCX case.

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

### Performance properties that are easy to lose

Three things carry the run cost, and all three regress silently:

- **A fixed reference set is embedded once per model**, through
  `model.encode_references()`. Encoding them per call was 48% of a run — roughly two hundred
  configuration phrases re-embedded once per experience metadata line. Never route *candidate*
  text through that cache: candidates are per-document and unbounded, so caching them is a leak.
- **Models load on first use**, via `LazyEmbeddingModel` / `LazyNerPredictor`. The presence
  check stays eager so missing weights still exit with their own code immediately.
- **`restruct/__init__.py` must not import the pipeline eagerly.** It re-exports `main` and
  `extract_resume` through PEP 562 `__getattr__` because a plain import cost four seconds of
  torch and transformers that `--help` never needs. `tests/test_performance.py` asserts that
  importing `restruct.cli` leaves `sys.modules` free of them.

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
is `prompt.txt`. Milestone 3 is complete, as are C13, C17 (CLI), C18 (perf), C19 (DOCX), C20 (Tesseract
detection; the 300→200 DPI half was measured and rejected — see `SETTINGS.ocr.dpi`) and C21
(schema version, packaging). All 21 commits of the plan are now implemented.

**Known and unfixed:** `header.job_titles` is the weakest field (P=0.60). The semantic tier
over-claims on taglines — `PRODUCT DESIGNER · UX/UI` yields two titles, and resume 6's
`ALEX MORGAN` is claimed as a title, which is why that fixture reports no name. Six of the eight
false positives predate the DOCX work. Fixing it means constraining what the semantic tier may
claim, and it is the single highest-value extraction change left.

C15 left one item of its plan undone on purpose: `_experience_line_entities` still reconciles
titles and companies across segments with its own logic rather than through `SpanResolver`. The
reconciliation is a comparison *between* segments, not a first-come claim, so it does not fit
the resolver's shape without a redesign that would move output. The `@` split and the date
spans in that function do run deterministically ahead of the model.

Model weights are resolved at run time by `cli._models_directory()`, not by
`Path(__file__).parents[2]`. That expression is the checkout when running from source and
`site-packages/..` in an installed wheel, so a pip-installed `restruct` reported missing weights
on a machine that had them. The order is `RESTRUCT_MODELS_DIRECTORY` (which settles it alone),
then `models/` beside the checkout *if it is one*, then `models/` under the working directory,
then `~/.restruct/models`; `ModelAssetsMissing` names every place it looked. The settings name the
model folder (`all-MiniLM-L6-v2`), and the loaders take the directory those folders live in.

## Test data

Fixtures may be `.pdf` or `.docx`; `tests/helpers.fixture_path()` resolves a stem to whichever
exists, so a fixture can change format without renaming its golden file or its labels. Every
synthetic fixture must have hand-written labels — `test_every_synthetic_fixture_has_a_label`
enforces it.

Never commit real resumes or their labels. `resumes-truths/` is gitignored for exactly this
reason, including any `resumes-truths/labels/` the scorecard picks up. Fixtures in
`resumes-synthetic/` are synthetic and safe to commit.
