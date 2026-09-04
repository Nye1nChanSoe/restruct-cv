# Restruct

Restruct turns messy PDF, DOCX, and scanned resumes into consistent, explainable, versioned JSON.
It combines OCR, layout geometry, NER, semantic similarity, and deterministic rules so every result
can be traced back to the thing on the page that produced it.

Built for applicant tracking systems, candidate search, analytics pipelines, talent platforms, and
other systems that need trustworthy resume data.

---

## Install

```bash
uv sync                      # from a checkout
```

or, from a built wheel:

```bash
uv build
pip install dist/restruct-*.whl
```

### Model weights

All inference weights are **local-only** and never fetched at run time. Two directories are
required:

```text
models/
  all-MiniLM-L6-v2/
  distilbert-NER/
```

They are looked for, in order, in `models/` beside the checkout (when running from one), `models/`
under the current working directory, and `~/.restruct/models`. Set
`RESTRUCT_MODELS_DIRECTORY=/path/to/models` to name the directory outright, which is the usual
answer for an installed copy; when it is set nothing else is consulted. A run with no weights
exits `20` and names every place it looked.

### Tesseract

Tesseract is a system dependency, needed **only** for scanned pages — a PDF with its own text
layer and a DOCX never ask for it.

```bash
brew install tesseract                 # macOS
apt-get install tesseract-ocr          # Debian/Ubuntu
```

It is found on `PATH`, or where the installers put it (Program Files on Windows, either Homebrew
prefix on macOS). PyMuPDF renders an OCR page at 300 DPI and `tesseract` is invoked directly,
without a shell, using the English language data, LSTM engine mode 1 and page segmentation mode 3.
Rendered page images are deleted after each document.

## Usage

```bash
uv run restruct <resume.pdf> -o out.json      # one resume; quiet, writes nothing else
uv run restruct <resume.docx> -o .            # a directory: writes <resume>.json into it
uv run restruct <resume.pdf> -o out.json --debug        # + stage 4-5 artifacts in out/
uv run restruct <resume.pdf> -o out.json --stages 1-3   # + those stages (implies --debug)
uv run restruct <resume.pdf> -o out.json --reconstruct  # + the result drawn back as a page
uv run restruct results/1/resume.json --reconstruct     # draw a result already extracted

uv run restruct                               # batch over resumes-synthetic/
uv run restruct --truths                      # batch over resumes-truths/ (local, gitignored)
uv run restruct --unsupported                 # batch over resumes-unsupported/
```

`--stages` selects **debug artifacts, never whether a pass runs**: every pass feeds the next, so a
flag that skipped one would quietly produce a different resume.

Exit codes are an API, grouped by decade — `1x` input, `2x` environment, `3x` extraction, `4x`
output, with `2` left to argparse. `errors.py` names every failure and `cli.py` is the only module
that maps one to a code, so a caller embedding restruct as a library catches an exception by type
instead of losing its process.

## Output

`resume.json` is lean and metadata-free: no bounding boxes, fonts, geometry, model names,
confidences or detection methods. Its first key is `schema_version`, and `resume.schema.json` in
the repository root is the published contract every fixture is validated against.

Sixteen destinations are always present, always in the same order. A section is `null` when the
resume has none, `[]` when it exists but yielded no entries; a key is never absent.

```json
{
  "schema_version": "1.0",
  "header_profile": { "name": "…", "emails": ["…"], "phones": ["…"] },
  "summary": null,
  "experience": [{ "job_titles": ["…"], "companies": ["…"], "bullets": ["…"] }],
  "education": [],
  "…": "…",
  "others": []
}
```

All the evidence behind those values — boxes, fonts, confidences, which method decided what —
lives in a separate track under `raw/`, and the overlays draw it:

```text
results/<name>/
  resume.json           the lean output
  raw/*.json            the evidence: boxes, fonts, confidences, methods
  debug/page-N.png      the combined overlay, one per source page
  debug/pass-*/         passes 1-4, gitignored
```

The pass-1 dump is named after the reader that produced it — `pymupdf.json` for a PDF, `docx.json`
for a DOCX, since the two read different things and only one of them has coordinates. The
single-file form writes it beside the other evidence in `raw/`; the batch writes it to
`debug/<name>.raw-<reader>.json`.

A model-backed box is drawn more heavily than a deterministic one, so a reader can tell whether a
box is something the document said or something a model concluded.

### Reading the result back

`--reconstruct` draws `resume.json` back out as a page — `reconstruction.pdf` and one PNG per
page — so the result can be proof-read by eye.

It answers a different question from the overlays. An overlay draws on top of the document, so it
shows whether a box landed on the right words; the document keeps making sense regardless of what
was understood. A reconstruction throws the page away and draws only what was understood, which is
what makes a bullet filed under education or a date read as a job title visible at a glance.

It is deliberately not a facsimile — imitating the original layout would hide the errors it exists
to reveal. Absent and empty fields are skipped, so what is on the page is what was extracted, and
anything the renderer cannot place is drawn in red under UNPLACED rather than dropped. Given a
`resume.json` as the input it draws that and runs nothing else, loading no models.

## Architecture

Five ordered passes over one shared in-memory document.

```text
ingestion/   physical extraction — native PDF text, per-page OCR fallback, DOCX
document/    shared types and document-wide statistics
layout/      row clustering, paragraph/bullet accumulation, unsupported-layout detection
structure/   heading detection, routing, compound headings, precedence resolver, separators
parsers/     one module per section shape (header, experience, education, skills, grouped, urls)
model.py     DistilBERT NER and MiniLM adapters, loaded on first use
patterns/    deterministic regex evidence, grouped by what it describes
debug/       artifacts (JSON) and overlays (Pillow), one canvas, one colour registry
schema.py    the lean, versioned output (contract: resume.schema.json)
errors.py    the failure taxonomy; only cli.py turns one into an exit code
pipeline.py  orchestration only — the only module that knows the stage order
cli.py       argparse and filesystem layout only
```

Extraction runs in a fixed precedence — deterministic, context-sensitive deterministic, NER,
MiniLM, geometry, then `other` — and `structure/resolver.py` **enforces** that order rather than
documenting it: a stage that runs after a weaker one has already claimed a span raises.

v1 targets single-column resumes. Layouts whose reading order cannot be recovered (column gutters,
vertical text, overlapping boxes, nested tables) are **recorded, never repaired**, in
`debug/layout-warnings.json`.

## Development

```bash
uv run pytest                                # full suite
uv run python -m tests.scorecard             # per-field precision/recall/F1
uv run restruct && git status --short results/    # clean == byte-identical output
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the test and scorecard workflow, and `CLAUDE.md` for the
design decisions behind each module.

Please do not submit **real** resumes or their labels; `resumes-truths/` is gitignored for exactly
that reason. Fixtures in `resumes-synthetic/` are synthetic and safe to commit.
