# extractor-v1

A deliberately small experiment:

1. PyMuPDF extracts native PDF lines and geometry.
2. Pages without meaningful native text fall back to PyMuPDF's Tesseract OCR integration.
3. `all-MiniLM-L6-v2` locates the first likely resume section heading.
4. Everything above that boundary becomes one top-level `headerProfile` region.
5. Contact regex claims and masks email, phone, and URL spans first.
6. The selected NER backend processes each remaining header line independently
   for names, locations, and contextual nationalities.
7. MiniLM and exact phrase matching classify split job-title candidates.
8. Geometry fills missing identity fields, and unmatched visible text is retained
   as light-gray `other` metadata instead of being discarded.
9. Up to three pipe, bullet, or spaced-slash segments per header line are compared
   with positive and negative job-title references; only confident title matches
   are emitted, with their original text unchanged.
10. MiniLM-confirmed section headings route the remaining lines into sections;
    relative font size and boldness distinguish subheadings from paragraph blocks.
11. Pillow draws one large header box and a smaller labeled box per detected item.

Input PDFs belong in `resumes-synthetic/`. Running the existing project entry point writes:

```text
results/
  1/
    sections-debug.json
    debug/
      header/
        header.json
        page-1.png
```

`debug/header/header.json` contains only the header extraction and the first likely
section that stopped it. `sections-debug.json` contains the subsequent experimental
section routing. The header debug image uses a dark outer box for the whole top region, then
labeled colored boxes for `name`, `job_title`, `location`, `email`, `phone`, and
`url` detections. URL entities preserve both their visible `text` and their PDF
annotation destination in `url`; regex-only URLs use their visible text for both.
The same URL extraction is applied to routed section headings and content blocks.
Annotation rectangles are matched with a small bounding-box tolerance, and their
destinations take precedence when the visible text also matches the URL regex.

The first native PyMuPDF read is also written to `debug/<resume>.raw-pymupdf.json`.
These generated dumps are ignored by Git; `debug/.gitkeep` retains the empty directory.
When OCR is actually used, its resulting PyMuPDF dictionary is written separately to
`debug/ocr/<resume>.ocr-pymupdf.json`. Native-text pages do not create OCR dumps.

Tesseract must be installed separately because it is a native program, not a Python package:

```bash
brew install tesseract
```

Run the project with:

```bash
uv run extractor-v1
```

All inference weights are loaded from the gitignored project-local `models/`
directory. The tracked `models/.gitkeep` preserves that directory without
committing model weights:

```text
models/
  all-MiniLM-L6-v2/
  gliner_small-v2.1/
  distilbert-NER/
```

DistilBERT is the default NER backend, so both commands below use it:

```bash
uv run extractor-v1
uv run extractor-v1 --ner-backend distilbert
```

GLiNER is optional. Install its dependency and select it explicitly with:

```bash
uv sync --extra gliner
uv run extractor-v1 --ner-backend gliner
```

Both backends use the same contact masking, line-by-line input, entity cleanup,
and fallback pipeline, so their generated `debug/header/header.json` metadata records which
backend and model revision produced the result.

The extractor does not download models at runtime. Populate the MiniLM and
DistilBERT local model directories for the default setup; the GLiNER directory is
needed only when that optional backend is selected. Both NER backends process
contact-masked header lines independently, while regex remains authoritative for
deterministic contact formats.

Place holdout PDFs inside `resume-truths/`, then run the same pipeline against only those PDFs:

```bash
uv run extractor-v1 --truths
```

The holdout results are kept separate from development results:

```text
resumes-truths/*.pdf
  -> results/0-truths/<resume>/sections-debug.json
  -> results/0-truths/<resume>/debug/header/header.json
  -> results/0-truths/<resume>/raw-pymupdf.json
  -> results/0-truths/<resume>/debug/page-N.png
```

Without `--truths`, only `resumes-synthetic/*.pdf` is processed and written to
`results/<resume>/` as usual.
