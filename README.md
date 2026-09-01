# extractor-v1

A deliberately small experiment:

1. PyMuPDF extracts native PDF lines and geometry.
2. Pages without meaningful native text fall back to PyMuPDF's Tesseract OCR integration.
3. `all-MiniLM-L6-v2` locates the first likely resume section heading.
4. Everything above that boundary becomes one top-level `headerProfile` region.
5. Contact regex claims and masks email, phone, and URL spans first.
6. `urchade/gliner_small-v2.1` processes each remaining header line independently
   for names, locations, and contextual nationalities.
7. MiniLM and exact phrase matching classify split job-title candidates.
8. Geometry fills missing identity fields, and unmatched visible text is retained
   as light-gray `other` metadata instead of being discarded.
9. Up to three pipe, bullet, or spaced-slash segments per header line are compared
   with positive and negative job-title references; only confident title matches
   are emitted, with their original text unchanged.
10. Pillow draws one large header box and a smaller labeled box per detected item.

Input PDFs belong in `resumes-synthetic/`. Running the existing project entry point writes:

```text
results/
  1/
    header.json
    debug/
      header/
        page-1.png
```

`header.json` contains the extracted top region and the first likely section that
stopped it. Its debug image uses a dark outer box for the whole top region, then
labeled colored boxes for `name`, `job_title`, `location`, `email`, `phone`, and
`url` detections. This stage deliberately does not reconstruct later sections.

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

The first run downloads MiniLM and GLiNER Small if they are not already cached.
GLiNER processes contact-masked header lines independently with resume-specific
entity labels; regex remains authoritative for deterministic contact formats.

Place holdout PDFs inside `resume-truths/`, then run the same pipeline against only those PDFs:

```bash
uv run extractor-v1 --truths
```

The holdout results are kept separate from development results:

```text
resumes-truths/*.pdf
  -> results/0-truths/<resume>/header.json
  -> results/0-truths/<resume>/raw-pymupdf.json
  -> results/0-truths/<resume>/debug/page-N.png
```

Without `--truths`, only `resumes-synthetic/*.pdf` is processed and written to
`results/<resume>/` as usual.
