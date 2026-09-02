# extractor-v1

The source is split by responsibility:

- `model.py` owns DistilBERT and MiniLM inference
- `routing.py` owns section/paragraph routing and summary debug output,
- `__init__.py` orchestrates PDF/OCR and header/contact extraction.

---

Input PDFs belong in `resumes-synthetic/`. Running the existing project entry point writes:

- `.json` files contain the extracted structured data,
- `page-N.png` files provide visually debuggable bounding boxes (bboxes).

```text
results/
  <resume-name>/
    sections-debug.json
    debug/
      header/
        header.json
        page-1.png
      summary/
        summary.json
        page-1.png
```

Tesseract must be installed

```bash
brew install tesseract
```

PyMuPDF renders each OCR page at 300 DPI, then Python invokes `tesseract` directly
without a shell. OCR uses the installed English language data, LSTM engine mode 1,
and page segmentation mode 3. Temporary page PNGs are deleted automatically after
each PDF.

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
  distilbert-NER/
```

Place holdout PDFs inside `resume-truths/`, then run the same pipeline against only those PDFs:

```bash
uv run extractor-v1 --truths
```
