# Restruct

Restruct turns messy PDF, DOCX, and scanned resumes into consistent, explainable, versioned JSON.
It combines OCR, layout geometry, NER, semantic similarity, and deterministic rules so every result can be traced to its source.

Built for use in applicant tracking systems, candidate search, analytics pipelines, talent platforms, and other systems that need trustworthy resume data.

---

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
      experience/
        experience.json
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

### Contribution

Contributions are warmly welcome.

You can contribute by:

- fixing bugs or improving extraction reliability,
- adding support for Burmese/English mixed-language resumes,
- improving recognition of Myanmar-specific names, phone numbers, addresses, universities, job titles, and date formats,
- contributing anonymized or synthetic Myanmar-style resume test cases,
- improving OCR (`tesseract`) handling for scanned resumes,
- testing the extractor against different resume formats and reporting edge cases,
- improving documentation or developer setup instructions.

Please avoid submitting **real** resumes containing PII data. Use anonymized data whenever possible.

Fianlly: _For major changes, please open an issue first to discuss your idea._
