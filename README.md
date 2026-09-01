# extractor-v1

A deliberately small experiment:

1. PyMuPDF extracts native PDF lines and geometry.
2. Pages without meaningful native text fall back to PyMuPDF's Tesseract OCR integration.
3. `all-MiniLM-L6-v2` classifies only short, visually prominent heading candidates.
4. Every line after an accepted heading is copied until the next accepted heading.
5. Pillow draws the heading and its content region using the same section color.

Input PDFs belong in `resumes-synthetic/`. Running the existing project entry point writes:

```text
results/
  1/
    sections.json
    debug/
      page-1.png
```

`sections.json` is the result to inspect. The images contain only heading and content boxes.

Tesseract must be installed separately because it is a native program, not a Python package:

```bash
brew install tesseract
```

Run the project with:

```bash
uv run extractor-v1
```

The first MiniLM load downloads its pinned model revision if it is not already cached.
