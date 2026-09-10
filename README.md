<h1 align="center">Restruct</h1>

<p align="center">
  <a href="https://pypi.org/project/restruct-cv/"><img src="https://img.shields.io/pypi/v/restruct-cv.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/restruct-cv/"><img src="https://img.shields.io/pypi/dm/restruct-cv.svg" alt="PyPI downloads"></a>
  <a href="https://pypi.org/project/restruct-cv/"><img src="https://img.shields.io/pypi/wheel/restruct-cv" alt="PyPI wheel"></a>
</p>
<p align="center">
  <a href="https://github.com/Nye1nChanSoe/restruct-cv/actions/workflows/ci.yml"><img src="https://github.com/Nye1nChanSoe/restruct-cv/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/restruct-cv/"><img src="https://img.shields.io/pypi/pyversions/restruct-cv.svg" alt="Python versions"></a>
  <a href="https://github.com/Nye1nChanSoe/restruct-cv/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/restruct-cv.svg" alt="License"></a>
</p>

<h3 align="center">Extract structured JSON from single-column resumes.</h3>

<p align="center"><strong>PDF (OCR) · DOCX · PNG · JPEG</strong></p>
<p align="center"><i>Runs on your machine. No resume upload and no API key.</i></p>

---

<p align="center">
  <img src="https://raw.githubusercontent.com/Nye1nChanSoe/restruct-cv/master/docs/readme-pipeline.webp" width="900"
       alt="Animated walkthrough: a single-column resume is supported while multi-column and sidebar layouts are not reliable; Restruct reads text or OCR, detects document structure, extracts fields, and writes a stable resume.json file.">
</p>

Restruct is built for **flat, single-column resumes with one top-to-bottom reading order**. It
finds the document's sections and extracts names, contact details, jobs, dates, schools and skills.

The output follows one published JSON schema. A missing section does not become a missing key, and
a layout the parser cannot read reliably is reported instead of silently rearranged.

- **Inputs:** PDF, including scanned PDFs through OCR; DOCX; PNG; and JPEG.
- **Layout scope:** one content column. Sidebars, multiple columns, text inside graphics and nested
  tables are detected as unsupported layouts rather than repaired by guesswork.
- **Local processing:** after the model files are installed, resume extraction makes no network request.
- **Stable output:** sixteen sections in a fixed order, validated against `resume.schema.json`.
- **Inspectable results:** reconstruct the extracted JSON or draw the detected fields on the source page.

<br>

## Why Restruct exists

Restruct grew out of my work on [Open LinkedOut](https://github.com/Nye1nChanSoe/open-linkedout), a lightweight, local-first job scraping and matching system.
Small local models consumed too much RAM and disk space while still hallucinating resume details.
Restruct uses document structure and explicit patterns first. Local models are used only when the
document itself does not settle the meaning of a span.

<br>

## Built with

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&amp;logo=python&amp;logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/DistilBERT-FFD21E?style=flat&amp;logo=huggingface&amp;logoColor=black" alt="DistilBERT">
  <img src="https://img.shields.io/badge/all--MiniLM--L6--v2-FFD21E?style=flat&amp;logo=huggingface&amp;logoColor=black" alt="all-MiniLM-L6-v2">
  <img src="https://img.shields.io/badge/Tesseract-OCR-5A5A5A?style=flat" alt="Tesseract OCR">
</p>
<p>
  <img src="https://img.shields.io/badge/PDF-PyMuPDF-EC1C24?style=flat" alt="PyMuPDF">
  <img src="https://img.shields.io/badge/DOCX-python--docx-2B579A?style=flat" alt="python-docx">
  <img src="https://img.shields.io/badge/Local--First-111111?style=flat" alt="Local First">
</p>

<br>

## Install

```bash
uv add restruct-cv
uv run restruct --install-models
```

<br>

**About those two commands**

- The distribution is **`restruct-cv`**; the command it installs is **`restruct`**. The
  unqualified name was already taken on PyPI by an unrelated project.
- `--install-models` downloads the two models Restruct uses: **352 MB, once**. They are ordinary
  local files, verified by checksum as they arrive.
- It is the **only** command that touches the network. Everything after it works offline.

<br>

**Scanned PDFs, PNG files and JPEG files** contain pixels rather than readable text, so they also
need Tesseract. A native-text PDF or DOCX does not. Each PNG or JPEG is treated as one resume page,
with its original proportions and camera-orientation metadata preserved.

```bash
brew install tesseract                 # macOS
apt-get install tesseract-ocr          # Debian / Ubuntu
```

<br>

## Use it

```bash
uv run restruct resume.pdf -o .
```

That writes `resume.json` next to you, and prints nothing.

**`-o` takes a file or a directory:**

| You write     | You get                                |
| ------------- | -------------------------------------- |
| `-o out.json` | exactly that file                      |
| `-o .`        | `resume.json` in the current directory |
| `-o results/` | `resume.json` inside `results/`        |

A directory gets `<resume>.json`, **named after the input**, so extracting several resumes into
one place doesn't have each one overwrite the last.

<br>

## See what was understood

```bash
uv run restruct resume.pdf -o . --reconstruct
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Nye1nChanSoe/restruct-cv/master/examples/7.anomaly-reconstruction-page-1.png" width="620"
       alt="A resume redrawn from the extracted JSON: name, titles and contact line at the top, then SUMMARY and EXPERIENCE sections with job title, employer, location, dates and bullets.">
</p>

This draws the extracted result **back out as a readable page**, `reconstruction.pdf` plus a PNG
per page, built only from what was extracted, with the original layout thrown away.

That's the whole point:

- A bullet filed under the wrong section is **obvious at a glance** here, and invisible in a wall
  of JSON.
- A date read as a job title shows up in the wrong line of the header.
- Anything Restruct **could not place** is drawn in red under `UNPLACED`, rather than quietly
  dropped.

It is deliberately **not** a facsimile. Imitating the original layout would hide the very errors
it exists to reveal.

You can also draw a result you already have, without re-extracting anything:

```bash
uv run restruct resume.json --reconstruct
```

<br>

## Check it is ATS-friendly

```bash
uv run restruct resume.pdf -o . --ats
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Nye1nChanSoe/restruct-cv/master/examples/7.anomaly/debug/page-1.png" width="620"
       alt="The same resume with coloured boxes drawn over it: name, job titles, phone, email and location in the header, then boxes around each section heading, each skill group and each bullet.">
</p>

Alongside the JSON you get **an overlay of every page**, showing exactly what a parser could read
and where it thought each section began and ended.

- Boxes in the **wrong places, or missing**, are the layouts machines choke on: side-by-side
  columns, text inside a graphic, a table nested in a table.
- Each box is **labelled with the field it became**: `name`, `job_title`, `bullet`, `skill_group`.
- **Heavier boxes are model conclusions.** Lighter ones are things the document stated outright,
  so you can tell a guess from a fact.

Scans and photographs go through the same path. A PNG or JPEG is placed on a page of its own
proportions and read as a scan, and OCR is rebuilt into the same geometry a native PDF produces,
so a photographed page gets an overlay that looks like any other:

<p align="center">
  <img src="https://raw.githubusercontent.com/Nye1nChanSoe/restruct-cv/master/examples/9.ocr/debug/page-1.png" width="620"
       alt="A scanned resume page with the same style of extraction overlay drawn on it.">
</p>

<br>

## What you get back

`resume.json` is **plain data**: no bounding boxes, no model names, no confidence scores.

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

- **Sixteen sections**, always present, always in the same order.
- `null` when the resume has none, `[]` when the section exists but yielded nothing, **never
  absent**.
- [`resume.schema.json`](resume.schema.json) is the published contract, and every release is
  validated against it.
- A section Restruct isn't confident about goes to **`others`, with its original heading kept**,
  rather than being guessed into the wrong place.

Restruct is deliberately careful about what it claims. **v1 targets single-column resumes**, and a
layout whose reading order can't be recovered is _recorded_ as such, never silently repaired into
something that reads plausibly and is wrong.

<br>

## How accuracy is measured

A parser's failures matter more than its average. Restruct is scored **per field**, not with a
single headline number that hides which fields to trust.

```bash
uv run tools/dev.py scorecard
```

This currently reports field-level precision, recall and F1 (name, dates, job titles, ...) and
section-routing accuracy - the percent of content assigned to the right one of 16 fixed sections
versus landing in `others`. Scored against 10 hand-labeled resumes in `tests/labels/`: enough to
catch regressions, not enough to claim broad accuracy yet.

**Next release:** strict exact-match accuracy (no containment leniency) and OCR character/word
error rate, once the labeled corpus is large enough to make those numbers meaningful rather than
noisy.

<br>

### From Python

```python
from restruct import extract_resume
```

Failures are raised as **typed exceptions**, so embedding Restruct in a service doesn't cost you
your process. The command-line tool turns those into exit codes instead, grouped by decade:

| Code | Meaning                                                         |
| ---- | --------------------------------------------------------------- |
| `0`  | success                                                         |
| `1x` | input: not found, unsupported format, unreadable document       |
| `2x` | environment: models missing, Tesseract missing, download failed |
| `3x` | extraction                                                      |
| `4x` | output                                                          |

<br>

## Examples

Three extracted resumes are committed under [`examples/`](examples/), one per kind of input:

| Example                             | Source              | What's in it                                     |
| ----------------------------------- | ------------------- | ------------------------------------------------ |
| [`7.anomaly/`](examples/7.anomaly/) | native PDF, 3 pages | JSON and a page overlay per page                 |
| [`9.ocr/`](examples/9.ocr/)         | scanned PDF         | the same shapes, recovered by OCR                |
| [`11/`](examples/11/)               | DOCX                | JSON only, because a DOCX has no page to draw on |

Worth a look before installing anything.

<br>

## Contributing

Contributions are welcome. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the setup, the test
suite and the accuracy scorecard, and `CLAUDE.md` for the design decisions behind each module.

> ⚠️ Please do **not** submit real resumes, or labels derived from them. The fixtures in
> `resumes-synthetic/` are synthetic and safe to commit.

<br>

## License

[MIT](LICENSE)
