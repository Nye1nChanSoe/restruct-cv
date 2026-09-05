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
pip install dist/restruct_cv-*.whl
```

The distribution is `restruct-cv`; the import name and the command it installs are both
`restruct`. The unqualified name was already taken on PyPI by an unrelated project.

### Model weights

Inference runs on ONNX Runtime, so there is no torch, no `transformers` and no CUDA wheel in the
install. Weights are **local files**, not something an extraction ever fetches for itself; each
model directory holds a `model.onnx` and the tokenizer that goes with it:

```text
models/
  all-MiniLM-L6-v2/    model.onnx, tokenizer.json, tokenizer_config.json, sentence_bert_config.json
  distilbert-NER/      model.onnx, tokenizer.json, tokenizer_config.json, config.json
```

```bash
uv run restruct --install-models             # 352 MB, verified, into the directory below
uv run restruct --install-models /some/dir   # or somewhere named outright
```

That is the only command in restruct that uses the network. Each file is pinned by both its
source revision and its SHA-256, written to a temporary name and moved into place only once the
digest matches — so an install that is interrupted leaves nothing that looks loadable, and
running it again resumes rather than restarts.

An extraction with no weights **asks** rather than downloading: on a terminal it says what is
missing, how large the download is and where it would go, and waits for an answer. A
non-interactive run — a script, a container build, a pipe — is never prompted and exits `20`
with the same message it always gave, because a program that starts a 352 MB transfer nobody is
watching is worse than one that fails with an instruction.

Weights are looked for, in order, in `models/` beside the checkout (when running from one),
`models/` under the current working directory, and `~/.restruct/models`. Set
`RESTRUCT_MODELS_DIRECTORY=/path/to/models` to name the directory outright, which is the usual
answer for an installed copy; when it is set nothing else is consulted. An install writes to the
same place a run looks first, except that an installed copy writes to `~/.restruct/models` rather
than to whatever directory the shell happened to be in.

#### Producing them instead

`--install-models` fetches the fp32 ONNX exports the two source repositories publish themselves.
They were checked against a local export before being trusted: every golden snapshot is
byte-identical and the scorecard's macro F1 is unchanged. To export from the safetensors anyway:

```bash
pip install "huggingface_hub[cli]"
hf download sentence-transformers/all-MiniLM-L6-v2 \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --local-dir models/all-MiniLM-L6-v2
hf download dslim/distilbert-NER \
  --revision dfa2838a127384aabb82ed7719e16dab84c42a2a --local-dir models/distilbert-NER

uv run --group export python tools/export_onnx.py
```

The export needs torch, transformers and optimum; they are in the `export` dependency group and
are installed only by that command, never by an ordinary install. The exported weights are fp32,
which reproduces the torch pipeline exactly — `tools/export_onnx.py` records what int8
quantization was measured to cost on this corpus and why it does not ship.

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
uv run restruct examples/11/resume.json --reconstruct   # draw a result already extracted

uv run restruct                               # batch over resumes-synthetic/
uv run restruct --truths                      # batch over resumes-truths/ (local, gitignored)
uv run restruct --unsupported                 # batch over resumes-unsupported/

uv run restruct --install-models              # download the weights and exit
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

Drawing a result on its own writes those files flat beside the JSON, named after it
(`resume-reconstruction.pdf`, `resume-page-1.png`); `-o` names a directory to put them in
instead, under their plain names.

It answers a different question from the overlays. An overlay draws on top of the document, so it
shows whether a box landed on the right words; the document keeps making sense regardless of what
was understood. A reconstruction throws the page away and draws only what was understood, which is
what makes a bullet filed under education or a date read as a job title visible at a glance.

It is deliberately not a facsimile — imitating the original layout would hide the errors it exists
to reveal. Absent and empty fields are skipped, so what is on the page is what was extracted, and
anything the renderer cannot place is drawn in red under UNPLACED rather than dropped. Given a
`resume.json` as the input it draws that and runs nothing else, loading no models.

### Worked examples

`results/` is regenerated by every run and is not committed. Three resumes are, under
`examples/` — one per ingestion track, because what each track makes available is different:

| Example | Source | What it shows |
| --- | --- | --- |
| [`examples/7.anomaly/`](examples/7.anomaly/) | native PDF, 3 pages | The full evidence track. `debug/page-N.png` draws every section box on the page it came from, model-backed boxes drawn more heavily than deterministic ones. `raw/layout-warnings.json` is written even when empty, so an absent finding is distinguishable from an absent check. |
| [`examples/9.ocr/`](examples/9.ocr/) | scanned PDF | The same shapes from a page with no text layer. OCR is rebuilt into the line geometry the native path produces, so nothing downstream has an OCR case — which is visible here as an overlay that looks like any other. |
| [`examples/11/`](examples/11/) | DOCX | No overlays at all. A DOCX has no geometry and must never pretend to, so there is nothing to draw on — the structure comes from what the document states outright: styles, list markers, table cells. |

Each holds the `resume.json`, the `raw/*.json` evidence behind it, and the overlays where there
are any. They are copied out of `results/` by `tools/refresh_examples.py` rather than curated by
hand, so they are what the pipeline currently produces and not a snapshot of what it once did.
Their inputs are in `resumes-synthetic/`, and every one is synthetic.

## Architecture

Five ordered passes over one shared in-memory document.

```text
ingestion/   physical extraction — native PDF text, per-page OCR fallback, DOCX
document/    shared types and document-wide statistics
layout/      row clustering, paragraph/bullet accumulation, unsupported-layout detection
structure/   heading detection, routing, compound headings, precedence resolver, separators
parsers/     one module per section shape (header, experience, education, skills, grouped, urls)
model.py     the model-backed extraction stages, loaded on first use
encoders.py  ONNX Runtime adapters: MiniLM sentence embeddings, DistilBERT NER
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
uv run restruct && uv run python tools/refresh_examples.py
git status --short examples/    # clean == byte-identical
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the test and scorecard workflow, and `CLAUDE.md` for the
design decisions behind each module.

Please do not submit **real** resumes or their labels; `resumes-truths/` is gitignored for exactly
that reason. Fixtures in `resumes-synthetic/` are synthetic and safe to commit.
