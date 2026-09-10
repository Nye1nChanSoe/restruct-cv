"""Shared helpers for running the pipeline against a fixture resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIRECTORY = PROJECT_ROOT / "resumes-synthetic"
UNSUPPORTED_DIRECTORY = PROJECT_ROOT / "resumes-unsupported"
GOLDEN_DIRECTORY = Path(__file__).resolve().parent / "golden"
LABEL_DIRECTORY = Path(__file__).resolve().parent / "labels"

# Real CVs and their labels stay local; the directory is gitignored.
TRUTHS_DIRECTORY = PROJECT_ROOT / "resumes-truths"
TRUTHS_LABEL_DIRECTORY = TRUTHS_DIRECTORY / "labels"

# Fixtures whose pages carry no native text, so they need Tesseract.
# Fixtures that need Tesseract: the two scanned PDFs, and the image, which has
# no reader but OCR at all.
OCR_STEMS = frozenset({"5.ocr", "9.ocr", "12.ats"})


# Every format the fixtures may be written in. A stem is unique across them,
# so a fixture can be replaced with the same resume in another format without
# renaming its golden file or its labels.
FIXTURE_SUFFIXES = (".pdf", ".docx", ".png")


def synthetic_stems() -> list[str]:
    """Every synthetic fixture stem, in a stable order."""
    return sorted(
        path.stem
        for suffix in FIXTURE_SUFFIXES
        for path in SYNTHETIC_DIRECTORY.glob(f"*{suffix}")
    )


def fixture_path(stem: str, directory: Path = SYNTHETIC_DIRECTORY) -> Path:
    """The fixture file for a stem, whichever format it is written in."""
    for suffix in FIXTURE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no fixture named {stem} in {directory}")


def models_available() -> bool:
    """Both local model directories hold exported ONNX weights.

    Asked the way the CLI asks it: a directory that still holds only the
    safetensors it was exported from cannot be run, so a machine with one
    skips the model-backed tests rather than failing them.
    """
    return all(
        (PROJECT_ROOT / "models" / name / "model.onnx").is_file()
        for name in ("all-MiniLM-L6-v2", "distilbert-NER")
    )


def tesseract_available() -> bool:
    """Asked the same way the pipeline asks it.

    A plain PATH lookup would skip the OCR fixtures on a machine where the
    binary is installed somewhere the pipeline looks and PATH does not, which
    is the ordinary Windows install.
    """
    from restruct.ingestion.ocr import find_tesseract

    return find_tesseract() is not None


def run_pipeline(pdf_path: Path, workspace: Path, models: Any) -> dict[str, Any]:
    """Extract one resume into a throwaway workspace and return its clean JSON.

    Debug artifacts are redirected into the workspace so a test run never
    touches the repository's own ``debug/`` or ``results/`` directories.
    """
    from restruct import extract_resume
    from restruct.stages import raw_extraction_reader

    output_directory = workspace / pdf_path.stem
    extract_resume(
        pdf_path,
        output_directory,
        workspace
        / "debug"
        / f"{pdf_path.stem}.raw-{raw_extraction_reader(pdf_path)}.json",
        workspace / "debug" / "ocr" / f"{pdf_path.stem}.ocr-tesseract.json",
        models.embedding,
        models.ner,
    )
    return json.loads((output_directory / "resume.json").read_text(encoding="utf-8"))


def golden_path(stem: str) -> Path:
    return GOLDEN_DIRECTORY / f"{stem}.resume.json"


# The Tesseract build that produced the committed OCR snapshots. Written by
# `--update-golden`, read by the snapshot test.
GOLDEN_TESSERACT_VERSION_PATH = GOLDEN_DIRECTORY / "tesseract-version.txt"


def tesseract_version() -> str | None:
    """The installed engine's version string, or None if it is not installed.

    Asked through `find_tesseract` so it names the same binary the pipeline
    would run, not whichever one PATH happens to expose.
    """
    import subprocess

    from restruct.ingestion.ocr import find_tesseract

    executable = find_tesseract()
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # "tesseract 5.5.3\n leptonica-1.87.0 ..." -- the first line is the engine.
    return completed.stdout.splitlines()[0].strip() if completed.stdout else None


def label_path(stem: str) -> Path:
    return LABEL_DIRECTORY / f"{stem}.json"


def dump_json(value: Any) -> str:
    """Serialize exactly the way the pipeline writes ``resume.json``."""
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"
