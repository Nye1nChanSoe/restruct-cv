"""Shared helpers for running the pipeline against a fixture resume."""

from __future__ import annotations

import json
import shutil
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
OCR_STEMS = frozenset({"5.ocr", "9.ocr"})


def synthetic_stems() -> list[str]:
    """Every synthetic fixture stem, in a stable order."""
    return sorted(path.stem for path in SYNTHETIC_DIRECTORY.glob("*.pdf"))


def models_available() -> bool:
    """Both local model directories are present and non-empty."""
    return all(
        (PROJECT_ROOT / "models" / name).is_dir()
        and any((PROJECT_ROOT / "models" / name).iterdir())
        for name in ("all-MiniLM-L6-v2", "distilbert-NER")
    )


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def run_pipeline(pdf_path: Path, workspace: Path, models: Any) -> dict[str, Any]:
    """Extract one resume into a throwaway workspace and return its clean JSON.

    Debug artifacts are redirected into the workspace so a test run never
    touches the repository's own ``debug/`` or ``results/`` directories.
    """
    from extractor_v1 import extract_resume

    output_directory = workspace / pdf_path.stem
    extract_resume(
        pdf_path,
        output_directory,
        workspace / "debug" / f"{pdf_path.stem}.raw-pymupdf.json",
        workspace / "debug" / "ocr" / f"{pdf_path.stem}.ocr-tesseract.json",
        models.embedding,
        models.ner,
    )
    return json.loads((output_directory / "resume.json").read_text(encoding="utf-8"))


def golden_path(stem: str) -> Path:
    return GOLDEN_DIRECTORY / f"{stem}.resume.json"


def label_path(stem: str) -> Path:
    return LABEL_DIRECTORY / f"{stem}.json"


def dump_json(value: Any) -> str:
    """Serialize exactly the way the pipeline writes ``resume.json``."""
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"
