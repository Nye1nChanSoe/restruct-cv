"""Command-line entry point.

Kept separate from the pipeline so filesystem layout and terminal behavior stay
out of the engine, and the same extraction is reachable from Python.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from restruct.configs import SETTINGS
from restruct.model import load_embedding_model, load_ner_model
from restruct.pipeline import extract_resume


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the top resume profile region.")
    parser.add_argument(
        "--truths",
        action="store_true",
        help="Process only PDFs in resume-truths/ and write them under results/0-truths/.",
    )
    return parser.parse_args()

def main() -> None:
    arguments = _parse_arguments()
    project_root = Path(__file__).resolve().parents[2]
    if arguments.truths:
        input_directory = project_root / SETTINGS.paths.truths_input_directory
        output_root = project_root / SETTINGS.paths.truths_results_directory
    else:
        input_directory = project_root / SETTINGS.paths.input_directory
        output_root = project_root / SETTINGS.paths.results_directory
    raw_debug_directory = project_root / SETTINGS.debug.raw_extraction_directory
    ocr_debug_directory = project_root / SETTINGS.debug.ocr_extraction_directory
    model = load_embedding_model(project_root)
    ner_model = load_ner_model(project_root)

    for pdf_path in sorted(input_directory.glob("*.pdf")):
        resume_output = output_root / pdf_path.stem
        raw_debug_path = (
            resume_output / "raw-pymupdf.json"
            if arguments.truths
            else raw_debug_directory / f"{pdf_path.stem}.raw-pymupdf.json"
        )
        ocr_debug_path = (
            resume_output / "debug" / "ocr" / "raw-tesseract.json"
            if arguments.truths
            else ocr_debug_directory / f"{pdf_path.stem}.ocr-tesseract.json"
        )
        extract_resume(
            pdf_path,
            resume_output,
            raw_debug_path,
            ocr_debug_path,
            model,
            ner_model,
        )
        print(f"extracted: {pdf_path.name}")
