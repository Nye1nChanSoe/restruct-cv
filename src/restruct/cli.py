"""Command-line entry point.

Kept separate from the pipeline so filesystem layout, terminal output and
process status stay out of the engine. This is the only module that turns a
``RestructError`` into an exit code; the Python API raises instead, so a caller
embedding restruct catches an exception by type rather than losing its process.

Quiet on success. A tool that prints nothing when it worked is one you can put
in a pipe.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pymupdf

from restruct.configs import SETTINGS
from restruct.errors import (
    ExtractionFailed,
    InputNotFound,
    InvalidDocument,
    ModelAssetsMissing,
    OcrFailed,
    OutputWriteFailed,
    RestructError,
    TesseractMissing,
    UnsupportedFormat,
)
from restruct.pipeline import ALL_STAGES, DEFAULT_DEBUG_STAGES, extract_resume

SUPPORTED_SUFFIXES = (".pdf",)

# One code per failure a caller might handle differently. Grouped by decade so
# a new member of a family does not disturb the others: 1x input, 2x
# environment, 3x extraction, 4x output.
EXIT_OK = 0
EXIT_UNEXPECTED = 1
# 2 is argparse's own usage error and is left to it.
EXIT_INPUT_NOT_FOUND = 10
EXIT_UNSUPPORTED_FORMAT = 11
EXIT_INVALID_DOCUMENT = 12
EXIT_MODEL_ASSETS_MISSING = 20
EXIT_TESSERACT_MISSING = 21
EXIT_OCR_FAILED = 22
EXIT_EXTRACTION_FAILED = 30
EXIT_OUTPUT_WRITE_FAILED = 40

_EXIT_CODES: tuple[tuple[type[RestructError], int], ...] = (
    (InputNotFound, EXIT_INPUT_NOT_FOUND),
    (UnsupportedFormat, EXIT_UNSUPPORTED_FORMAT),
    (InvalidDocument, EXIT_INVALID_DOCUMENT),
    (ModelAssetsMissing, EXIT_MODEL_ASSETS_MISSING),
    (TesseractMissing, EXIT_TESSERACT_MISSING),
    (OcrFailed, EXIT_OCR_FAILED),
    (ExtractionFailed, EXIT_EXTRACTION_FAILED),
    (OutputWriteFailed, EXIT_OUTPUT_WRITE_FAILED),
)

_STAGE_RANGE_RE = re.compile(r"^(?P<first>[1-5])(?:-(?P<last>[1-5]))?$")


def parse_stages(value: str) -> frozenset[int]:
    """Read ``1-5``, ``3``, ``2,4,5`` or ``1-3,5`` into a set of stages."""
    stages: set[int] = set()
    for part in value.split(","):
        match = _STAGE_RANGE_RE.match(part.strip())
        if match is None:
            raise argparse.ArgumentTypeError(
                f"invalid stage selection {part.strip()!r}: "
                "use a stage (3), a range (1-3), or a list (2,4,5)"
            )
        first = int(match.group("first"))
        last = int(match.group("last") or first)
        if last < first:
            raise argparse.ArgumentTypeError(
                f"invalid stage range {part.strip()!r}: {last} is before {first}"
            )
        stages.update(range(first, last + 1))
    return frozenset(stages)


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="restruct",
        description="Extract a structured resume from a PDF.",
        epilog=(
            "With no PATH, every PDF in resumes-synthetic/ is extracted into "
            "results/ with all stages written, which is how the committed "
            "corpus is regenerated."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        metavar="PATH",
        help="The resume to extract. Omit to run the batch over a directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help=(
            "Where to write the resume JSON. Debug artifacts, if any, go in a "
            "directory beside it named after the file: '-o out.json' writes "
            "out/raw/ and out/debug/."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write debug artifacts for stages 4 and 5.",
    )
    parser.add_argument(
        "--stages",
        type=parse_stages,
        metavar="SPEC",
        help=(
            "Which stages' debug artifacts to write: 1-5, 3, 2,4,5, 1-3,5. "
            "Implies --debug. Selects artifacts only -- every pass always "
            "runs, because each one feeds the next."
        ),
    )
    parser.add_argument(
        "--truths",
        action="store_true",
        help="Batch over resumes-truths/ into results/0-truths/.",
    )
    parser.add_argument(
        "--unsupported",
        action="store_true",
        help=(
            "Batch over resumes-unsupported/ into results/1-unsupported/. "
            "Those parses are untrustworthy by definition; this is for reading "
            "the overlays that show why."
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.path is not None and arguments.output is None:
        parser.error("-o/--output is required when a PATH is given")
    return arguments


def _selected_stages(arguments: argparse.Namespace) -> frozenset[int]:
    """--stages implies --debug; --debug alone means stages 4 and 5."""
    if arguments.stages is not None:
        return arguments.stages
    return DEFAULT_DEBUG_STAGES if arguments.debug else frozenset()


def _validate(path: Path) -> None:
    """Check what can be checked before loading several hundred MB of models."""
    if not path.exists():
        raise InputNotFound(path)
    if path.suffix.casefold() not in SUPPORTED_SUFFIXES:
        raise UnsupportedFormat(path, SUPPORTED_SUFFIXES)
    try:
        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise InvalidDocument(path, "the document is password-protected")
            if document.page_count == 0:
                raise InvalidDocument(path, "the document has no pages")
    except InvalidDocument:
        raise
    except Exception as error:  # pymupdf raises several unrelated types
        raise InvalidDocument(path, str(error)) from error


def _silence_model_progress() -> None:
    """Stop the model libraries printing to the terminal.

    Loading weights draws a progress bar, which makes a successful run noisy
    and unusable in a pipe. This is presentation, so it lives here rather than
    in the engine, where a library caller may well want it left alone.
    """
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()
    except Exception:  # a version without the helper must not break the run
        pass


def _load_models(project_root: Path):
    from restruct.model import load_embedding_model, load_ner_model

    _silence_model_progress()
    for name in ("all-MiniLM-L6-v2", "distilbert-NER"):
        directory = project_root / "models" / name
        if not directory.is_dir() or not any(directory.iterdir()):
            raise ModelAssetsMissing(directory)
    return load_embedding_model(project_root), load_ner_model(project_root)


def _extract_one(
    pdf_path: Path,
    output_path: Path,
    stages: frozenset[int],
    models,
) -> None:
    """Extract one resume to an explicit output file."""
    artifact_directory = output_path.with_suffix("")
    try:
        extract_resume(
            pdf_path,
            artifact_directory,
            artifact_directory / "raw" / "pymupdf.json",
            artifact_directory / "raw" / "tesseract.json",
            models[0],
            models[1],
            stages=stages,
        )
    except RestructError:
        raise
    except Exception as error:
        raise ExtractionFailed(pdf_path, error) from error

    # -o names the result; the artifact directory holds artifacts. Leaving a
    # second copy of resume.json inside it would give two files that can drift.
    produced = artifact_directory / "resume.json"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(produced.read_text(encoding="utf-8"), encoding="utf-8")
        produced.unlink(missing_ok=True)
        if artifact_directory.is_dir() and not any(artifact_directory.iterdir()):
            artifact_directory.rmdir()
    except OSError as error:
        raise OutputWriteFailed(output_path, str(error)) from error


def _batch(input_directory: Path, output_root: Path, models, stages) -> None:
    """Regenerate a whole corpus. Writes every stage, which is its purpose."""
    project_root = Path(__file__).resolve().parents[2]
    raw_debug_directory = project_root / SETTINGS.debug.raw_extraction_directory
    ocr_debug_directory = project_root / SETTINGS.debug.ocr_extraction_directory
    local = output_root != project_root / SETTINGS.paths.results_directory

    for pdf_path in sorted(input_directory.glob("*.pdf")):
        resume_output = output_root / pdf_path.stem
        extract_resume(
            pdf_path,
            resume_output,
            (
                resume_output / "raw-pymupdf.json"
                if local
                else raw_debug_directory / f"{pdf_path.stem}.raw-pymupdf.json"
            ),
            (
                resume_output / "debug" / "ocr" / "raw-tesseract.json"
                if local
                else ocr_debug_directory / f"{pdf_path.stem}.ocr-tesseract.json"
            ),
            models[0],
            models[1],
            stages=stages,
        )
        print(f"extracted: {pdf_path.name}")


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    project_root = Path(__file__).resolve().parents[2]
    stages = _selected_stages(arguments)

    try:
        if arguments.path is not None:
            _validate(arguments.path)
            models = _load_models(project_root)
            _extract_one(arguments.path, arguments.output, stages, models)
            return EXIT_OK

        if arguments.truths:
            input_directory = project_root / SETTINGS.paths.truths_input_directory
            output_root = project_root / SETTINGS.paths.truths_results_directory
        elif arguments.unsupported:
            input_directory = project_root / SETTINGS.paths.unsupported_input_directory
            output_root = project_root / SETTINGS.paths.unsupported_results_directory
        else:
            input_directory = project_root / SETTINGS.paths.input_directory
            output_root = project_root / SETTINGS.paths.results_directory
        models = _load_models(project_root)
        # The batch exists to regenerate the committed corpus, so it writes
        # everything unless told otherwise. Anything less and a stale artifact
        # would survive a run and make `git status results/` read as clean.
        _batch(
            input_directory,
            output_root,
            models,
            arguments.stages if arguments.stages is not None else ALL_STAGES,
        )
        return EXIT_OK
    except RestructError as error:
        print(f"restruct: {error}", file=sys.stderr)
        for error_type, code in _EXIT_CODES:
            if isinstance(error, error_type):
                return code
        return EXIT_UNEXPECTED


