"""Tesseract OCR for pages with no usable native text.

PyMuPDF renders the page and Tesseract is invoked directly, without a shell.
The TSV output is rebuilt into the same physical types the native path produces,
so nothing downstream needs OCR-specific handling.

The one honest difference is granularity: Tesseract reports word boxes, not
character boxes, so an OCR span carries ``granularity="word"``. Word
reconstruction reads that flag instead of guessing.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import pymupdf

from restruct.configs import SETTINGS
from restruct.errors import OcrFailed, TesseractMissing
from restruct.document.physical import Span, TextLine, Token
from restruct.geometry import union


def _known_install_locations() -> tuple[str, ...]:
    """Where an installer puts Tesseract when it does not put it on PATH.

    The Windows installer the error message names is the common case: it
    installs into Program Files and leaves PATH alone, so ``tesseract`` is
    absent from a fresh shell while the binary sits right there. Homebrew's two
    prefixes are here for the same reason -- a GUI-launched process inherits a
    PATH that often has neither. Nothing here installs anything; it only looks.
    """
    if sys.platform == "win32":
        directories = [
            directory
            for directory in (
                os.environ.get("ProgramFiles"),
                os.environ.get("ProgramFiles(x86)"),
            )
            if directory
        ]
        local_applications = os.environ.get("LOCALAPPDATA")
        if local_applications:
            directories.append(os.path.join(local_applications, "Programs"))
        return tuple(
            os.path.join(directory, "Tesseract-OCR", "tesseract.exe")
            for directory in directories
        )
    if sys.platform == "darwin":
        return ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract")
    return ("/usr/bin/tesseract", "/usr/local/bin/tesseract")


def find_tesseract() -> str | None:
    """The Tesseract binary to run, or None when it cannot be found.

    Deliberately not called up front. A native PDF and a DOCX never render a
    page, and requiring an OCR engine to read a document that needs none would
    turn an optional dependency into a mandatory one. This is asked once per
    page that actually has too little text to parse, and ``shutil.which`` is
    far cheaper than the render it precedes.
    """
    configured = SETTINGS.ocr.tesseract_command
    found = shutil.which(configured)
    if found:
        return found
    # A path in the settings that PATH lookup cannot resolve.
    if os.path.isabs(configured) and os.access(configured, os.X_OK):
        return configured
    for candidate in _known_install_locations():
        if os.access(candidate, os.X_OK):
            return candidate
    return None


def _run_ocr_command(arguments: list[str], program_name: str) -> subprocess.CompletedProcess[str]:
    """Run one required native OCR command with a focused error message."""
    try:
        return subprocess.run(
            arguments,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise TesseractMissing(program_name) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "unknown error"
        raise OcrFailed(f"{program_name}: {detail}") from error


def _tesseract_words(
    page: pymupdf.Page,
    page_index: int,
    temporary_directory: Path,
) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    """Render the page and return recognised words grouped by Tesseract line.

    Word boxes are scaled from render pixels back into PDF space, so every
    coordinate downstream is in the same units as the native path.
    """
    # Before the render, not after: a missing engine is not worth several
    # hundred milliseconds of rasterising a page nothing will read.
    executable = find_tesseract()
    if executable is None:
        raise TesseractMissing(SETTINGS.ocr.tesseract_command)

    page_number = page_index + 1
    image_path = temporary_directory / f"page-{page_number}.png"
    pixmap = page.get_pixmap(dpi=SETTINGS.ocr.dpi, alpha=False)
    pixmap.save(image_path)
    x_scale = page.rect.width / pixmap.width
    y_scale = page.rect.height / pixmap.height

    result = _run_ocr_command(
        [
            executable,
            str(image_path),
            "stdout",
            "-l",
            SETTINGS.ocr.language,
            "--oem",
            str(SETTINGS.ocr.engine_mode),
            "--psm",
            str(SETTINGS.ocr.page_segmentation_mode),
            "tsv",
        ],
        SETTINGS.ocr.tesseract_command,
    )

    words_by_line: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    reader = csv.DictReader(
        io.StringIO(result.stdout),
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
    )
    for row in reader:
        text = str(row.get("text") or "").strip()
        # Level 5 is a word; anything coarser would double-count.
        if row.get("level") != "5" or not text:
            continue
        left, top = int(row["left"]), int(row["top"])
        width, height = int(row["width"]), int(row["height"])
        line_key = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
        words_by_line.setdefault(line_key, []).append(
            {
                "text": text,
                "bbox": (
                    left * x_scale,
                    top * y_scale,
                    (left + width) * x_scale,
                    (top + height) * y_scale,
                ),
                "confidence": float(row.get("conf") or -1.0),
                # Tesseract gives no font size; glyph height is the best proxy.
                "size": height * y_scale,
            }
        )
    return words_by_line


def ocr_page(
    page: pymupdf.Page,
    page_index: int,
    temporary_directory: Path,
) -> tuple[tuple[TextLine, ...], dict[str, Any]]:
    """OCR one page into physical lines, plus the raw dump for debugging."""
    page_number = page_index + 1
    words_by_line = _tesseract_words(page, page_index, temporary_directory)

    lines_by_block: dict[int, list[tuple[TextLine, dict[str, Any]]]] = {}
    for (block_number, _, _), words in words_by_line.items():
        box = tuple(float(value) for value in union(word["bbox"] for word in words))
        text = " ".join(str(word["text"]) for word in words)
        confidences = [
            float(word["confidence"]) for word in words if float(word["confidence"]) >= 0
        ]
        line_confidence = sum(confidences) / len(confidences) if confidences else None

        # One span per line, sized by the median word height: a single tall
        # glyph must not make the whole line read as a heading.
        span = Span(
            text=text,
            bbox=box,
            font="TesseractOCR",
            size=statistics.median(float(word["size"]) for word in words),
            flags=0,
            tokens=tuple(
                Token(
                    text=str(word["text"]),
                    bbox=tuple(float(value) for value in word["bbox"]),
                    confidence=float(word["confidence"]),
                )
                for word in words
            ),
            granularity="word",
            confidence=line_confidence,
        )
        line = TextLine(page=page_number, bbox=box, spans=(span,), used_ocr=True)
        raw_line = {
            "bbox": box,
            "spans": [
                {
                    "text": span.text,
                    "bbox": box,
                    "size": span.size,
                    "font": span.font,
                    "flags": span.flags,
                }
            ],
            "ocrConfidence": line_confidence,
        }
        lines_by_block.setdefault(block_number, []).append((line, raw_line))

    lines: list[TextLine] = []
    blocks: list[dict[str, Any]] = []
    for block_number, entries in lines_by_block.items():
        lines.extend(line for line, _ in entries)
        raw_lines = [raw_line for _, raw_line in entries]
        blocks.append(
            {
                "type": 0,
                "number": block_number,
                "bbox": tuple(
                    float(value) for value in union(raw["bbox"] for raw in raw_lines)
                ),
                "lines": raw_lines,
            }
        )

    return tuple(lines), {
        "page": page_number,
        "width": page.rect.width,
        "height": page.rect.height,
        "engine": "tesseract_cli",
        "renderer": "pymupdf_pixmap",
        "dpi": SETTINGS.ocr.dpi,
        "pageSegmentationMode": SETTINGS.ocr.page_segmentation_mode,
        "blocks": blocks,
    }
