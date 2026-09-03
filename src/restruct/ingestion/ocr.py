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
import statistics
import subprocess
from pathlib import Path
from typing import Any

import pymupdf

from restruct.configs import SETTINGS
from restruct.errors import OcrFailed, TesseractMissing
from restruct.document.physical import Span, TextLine, Token
from restruct.geometry import union


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
    page_number = page_index + 1
    image_path = temporary_directory / f"page-{page_number}.png"
    pixmap = page.get_pixmap(dpi=SETTINGS.ocr.dpi, alpha=False)
    pixmap.save(image_path)
    x_scale = page.rect.width / pixmap.width
    y_scale = page.rect.height / pixmap.height

    result = _run_ocr_command(
        [
            SETTINGS.ocr.tesseract_command,
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
