"""Tesseract OCR for pages with no usable native text.

PyMuPDF renders the page and Tesseract is invoked directly, without a shell.
Its TSV output is rebuilt into the same line geometry the native path produces,
so nothing downstream needs OCR-specific handling.
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
        raise RuntimeError(
            f"required OCR program is not installed: {program_name}"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "unknown error"
        raise RuntimeError(f"{program_name} failed: {detail}") from error

def _tesseract_page_dict(
    page: pymupdf.Page,
    page_index: int,
    temporary_directory: Path,
) -> dict[str, Any]:
    """Render one PDF page with PyMuPDF and rebuild line geometry from TSV."""
    page_number = page_index + 1
    image_path = temporary_directory / f"page-{page_number}.png"
    pixmap = page.get_pixmap(dpi=SETTINGS.ocr.dpi, alpha=False)
    pixmap.save(image_path)
    pixel_width, pixel_height = pixmap.width, pixmap.height
    x_scale = page.rect.width / pixel_width
    y_scale = page.rect.height / pixel_height

    tesseract_result = _run_ocr_command(
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
        io.StringIO(tesseract_result.stdout),
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
    )
    for row in reader:
        text = str(row.get("text") or "").strip()
        if row.get("level") != "5" or not text:
            continue
        left = int(row["left"])
        top = int(row["top"])
        width = int(row["width"])
        height = int(row["height"])
        word = {
            "text": text,
            "bbox": (
                left * x_scale,
                top * y_scale,
                (left + width) * x_scale,
                (top + height) * y_scale,
            ),
            "confidence": float(row.get("conf") or -1.0),
            "size": height * y_scale,
        }
        line_key = (
            int(row["block_num"]),
            int(row["par_num"]),
            int(row["line_num"]),
        )
        words_by_line.setdefault(line_key, []).append(word)

    lines_by_block: dict[int, list[dict[str, Any]]] = {}
    for (block_number, _, _), words in words_by_line.items():
        rectangle = union(word["bbox"] for word in words)
        line_text = " ".join(str(word["text"]) for word in words)
        confidence_values = [
            float(word["confidence"])
            for word in words
            if float(word["confidence"]) >= 0
        ]
        raw_line = {
            "bbox": tuple(float(value) for value in rectangle),
            "spans": [
                {
                    "text": line_text,
                    "bbox": tuple(float(value) for value in rectangle),
                    "size": statistics.median(
                        float(word["size"]) for word in words
                    ),
                    "font": "TesseractOCR",
                    "flags": 0,
                }
            ],
            "ocrConfidence": (
                sum(confidence_values) / len(confidence_values)
                if confidence_values
                else None
            ),
        }
        lines_by_block.setdefault(block_number, []).append(raw_line)

    blocks: list[dict[str, Any]] = []
    for block_number, block_lines in lines_by_block.items():
        block_rectangle = union(raw_line["bbox"] for raw_line in block_lines)
        blocks.append(
            {
                "type": 0,
                "number": block_number,
                "bbox": tuple(float(value) for value in block_rectangle),
                "lines": block_lines,
            }
        )
    return {
        "page": page_number,
        "width": page.rect.width,
        "height": page.rect.height,
        "engine": "tesseract_cli",
        "renderer": "pymupdf_pixmap",
        "dpi": SETTINGS.ocr.dpi,
        "pageSegmentationMode": SETTINGS.ocr.page_segmentation_mode,
        "blocks": blocks,
    }
