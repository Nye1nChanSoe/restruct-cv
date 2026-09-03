"""Native PDF text extraction, with OCR fallback per page.

A page falls back to OCR only when it carries too little real text to be worth
parsing, so a text PDF never pays the OCR cost.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pymupdf

from restruct.configs import SETTINGS
from restruct.document.types import ExtractedLine
from restruct.ingestion.ocr import _tesseract_page_dict


def _meaningful_character_count(text: str) -> int:
    return sum(character.isalnum() for character in text)

def _line_from_pdf_dict(
    raw_line: dict[str, Any],
    *,
    page_number: int,
    used_ocr: bool,
) -> ExtractedLine | None:
    spans = raw_line.get("spans", [])
    text = "".join(str(span.get("text", "")) for span in spans).strip()
    if not text:
        return None

    sizes = [float(span.get("size", 0.0)) for span in spans]
    fonts = [str(span.get("font", "")).casefold() for span in spans]
    flags = [int(span.get("flags", 0)) for span in spans]
    return ExtractedLine(
        page=page_number,
        text=text,
        bbox=tuple(float(value) for value in raw_line["bbox"]),
        size=max(sizes, default=0.0),
        bold=any(flag & 16 for flag in flags)
        or any("bold" in font or "black" in font for font in fonts),
        used_ocr=used_ocr,
    )

def extract_lines(
    document: pymupdf.Document,
) -> tuple[list[ExtractedLine], list[dict[str, Any]], list[dict[str, Any]]]:
    """Use PyMuPDF for native text/rendering and Tesseract TSV for OCR pages."""
    lines: list[ExtractedLine] = []
    raw_pages: list[dict[str, Any]] = []
    ocr_pages: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="restruct-ocr-") as temporary_name:
        temporary_directory = Path(temporary_name)
        for page_index, page in enumerate(document):
            native_page_dict = page.get_text("dict", sort=True)
            native_text_blocks = [
                block
                for block in native_page_dict.get("blocks", [])
                if block.get("type") == 0
            ]
            raw_pages.append(
                {
                    "page": page_index + 1,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "blocks": native_text_blocks,
                }
            )
            native_text = "".join(
                str(span.get("text", ""))
                for block in native_text_blocks
                for raw_line in block.get("lines", [])
                for span in raw_line.get("spans", [])
            )
            used_ocr = (
                SETTINGS.ocr.enabled
                and _meaningful_character_count(native_text)
                < SETTINGS.ocr.native_text_min_characters
            )

            if used_ocr:
                page_dict = _tesseract_page_dict(
                    page,
                    page_index,
                    temporary_directory,
                )
                ocr_pages.append(page_dict)
            else:
                page_dict = native_page_dict

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for raw_line in block.get("lines", []):
                    line = _line_from_pdf_dict(
                        raw_line,
                        page_number=page_index + 1,
                        used_ocr=used_ocr,
                    )
                    if line is not None:
                        lines.append(line)
    return lines, raw_pages, ocr_pages
