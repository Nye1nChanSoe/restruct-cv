"""Native PDF ingestion: read each page once into the physical representation.

A page falls back to OCR only when it carries too little real text to be worth
parsing, so a text PDF never pays the OCR cost. Both paths produce the same
types, so nothing downstream needs to know which ran.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pymupdf

from restruct.configs import SETTINGS
from restruct.document.physical import (
    Document,
    ImageRegion,
    LinkAnnotation,
    Page,
    Rule,
    Span,
    TextLine,
    Token,
)
from restruct.document.types import ExtractedLine
from restruct.ingestion.ocr import ocr_page

# A drawing this thin in one axis is a rule rather than a filled shape.
_RULE_THICKNESS = 2.0
# Shorter than this and a rule is a glyph artefact, not a separator.
_MINIMUM_RULE_LENGTH = 10.0


def _meaningful_character_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _tokens_from_chars(raw_chars: list[dict[str, Any]]) -> tuple[Token, ...]:
    return tuple(
        Token(
            text=str(char.get("c", "")),
            bbox=tuple(float(value) for value in char["bbox"]),
            origin=tuple(float(value) for value in char["origin"])
            if char.get("origin")
            else None,
            synthetic=bool(char.get("synthetic", False)),
        )
        for char in raw_chars
    )


def _span_from_raw(raw_span: dict[str, Any]) -> Span:
    tokens = _tokens_from_chars(raw_span.get("chars", []))
    return Span(
        # rawdict reports characters rather than a text field; joining them
        # reproduces the dict-mode text exactly (verified across all fixtures).
        text="".join(token.text for token in tokens),
        bbox=tuple(float(value) for value in raw_span["bbox"]),
        font=str(raw_span.get("font", "")),
        size=float(raw_span.get("size", 0.0)),
        flags=int(raw_span.get("flags", 0)),
        tokens=tokens,
        granularity="character",
        color=int(raw_span.get("color", 0)),
        origin=tuple(float(value) for value in raw_span["origin"])
        if raw_span.get("origin")
        else None,
    )


def _debug_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Blocks for the debug dump, with per-character geometry summarised.

    The characters themselves are consumed in memory by word reconstruction and
    nothing reads them back from this file, so writing them out inflated the
    dump 12-20x for no benefit. The count is kept so the dump still shows that
    character geometry was captured.
    """
    summarised: list[dict[str, Any]] = []
    for block in blocks:
        lines = []
        for raw_line in block.get("lines", []):
            spans = []
            for raw_span in raw_line.get("spans", []):
                characters = raw_span.get("chars", [])
                spans.append(
                    {
                        key: value
                        for key, value in raw_span.items()
                        if key != "chars"
                    }
                    | {
                        "text": "".join(str(c.get("c", "")) for c in characters),
                        "characterCount": len(characters),
                    }
                )
            lines.append({**raw_line, "spans": spans})
        summarised.append({**block, "lines": lines})
    return summarised


def _lines_from_blocks(blocks: list[dict[str, Any]], page_number: int) -> tuple[TextLine, ...]:
    lines: list[TextLine] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for raw_line in block.get("lines", []):
            spans = tuple(_span_from_raw(span) for span in raw_line.get("spans", []))
            if not spans:
                continue
            lines.append(
                TextLine(
                    page=page_number,
                    bbox=tuple(float(value) for value in raw_line["bbox"]),
                    spans=spans,
                    direction=tuple(float(value) for value in raw_line.get("dir", (1.0, 0.0))),
                )
            )
    return tuple(lines)


def _rules(page: pymupdf.Page) -> tuple[Rule, ...]:
    """Drawn horizontal and vertical lines, which often separate sections."""
    found: list[Rule] = []
    for drawing in page.get_drawings():
        rectangle = pymupdf.Rect(drawing["rect"])
        box = tuple(float(value) for value in rectangle)
        if rectangle.height <= _RULE_THICKNESS and rectangle.width >= _MINIMUM_RULE_LENGTH:
            found.append(Rule(page.number + 1, box, "horizontal", rectangle.height))
        elif rectangle.width <= _RULE_THICKNESS and rectangle.height >= _MINIMUM_RULE_LENGTH:
            found.append(Rule(page.number + 1, box, "vertical", rectangle.width))
    return tuple(found)


def _images(page_dictionary: dict[str, Any], page_number: int) -> tuple[ImageRegion, ...]:
    return tuple(
        ImageRegion(page_number, tuple(float(value) for value in block["bbox"]))
        for block in page_dictionary.get("blocks", [])
        if block.get("type") == 1
    )


def _links(page: pymupdf.Page) -> tuple[LinkAnnotation, ...]:
    found: list[LinkAnnotation] = []
    for link in page.get_links():
        uri = str(link.get("uri") or "").strip()
        rectangle = link.get("from")
        if not uri or rectangle is None:
            continue
        found.append(
            LinkAnnotation(
                page=page.number + 1,
                bbox=tuple(float(value) for value in pymupdf.Rect(rectangle)),
                uri=uri,
            )
        )
    return tuple(found)


def read_document(document: pymupdf.Document) -> Document:
    """Read every page once into the shared physical representation."""
    pages: list[Page] = []
    raw_pages: list[dict[str, Any]] = []
    ocr_pages: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="restruct-ocr-") as temporary_name:
        temporary_directory = Path(temporary_name)
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            # rawdict carries everything dict does, plus per-character geometry,
            # so one read serves both this pass and word reconstruction.
            native = page.get_text("rawdict", sort=True)
            text_blocks = [b for b in native.get("blocks", []) if b.get("type") == 0]
            raw_pages.append(
                {
                    "page": page_number,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "blocks": _debug_blocks(text_blocks),
                }
            )
            native_text = "".join(
                str(char.get("c", ""))
                for block in text_blocks
                for raw_line in block.get("lines", [])
                for span in raw_line.get("spans", [])
                for char in span.get("chars", [])
            )
            used_ocr = (
                SETTINGS.ocr.enabled
                and _meaningful_character_count(native_text)
                < SETTINGS.ocr.native_text_min_characters
            )

            if used_ocr:
                lines, ocr_page_dictionary = ocr_page(page, page_index, temporary_directory)
                ocr_pages.append(ocr_page_dictionary)
            else:
                lines = _lines_from_blocks(text_blocks, page_number)

            pages.append(
                Page(
                    number=page_number,
                    width=page.rect.width,
                    height=page.rect.height,
                    rotation=int(page.rotation or 0),
                    used_ocr=used_ocr,
                    lines=lines,
                    rules=_rules(page),
                    images=_images(native, page_number),
                    links=_links(page),
                )
            )

    return Document(
        pages=tuple(pages),
        raw_pages=tuple(raw_pages),
        ocr_pages=tuple(ocr_pages),
    )


def extracted_lines(document: Document) -> list[ExtractedLine]:
    """Derive the flat line view the section parsers still consume.

    This is the bridge from Pass 1 to the existing parsers. It reproduces
    exactly what the previous extractor produced, so introducing the physical
    representation changes no output; passes 2 and 3 replace it.
    """
    lines: list[ExtractedLine] = []
    for line in document.lines:
        text = line.text.strip()
        if not text:
            continue
        lines.append(
            ExtractedLine(
                page=line.page,
                text=text,
                bbox=line.bbox,
                size=line.size,
                bold=line.bold,
                used_ocr=line.used_ocr,
            )
        )
    return lines


def extract_lines(
    document: pymupdf.Document,
) -> tuple[list[ExtractedLine], list[dict[str, Any]], list[dict[str, Any]]]:
    """Backwards-compatible entry point for the current pipeline."""
    read = read_document(document)
    return extracted_lines(read), list(read.raw_pages), list(read.ocr_pages)
