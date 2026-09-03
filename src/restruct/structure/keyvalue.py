"""Key-value pairs inside a section.

These stay neutral structural objects here; deciding what a label *means* is
semantic work that belongs to the extraction stage.
"""
from __future__ import annotations

from typing import Any

import pymupdf

from restruct.document.types import ExtractedLine
from restruct.geometry import resolve_span_box
from restruct.patterns.separators import (
    KEY_VALUE_COLON_RE,
    KEY_VALUE_DASH_RE,
    KEY_VALUE_TAB_RE,
)


def _skill_inline_parts(
    text: str,
    source_start: int = 0,
    *,
    allow_single_dash_body: bool = True,
) -> tuple[str, int, int, str, int, int, str] | None:
    """Return a short group label and untouched body from a delimiter row."""
    for pattern, method in (
        (KEY_VALUE_COLON_RE, "delimiter_colon"),
        (KEY_VALUE_TAB_RE, "delimiter_tab"),
        (KEY_VALUE_DASH_RE, "delimiter_dash"),
    ):
        match = pattern.match(text)
        if match is None:
            continue
        label = match.group("label").strip()
        body = match.group("body").strip()
        if (
            not label
            or not body
            or len(label.split()) > 7
            or len(label) > 50
            or label.casefold().startswith(("http", "www."))
        ):
            continue
        if (
            method == "delimiter_dash"
            and not allow_single_dash_body
            and "," not in body
            and ";" not in body
        ):
            continue
        label_offset = match.start("label") + len(match.group("label")) - len(match.group("label").lstrip())
        body_offset = match.start("body") + len(match.group("body")) - len(match.group("body").lstrip())
        label_start = source_start + label_offset
        body_start = source_start + body_offset
        return (
            label,
            label_start,
            label_start + len(label),
            body,
            body_start,
            body_start + len(body),
            method,
        )
    return None

def _skill_subheading_value(
    document: pymupdf.Document,
    line: ExtractedLine,
    text: str,
    start: int,
    end: int,
    detection_method: str,
) -> dict[str, Any]:
    return {
        "text": text,
        "page": line.page,
        "bbox": resolve_span_box(document, line, text, start, end),
        "detectionMethod": detection_method,
    }
