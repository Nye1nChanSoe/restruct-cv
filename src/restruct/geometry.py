"""Shared box arithmetic.

Every stage reasons about rectangles, and before this module each did so with
its own copy of the same expression. Keeping the arithmetic here means a change
to how a span is located applies everywhere at once.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import pymupdf

from restruct.configs import SETTINGS

Box = Sequence[float]


def rounded(box: Box | pymupdf.Rect) -> list[float]:
    """A box at the precision the debug artifacts record."""
    return [round(float(value), 2) for value in box]


def pixel_box(box: Box) -> tuple[int, int, int, int]:
    """Convert a PDF-space box to pixels in a debug render."""
    return tuple(round(value * SETTINGS.debug.scale) for value in box)  # type: ignore[return-value]


def union(boxes: Iterable[Box | pymupdf.Rect]) -> pymupdf.Rect:
    """The smallest rectangle covering every input box."""
    rectangle: pymupdf.Rect | None = None
    for box in boxes:
        candidate = pymupdf.Rect(box)
        rectangle = candidate if rectangle is None else rectangle | candidate
    if rectangle is None:
        raise ValueError("union() requires at least one box")
    return rectangle


def union_by_page(lines: Iterable[Any]) -> list[dict[str, Any]]:
    """One covering box per page, for lines that may span several pages."""
    boxes_by_page: dict[int, pymupdf.Rect] = {}
    for line in lines:
        rectangle = pymupdf.Rect(line.bbox)
        boxes_by_page[line.page] = boxes_by_page.get(line.page, rectangle) | rectangle
    return [
        {"page": page, "bbox": rounded(rectangle)}
        for page, rectangle in sorted(boxes_by_page.items())
    ]


def vertical_overlap(first: Box | pymupdf.Rect, second: Box | pymupdf.Rect) -> float:
    """Height shared by two boxes; negative when they do not overlap."""
    a, b = pymupdf.Rect(first), pymupdf.Rect(second)
    return min(a.y1, b.y1) - max(a.y0, b.y0)


def horizontal_overlap(first: Box | pymupdf.Rect, second: Box | pymupdf.Rect) -> float:
    """Width shared by two boxes, clamped at zero."""
    a, b = pymupdf.Rect(first), pymupdf.Rect(second)
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


def estimate_span_box(line: Any, start: int, end: int) -> pymupdf.Rect | None:
    """Locate a character range by proportion of the line's width.

    Used when the PDF text search cannot find the span, which is routine for
    OCR lines whose reconstructed text does not match any searchable content.
    """
    if not line.text or not 0 <= start < end <= len(line.text):
        return None
    box = pymupdf.Rect(line.bbox)
    character_width = box.width / len(line.text)
    return pymupdf.Rect(
        box.x0 + character_width * start,
        box.y0,
        box.x0 + character_width * end,
        box.y1,
    )


def resolve_span_box(
    document: pymupdf.Document,
    line: Any,
    text: str,
    start: int,
    end: int,
    *,
    prefer_nearest: bool = True,
) -> list[float]:
    """Find the box for ``text`` at ``start``..``end`` within one line.

    ``prefer_nearest`` picks, among several search hits, the one closest to
    where the character offsets say the span sits. That matters when a word
    repeats in a line: taking the first hit can box the wrong occurrence.
    """
    found = document[line.page - 1].search_for(text, clip=pymupdf.Rect(line.bbox))
    estimated = estimate_span_box(line, start, end)

    if found and prefer_nearest and estimated is not None:
        rectangle = min(
            found,
            key=lambda candidate: abs(
                (candidate.x0 + candidate.x1) - (estimated.x0 + estimated.x1)
            ),
        )
    elif found:
        rectangle = found[0]
    elif estimated is not None:
        rectangle = estimated
    else:
        rectangle = pymupdf.Rect(line.bbox)
    return rounded(rectangle)
