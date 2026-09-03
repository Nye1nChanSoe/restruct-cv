"""Layouts v1 does not claim to read correctly.

The design brief targets single-column ATS-friendly resumes. Everything else --
independent columns, sidebars, nested tables, vertical text, text drawn inside a
graphic, overlapping text boxes -- can still be flattened into a sequence of
lines, and the result will look plausible while silently reordering the
document. That failure is worse than an obvious one, because nothing downstream
can tell it happened.

So each shape is detected and recorded. The parse continues, because a partial
resume is more useful than none, but the reading order is no longer treated as
evidence: row grouping refuses to join across a column gutter, and the warnings
are written where a reader will see them.

Nothing here repairs a layout. Multi-column and table-aware reading-order
reconstruction is a later version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from restruct.document.physical import BBox, Document, Page
from restruct.document.stats import DocumentStatistics
from restruct.geometry import overlap_ratio

WarningKind = Literal[
    "multiple_columns",
    "vertical_text",
    "overlapping_text",
    "text_in_graphics",
    "nested_table",
]

# Two text boxes sharing this much of the smaller one are drawn on top of each
# other. On the fixtures the worst honest pair -- two consecutive bullets whose
# descenders and ascenders interleave -- reaches 0.11.
_OVERLAPPING_TEXT_RATIO = 0.5

# A line this far inside an image region is part of the graphic rather than
# merely sitting over a background.
_TEXT_IN_GRAPHICS_RATIO = 0.9

# An image covering this much of a page is a background or a scan, and text on
# top of it is ordinary text.
_FULL_PAGE_IMAGE_RATIO = 0.7

# Rules within this much of each other in x count as the same table edge.
_RULE_EDGE_TOLERANCE = 2.0

# Fewer aligned rules than this is a set of heading underlines, not a table.
_MINIMUM_TABLE_RULES = 3

# A nested table's separators are materially shorter than the table enclosing
# them; anything closer than this is the same table's own grid.
_NESTED_RULE_WIDTH_RATIO = 0.9


@dataclass(frozen=True)
class LayoutWarning:
    """One reason this document's reading order may not be recoverable."""

    kind: WarningKind
    page: int
    detail: str
    bbox: BBox | None = None


def _column_warnings(statistics: DocumentStatistics) -> list[LayoutWarning]:
    """Independent columns, including the sidebar case.

    A sidebar and a pair of balanced columns are the same measurement -- a
    corridor no line crosses, with text living on both sides of it at the same
    height -- and differ only in how wide the two sides are. Both are reported
    as one warning carrying those widths, rather than as two detectors with a
    threshold invented to separate them.
    """
    return [
        LayoutWarning(
            kind="multiple_columns",
            page=gutter.page,
            detail=(
                f"{gutter.width:.0f}pt gutter at x={gutter.left:.0f}-{gutter.right:.0f} "
                f"separates {gutter.left_column_lines} lines of text from "
                f"{gutter.right_column_lines}; reading order across it is a guess"
            ),
            bbox=gutter.bbox,
        )
        for gutter in statistics.column_gutters
    ]


def _vertical_text_warnings(page: Page) -> list[LayoutWarning]:
    """Text set at an angle, which has no place in a top-to-bottom reading."""
    rotated = [line for line in page.lines if line.text.strip() and not line.is_horizontal]
    if not rotated:
        return []
    return [
        LayoutWarning(
            kind="vertical_text",
            page=page.number,
            detail=f"{len(rotated)} line(s) are not set horizontally",
            bbox=rotated[0].bbox,
        )
    ]


def _overlapping_text_warnings(page: Page) -> list[LayoutWarning]:
    """Text boxes drawn on top of one another.

    Overlapping boxes mean the source positioned text absolutely rather than in
    a flow, so the order the boxes were emitted in carries no reading order.
    """
    lines = [line for line in page.lines if line.text.strip()]
    for index, line in enumerate(lines):
        for other in lines[index + 1 :]:
            if overlap_ratio(line.bbox, other.bbox) >= _OVERLAPPING_TEXT_RATIO:
                return [
                    LayoutWarning(
                        kind="overlapping_text",
                        page=page.number,
                        detail=(
                            f"{line.text.strip()[:40]!r} and {other.text.strip()[:40]!r} "
                            "are drawn on top of each other"
                        ),
                        bbox=line.bbox,
                    )
                ]
    return []


def _text_in_graphics_warnings(page: Page) -> list[LayoutWarning]:
    """Text sitting inside a decorative graphic rather than in the text flow.

    A scanned page is excluded outright: every line on it is inside the page
    image by definition, which is the OCR path working, not a bad layout.
    """
    if page.used_ocr or page.width <= 0 or page.height <= 0:
        return []

    page_area = page.width * page.height
    decorative = [
        region
        for region in page.images
        if (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1])
        < page_area * _FULL_PAGE_IMAGE_RATIO
    ]
    for region in decorative:
        inside = [
            line
            for line in page.lines
            if line.text.strip()
            and overlap_ratio(line.bbox, region.bbox) >= _TEXT_IN_GRAPHICS_RATIO
        ]
        if inside:
            return [
                LayoutWarning(
                    kind="text_in_graphics",
                    page=page.number,
                    detail=f"{len(inside)} line(s) sit inside an image region",
                    bbox=region.bbox,
                )
            ]
    return []


def _nested_table_warnings(page: Page) -> list[LayoutWarning]:
    """A table drawn inside another table's cell.

    An outer table's row separators all run its full width, so three or more
    horizontal rules sharing both end points are a table grid. A rule that sits
    between two of them but stops short of both ends belongs to something drawn
    inside one cell -- which means the cell has structure of its own, and its
    text cannot be read as a single value.

    Three rules are required so that a page of heading underlines, which share
    a left edge but no right edge, cannot be mistaken for a grid.
    """
    rules = sorted(page.horizontal_rules, key=lambda rule: rule.bbox[1])
    grids: dict[tuple[float, float], list] = {}
    for rule in rules:
        key = (
            round(rule.bbox[0] / _RULE_EDGE_TOLERANCE),
            round(rule.bbox[2] / _RULE_EDGE_TOLERANCE),
        )
        grids.setdefault(key, []).append(rule)  # type: ignore[arg-type]

    for aligned in grids.values():
        if len(aligned) < _MINIMUM_TABLE_RULES:
            continue
        left = min(rule.bbox[0] for rule in aligned)
        right = max(rule.bbox[2] for rule in aligned)
        width = right - left
        top = min(rule.bbox[1] for rule in aligned)
        bottom = max(rule.bbox[3] for rule in aligned)
        if width <= 0:
            continue
        for rule in rules:
            if rule in aligned:
                continue
            if not (top < rule.bbox[1] and rule.bbox[3] < bottom):
                continue
            if rule.bbox[0] < left or rule.bbox[2] > right:
                continue
            if rule.length >= width * _NESTED_RULE_WIDTH_RATIO:
                continue
            return [
                LayoutWarning(
                    kind="nested_table",
                    page=page.number,
                    detail=(
                        f"a {rule.length:.0f}pt rule sits inside a {width:.0f}pt "
                        "table grid, so a cell contains a table of its own"
                    ),
                    bbox=rule.bbox,
                )
            ]
    return []


def detect_unsupported_layouts(
    document: Document,
    statistics: DocumentStatistics,
) -> tuple[LayoutWarning, ...]:
    """Every reason this document's reading order may not be recoverable.

    Empty for the single-column resumes v1 targets, which is what makes a
    non-empty result worth surfacing.
    """
    warnings = _column_warnings(statistics)
    for page in document.pages:
        warnings.extend(_vertical_text_warnings(page))
        warnings.extend(_overlapping_text_warnings(page))
        warnings.extend(_text_in_graphics_warnings(page))
        warnings.extend(_nested_table_warnings(page))
    return tuple(sorted(warnings, key=lambda warning: (warning.page, warning.kind)))
