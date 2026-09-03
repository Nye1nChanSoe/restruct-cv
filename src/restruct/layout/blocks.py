"""The continuation rule that joins a line to the block above it.

Every section parser needs to answer the same question -- does this line
continue the paragraph or bullet before it, or start a new one -- and before
this module each answered it with its own copy of the arithmetic. Nine copies
had drifted into three different rules, which was invisible while they were
spread across the file.

The rule itself stays here; each caller still states which variant it wants, so
the remaining differences are declared rather than accidental. Unifying them is
a behavior change and belongs to the pass-3 rewrite, not to a refactor.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import pymupdf

from restruct.document.stats import DocumentStatistics
from restruct.geometry import horizontal_overlap, rounded, union

# Consecutive lines may overlap slightly when glyph boxes include leading, so a
# small negative gap still reads as "directly below".
MINIMUM_GAP = -2.0


def continues_block(
    previous_box: Sequence[float] | pymupdf.Rect,
    current_box: Sequence[float] | pymupdf.Rect,
    *,
    same_page: bool,
    require_horizontal_overlap: bool,
    statistics: DocumentStatistics,
) -> bool:
    """Whether ``current_box`` continues the block ending at ``previous_box``.

    ``same_page`` is passed in rather than derived because the callers hold the
    page number in different shapes, and two of them deliberately do not check
    it at all.

    ``require_horizontal_overlap`` distinguishes prose continuation, where the
    next line must sit under the previous one, from bullet continuation, where
    a hanging indent may leave no overlap.
    """
    if not same_page:
        return False
    previous, current = pymupdf.Rect(previous_box), pymupdf.Rect(current_box)
    gap = current.y0 - previous.y1
    if gap < MINIMUM_GAP or not statistics.is_paragraph_gap(gap):
        return False
    if require_horizontal_overlap:
        return horizontal_overlap(previous, current) > 0
    return True


def extend_block(
    block: dict[str, Any],
    *,
    text: str,
    box: Sequence[float] | pymupdf.Rect,
    entities: Iterable[dict[str, Any]] = (),
) -> None:
    """Append a continuation line to a block, growing its box and evidence.

    The newline is preserved rather than collapsed to a space so a later stage
    can still see where the physical line broke.
    """
    block["text"] += "\n" + text
    block["bbox"] = rounded(union([block["bbox"], box]))
    entities = list(entities)
    if entities:
        block.setdefault("entities", []).extend(entities)


def append_paragraph(
    group: dict[str, Any],
    *,
    text: str,
    page: int,
    bbox: list[float],
    entities: list[dict[str, Any]],
    statistics: DocumentStatistics,
) -> None:
    """Append prose to a group, continuing the previous paragraph when it fits."""
    current_box = pymupdf.Rect(bbox)
    if group["paragraphs"] and group["paragraphs"][-1]["page"] == page:
        previous = group["paragraphs"][-1]
        if continues_block(
            previous["bbox"],
            current_box,
            same_page=True,
            require_horizontal_overlap=True,
            statistics=statistics,
        ):
            extend_block(previous, text=text, box=current_box, entities=entities)
            group["_lastType"] = "paragraph"
            group["_lastLineBbox"] = bbox
            return
    value: dict[str, Any] = {
        "text": text,
        "page": page,
        "bbox": bbox,
        "detectionMethod": "geometry_default",
    }
    if entities:
        value["entities"] = entities
    group["paragraphs"].append(value)
    group["_lastType"] = "paragraph"
    group["_lastLineBbox"] = bbox
