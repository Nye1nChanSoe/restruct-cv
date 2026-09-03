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

# How far an unclosed bracket may reach for its closing one, in line heights.
# A resume wraps a parenthetical onto the next line, not across a section: the
# bound is what stops an unmatched "(" joining everything after it.
_UNCLOSED_BRACKET_LINES = 3.0


def _has_unclosed_bracket(text: str) -> bool:
    from restruct.structure.separators import is_parenthetically_complete

    return bool(text) and not is_parenthetically_complete(text)


def continues_block(
    previous_box: Sequence[float] | pymupdf.Rect,
    current_box: Sequence[float] | pymupdf.Rect,
    *,
    same_page: bool,
    require_horizontal_overlap: bool,
    statistics: DocumentStatistics,
    previous_text: str = "",
) -> bool:
    """Whether ``current_box`` continues the block ending at ``previous_box``.

    ``same_page`` is passed in rather than derived because the callers hold the
    page number in different shapes, and two of them deliberately do not check
    it at all.

    ``require_horizontal_overlap`` distinguishes prose continuation, where the
    next line must sit under the previous one, from bullet continuation, where
    a hanging indent may leave no overlap.

    ``previous_text`` lets the block say it is not finished. A line ending with
    a bracket still open -- "Forklift Safety Awareness (non-licensed" -- is
    half a phrase, and the geometry gap is the wrong question to ask about it:
    the author's own punctuation says the thought continues. That evidence
    overrides the gap, but nothing else: still the same page, still below, and
    still within a few lines, so one stray bracket cannot swallow a section.
    """
    if not same_page:
        return False
    previous, current = pymupdf.Rect(previous_box), pymupdf.Rect(current_box)
    gap = current.y0 - previous.y1
    if gap < MINIMUM_GAP:
        return False
    if not statistics.is_paragraph_gap(gap):
        if not _has_unclosed_bracket(previous_text):
            return False
        if gap > max(statistics.median_line_height, 1.0) * _UNCLOSED_BRACKET_LINES:
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


def last_block_text(entry: dict[str, Any]) -> str:
    """The text of the block an entry is currently accumulating into.

    Callers pass this to ``continues_block`` so a block whose brackets are
    still open can say the thought is unfinished. ``_lastType`` records which
    list the entry appended to most recently; when it is unset, either list
    will do, because only an entry that has accumulated nothing has neither.
    """
    kinds = ("bullets", "paragraphs") if entry.get("_lastType") == "bullet" else (
        "paragraphs",
        "bullets",
    )
    for kind in kinds:
        items = entry.get(kind) or []
        if items:
            last = items[-1]
            return str(last.get("text", "")) if isinstance(last, dict) else str(last)
    return ""
