"""Pass 3: line-level reconstruction from words.

Two things happen here that the span-level view could not do.

A line gets a real typographic baseline, taken from glyph origins rather than
from the bottom of its bounding box. Box bottoms move with descenders and with
the tallest glyph on the line, so two cells of the same row can have visibly
different box bottoms while sharing one baseline.

A line is also divided into cells wherever a gap is far too wide to be word
spacing. That records *that* the line is divided, and preserves the parts, while
leaving what the parts mean to a later pass. This matters most on scanned pages,
where OCR merges a left-hand title and a right-hand date range that a native PDF
would have reported as two separate lines.
"""

from __future__ import annotations

import statistics as statistics_module

from restruct.document.physical import Cell, TextLine, Word
from restruct.document.stats import DocumentStatistics
from restruct.geometry import union


def line_baseline(line: TextLine) -> float:
    """The line's typographic baseline.

    Glyph origins give it exactly. OCR reports none, so the box bottom stands in
    -- less precise, but consistent within a scanned page, which is what row
    grouping needs.
    """
    origins = [
        token.origin[1]
        for span in line.spans
        for token in span.tokens
        if token.origin is not None and token.text.strip()
    ]
    if origins:
        return float(statistics_module.median(origins))
    return float(line.bbox[3])


def cells_in_line(line: TextLine, statistics: DocumentStatistics) -> tuple[Cell, ...]:
    """Divide a line wherever a gap is far too wide to be word spacing.

    Returns a single cell for an ordinary line, so callers can treat every line
    the same way and ask ``len(cells) > 1`` when they care.
    """
    words = line.words
    if not words:
        return ()

    threshold = statistics.cell_gap_threshold
    groups: list[list[Word]] = [[words[0]]]
    for previous, current in zip(words, words[1:]):
        gap = current.bbox[0] - previous.bbox[2]
        if threshold > 0 and gap >= threshold:
            groups.append([current])
        else:
            groups[-1].append(current)

    return tuple(
        Cell(
            text=" ".join(word.text for word in group),
            bbox=tuple(float(value) for value in union(word.bbox for word in group)),
            words=tuple(group),
        )
        for group in groups
    )


def is_row_like(cells: tuple[Cell, ...]) -> bool:
    """Whether a line carries separated cells rather than continuous text."""
    return len(cells) > 1
