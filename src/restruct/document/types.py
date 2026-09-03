"""Shared physical types for the reconstructed document.

These describe *what was found on the page* and carry no inference. Keeping them
out of the model module lets the layout and parser stages depend on the document
representation without importing DistilBERT or MiniLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from restruct.document.physical import Cell, Word


@dataclass(frozen=True)
class ExtractedLine:
    """One physical line of text with the geometry and typography behind it.

    The trailing fields are pass-3 reconstruction, added with defaults so the
    section parsers can adopt them one at a time: ``baseline`` is the true
    typographic baseline, ``words`` the reconstructed words, and ``cells`` the
    line's divisions when a gap is far too wide to be word spacing.
    """

    page: int
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    bold: bool
    used_ocr: bool
    baseline: float = 0.0
    words: tuple[Word, ...] = ()
    cells: tuple[Cell, ...] = ()
    # (table, row, column) when the source stated it, as a DOCX does. None for
    # a PDF, where a table has to be recovered from gaps between boxes.
    table_cell: tuple[int, int, int] | None = None

    @property
    def row_like(self) -> bool:
        """Whether this line holds separated cells rather than running text."""
        return len(self.cells) > 1


@dataclass(frozen=True)
class DetectedHeading:
    """A line accepted as a section heading, with the evidence that accepted it."""

    line_index: int
    section_type: str
    similarity: float
    runner_up_similarity: float


@dataclass(frozen=True)
class HeaderEntityMatch:
    """A claimed character span within one line, and how it was claimed.

    ``start``/``end`` index into the source line's text, so a later stage can
    always recover the original unsplit text behind any decision.
    """

    kind: str
    text: str
    line_index: int
    start: int
    end: int
    detection_method: str
    confidence: float | None = None
    url: str | None = None
    bbox: tuple[float, float, float, float] | None = None


def overlaps_existing(
    matches: list[HeaderEntityMatch],
    *,
    line_index: int,
    start: int,
    end: int,
) -> bool:
    """Whether an already-claimed span covers any of ``start``..``end``.

    Extraction runs in precedence order, so each stage uses this to leave spans
    that an earlier, more reliable stage already claimed.
    """
    return any(
        match.line_index == line_index
        and match.start < end
        and start < match.end
        for match in matches
    )


def append_regex_matches(
    matches: list[HeaderEntityMatch],
    *,
    line_index: int,
    text: str,
    kind: str,
    pattern: re.Pattern[str],
) -> None:
    """Claim every match of ``pattern`` that no earlier stage already claimed."""
    for match in pattern.finditer(text):
        if overlaps_existing(
            matches,
            line_index=line_index,
            start=match.start(),
            end=match.end(),
        ):
            continue
        matches.append(
            HeaderEntityMatch(
                kind=kind,
                text=match.group(0),
                line_index=line_index,
                start=match.start(),
                end=match.end(),
                detection_method="regex",
                url=match.group(0) if kind == "url" else None,
            )
        )
