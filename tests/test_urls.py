"""Guards on what a link annotation contributes as visible text.

A PDF link rectangle and the glyph boxes under it genuinely disagree, so the
rectangle is widened by a tolerance before words are matched against it. That
tolerance is what let the separator beside a link fall inside it: a header
reading ``LinkedIn | GitHub`` produced a second link labelled ``| GitHub``.
"""

from __future__ import annotations

import pytest

from restruct.document.types import ExtractedLine
from restruct.parsers.urls import _trimmed_label


def line(text: str) -> ExtractedLine:
    return ExtractedLine(
        page=1,
        text=text,
        bbox=(72.0, 100.0, 72.0 + 5.0 * len(text), 112.0),
        size=11.0,
        bold=False,
        used_ocr=False,
    )


HEADER = "LinkedIn | GitHub"


def test_a_separator_swept_in_by_the_tolerance_is_trimmed() -> None:
    """Regression: the label kept the pipe that separates it from the link
    before it."""
    source = line(HEADER)
    assert _trimmed_label(source, HEADER.index("|"), len(HEADER)) == (
        "GitHub",
        HEADER.index("GitHub"),
        len(HEADER),
    )


def test_the_offsets_narrow_with_the_text() -> None:
    """A URL match is reversed onto its source line later, so a label whose
    offsets no longer bracket it is a silent mis-split."""
    source = line(HEADER)
    trimmed = _trimmed_label(source, HEADER.index("|"), len(HEADER))
    assert trimmed is not None
    text, start, end = trimmed
    assert source.text[start:end] == text


def test_an_exact_label_is_returned_unchanged() -> None:
    source = line(HEADER)
    assert _trimmed_label(source, 0, len("LinkedIn")) == ("LinkedIn", 0, 8)


@pytest.mark.parametrize(
    "text",
    [
        "Demo tools",
        "restruct-cv",
        "Open-LinkedOut",
        "github.com/example",
    ],
)
def test_a_label_that_needs_no_trimming_keeps_every_character(text: str) -> None:
    """Including labels that contain a hyphen, a slash or a dot: those are part
    of the name, and only *edge* punctuation is separator residue."""
    source = line(text)
    assert _trimmed_label(source, 0, len(text)) == (text, 0, len(text))


def test_a_span_of_only_separators_yields_no_label() -> None:
    """Better no label than an empty one: the URL is still recorded, and an
    empty visible text would read as a link to nothing."""
    source = line("A | B")
    assert _trimmed_label(source, 1, 4) is None
