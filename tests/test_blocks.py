"""Guards on the block continuation rule.

Nine copies of this arithmetic were spread across the section parsers and had
drifted into three different rules. These tests pin the one rule, and pin the
two axes on which the callers still deliberately differ.

The gap threshold is now measured from the document rather than taken as a
multiple of the two boxes involved, so the fixture below builds a document with
known leading and the tests are expressed against that.
"""

from __future__ import annotations

import pytest

from restruct.document.physical import Document, Page, Span, TextLine
from restruct.document.stats import DocumentStatistics, measure
from restruct.layout.blocks import MINIMUM_GAP, continues_block, extend_block

LINE_HEIGHT = 10.0
LINE_GAP = 2.0


def line(top: float, left: float = 0.0, right: float = 100.0) -> tuple[float, ...]:
    return (left, top, right, top + LINE_HEIGHT)


@pytest.fixture(scope="module")
def statistics() -> DocumentStatistics:
    """A document set with a 10pt line height and 2pt leading."""
    lines = []
    for index in range(8):
        top = index * (LINE_HEIGHT + LINE_GAP)
        span = Span(
            text="ordinary body text",
            bbox=(0.0, top, 100.0, top + LINE_HEIGHT),
            font="Helvetica",
            size=LINE_HEIGHT,
            flags=0,
        )
        lines.append(TextLine(1, span.bbox, (span,)))
    return measure(Document(pages=(Page(1, 612.0, 792.0, lines=tuple(lines)),)))


def test_the_fixture_measures_what_it_claims(statistics: DocumentStatistics) -> None:
    assert statistics.median_line_gap == pytest.approx(LINE_GAP)
    assert statistics.median_line_height == pytest.approx(LINE_HEIGHT)


def test_a_line_at_the_documents_own_leading_continues(statistics: DocumentStatistics) -> None:
    assert continues_block(
        line(0),
        line(LINE_HEIGHT + LINE_GAP),
        same_page=True,
        require_horizontal_overlap=True,
        statistics=statistics,
    )


def test_a_slight_overlap_still_continues(statistics: DocumentStatistics) -> None:
    """Glyph boxes include leading, so consecutive lines can overlap a little."""
    assert continues_block(
        line(0),
        line(LINE_HEIGHT + MINIMUM_GAP),
        same_page=True,
        require_horizontal_overlap=True,
        statistics=statistics,
    )


def test_a_gap_well_beyond_the_documents_leading_starts_a_new_block(
    statistics: DocumentStatistics,
) -> None:
    """The old rule allowed 1.25x the tallest box, which on real resumes
    exceeded every gap present and so never separated anything."""
    assert not continues_block(
        line(0),
        line(LINE_HEIGHT + LINE_GAP * 6),
        same_page=True,
        require_horizontal_overlap=True,
        statistics=statistics,
    )


def test_an_oversized_line_does_not_license_an_oversized_gap(
    statistics: DocumentStatistics,
) -> None:
    """A tall heading above a paragraph must not pull the paragraph into it."""
    tall = (0.0, 0.0, 100.0, 40.0)
    assert not continues_block(
        tall,
        line(40.0 + LINE_GAP * 6),
        same_page=True,
        require_horizontal_overlap=True,
        statistics=statistics,
    )


def test_a_line_above_never_continues(statistics: DocumentStatistics) -> None:
    assert not continues_block(
        line(100), line(0), same_page=True, require_horizontal_overlap=True,
        statistics=statistics,
    )


def test_a_different_page_never_continues(statistics: DocumentStatistics) -> None:
    assert not continues_block(
        line(0), line(LINE_HEIGHT), same_page=False, require_horizontal_overlap=True,
        statistics=statistics,
    )


def test_horizontal_overlap_separates_prose_from_hanging_bullets(
    statistics: DocumentStatistics,
) -> None:
    """A column to the right is not a continuation; a hanging indent is."""
    indented = line(LINE_HEIGHT + LINE_GAP, left=200.0, right=300.0)
    assert not continues_block(
        line(0), indented, same_page=True, require_horizontal_overlap=True,
        statistics=statistics,
    )
    assert continues_block(
        line(0), indented, same_page=True, require_horizontal_overlap=False,
        statistics=statistics,
    )


def test_extend_block_joins_text_grows_the_box_and_keeps_evidence() -> None:
    block = {"text": "Perform preventive", "bbox": [0.0, 0.0, 50.0, 10.0]}
    extend_block(
        block,
        text="and corrective maintenance.",
        box=(0.0, 10.0, 80.0, 20.0),
        entities=[{"type": "url", "text": "example.com"}],
    )
    assert block["text"] == "Perform preventive\nand corrective maintenance."
    assert block["bbox"] == [0.0, 0.0, 80.0, 20.0]
    assert block["entities"] == [{"type": "url", "text": "example.com"}]


def test_extend_block_preserves_the_physical_line_break() -> None:
    """The newline stays so a later pass can still see where the line broke."""
    block = {"text": "first", "bbox": [0.0, 0.0, 10.0, 10.0]}
    extend_block(block, text="second", box=(0.0, 10.0, 10.0, 20.0))
    assert "\n" in block["text"]


def test_extend_block_adds_no_entities_key_when_there_is_no_evidence() -> None:
    block = {"text": "first", "bbox": [0.0, 0.0, 10.0, 10.0]}
    extend_block(block, text="second", box=(0.0, 10.0, 10.0, 20.0))
    assert "entities" not in block


def test_extend_block_appends_to_existing_evidence() -> None:
    block = {
        "text": "first",
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "entities": [{"type": "url", "text": "a.com"}],
    }
    extend_block(
        block,
        text="second",
        box=(0.0, 10.0, 10.0, 20.0),
        entities=[{"type": "url", "text": "b.com"}],
    )
    assert [entity["text"] for entity in block["entities"]] == ["a.com", "b.com"]
