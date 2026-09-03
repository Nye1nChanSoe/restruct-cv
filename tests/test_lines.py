"""Guards on pass-3 line reconstruction: baselines and cells."""

from __future__ import annotations

import pymupdf
import pytest

from restruct.document.physical import Document, Page, Span, TextLine, Word
from restruct.document.stats import measure
from restruct.ingestion.native import extracted_lines, read_document
from restruct.layout.lines import cells_in_line, is_row_like, line_baseline
from restruct.layout.rows import _visual_rows
from restruct.layout.words import reconstruct_words
from tests.helpers import SYNTHETIC_DIRECTORY, tesseract_available


def word(text: str, left: float, right: float, *, top: float = 10.0, bottom: float = 22.0) -> Word:
    return Word(text=text, page=1, bbox=(left, top, right, bottom))


def line_with(words: list[Word], *, top: float = 10.0, bottom: float = 22.0) -> TextLine:
    span = Span(
        text=" ".join(w.text for w in words),
        bbox=(words[0].bbox[0], top, words[-1].bbox[2], bottom),
        font="Helvetica",
        size=10.0,
        flags=0,
    )
    return TextLine(1, span.bbox, (span,), words=tuple(words))


def statistics_with(space_width: float, line_height: float = 12.0):
    """Statistics standing in for a measured document."""
    from restruct.document.physical import Token

    tokens = tuple(
        Token(text=character, bbox=(index * 5.0, 10.0, index * 5.0 + 5.0, 10.0 + line_height))
        for index, character in enumerate("reference text")
    )
    spaces = tuple(
        Token(text=" ", bbox=(0.0, 10.0, space_width, 10.0 + line_height)) for _ in range(4)
    )
    span = Span(
        text="reference text",
        bbox=(0.0, 10.0, 70.0, 10.0 + line_height),
        font="Helvetica",
        size=10.0,
        flags=0,
        tokens=tokens + spaces,
    )
    return measure(
        Document(pages=(Page(1, 612.0, 792.0, lines=(TextLine(1, span.bbox, (span,)),)),))
    )


# -- baselines --------------------------------------------------------------


def test_baseline_comes_from_glyph_origins_not_the_box_bottom() -> None:
    """Box bottoms move with descenders; the baseline does not."""
    from restruct.document.physical import Token

    tokens = (
        Token(text="a", bbox=(0, 10, 5, 22), origin=(0.0, 20.0)),
        Token(text="g", bbox=(5, 10, 10, 26), origin=(5.0, 20.0)),   # descender
    )
    span = Span(text="ag", bbox=(0, 10, 10, 26), font="Helvetica", size=10.0, flags=0, tokens=tokens)
    line = TextLine(1, span.bbox, (span,))
    assert line_baseline(line) == 20.0
    assert line.bbox[3] == 26.0


def test_baseline_falls_back_to_the_box_when_no_origins_exist() -> None:
    """OCR reports no origins; the box bottom is consistent within a scan."""
    span = Span(text="abc", bbox=(0, 10, 30, 22), font="TesseractOCR", size=10.0, flags=0)
    assert line_baseline(TextLine(1, span.bbox, (span,))) == 22.0


# -- cells ------------------------------------------------------------------


def test_an_ordinary_line_is_a_single_cell() -> None:
    statistics = statistics_with(space_width=3.0)
    line = line_with([word("Senior", 0, 30), word("Analyst", 33, 70)])
    cells = cells_in_line(line, statistics)
    assert len(cells) == 1
    assert not is_row_like(cells)


def test_a_wide_gap_divides_a_line_into_cells() -> None:
    statistics = statistics_with(space_width=3.0)
    line = line_with(
        [word("Senior", 0, 30), word("Analyst", 33, 70), word("May", 300, 330), word("2023", 333, 370)]
    )
    cells = cells_in_line(line, statistics)
    assert is_row_like(cells)
    assert [cell.text for cell in cells] == ["Senior Analyst", "May 2023"]


def test_cells_keep_their_words_and_decide_nothing() -> None:
    """A cell records that a line is divided, never what the parts mean."""
    statistics = statistics_with(space_width=3.0)
    line = line_with([word("Title", 0, 30), word("2020", 300, 340)])
    left, right = cells_in_line(line, statistics)
    assert [w.text for w in left.words] == ["Title"]
    assert right.bbox[0] == 300
    assert not hasattr(left, "role")


def test_a_line_without_words_has_no_cells() -> None:
    span = Span(text="x", bbox=(0, 0, 5, 10), font="Helvetica", size=10.0, flags=0)
    assert cells_in_line(TextLine(1, span.bbox, (span,)), statistics_with(3.0)) == ()


# -- row grouping -----------------------------------------------------------


def test_rows_group_on_baselines_not_box_overlap() -> None:
    """Two cells of one row share a baseline even with different box heights."""
    from restruct.document.types import ExtractedLine

    lines = [
        ExtractedLine(1, "Senior Site Engineer", (72, 100, 200, 118), 12.0, True, False, baseline=114.0),
        ExtractedLine(1, "Mar 2022 - Present", (400, 104, 520, 116), 9.0, False, False, baseline=114.0),
        ExtractedLine(1, "next line down", (72, 130, 200, 142), 9.0, False, False, baseline=140.0),
    ]
    rows = _visual_rows(lines, range(0, 3), statistics_with(3.0, line_height=12.0))
    assert [len(row) for row in rows] == [2, 1]
    assert [line.text for _, line in rows[0]] == ["Senior Site Engineer", "Mar 2022 - Present"]


def test_rows_never_span_pages() -> None:
    from restruct.document.types import ExtractedLine

    lines = [
        ExtractedLine(1, "last on page one", (72, 700, 200, 712), 9.0, False, False, baseline=710.0),
        ExtractedLine(2, "first on page two", (72, 700, 200, 712), 9.0, False, False, baseline=710.0),
    ]
    rows = _visual_rows(lines, range(0, 2), statistics_with(3.0))
    assert [len(row) for row in rows] == [1, 1]


# -- against the real fixtures ---------------------------------------------


@pytest.mark.parametrize("stem", ["1", "2", "6", "7.anomaly"])
def test_native_resumes_have_no_row_like_lines(stem: str) -> None:
    """A native PDF reports the two halves of a row as two separate lines,
    so finding cells inside one line would be a false positive."""
    document = read_document(pymupdf.open(SYNTHETIC_DIRECTORY / f"{stem}.pdf"))
    statistics = measure(document)
    document = reconstruct_words(document, statistics)
    assert not [line for line in extracted_lines(document, statistics) if line.row_like]


@pytest.mark.skipif(not tesseract_available(), reason="needs tesseract")
def test_ocr_title_and_date_columns_are_recovered_as_cells() -> None:
    """OCR merges the two columns onto one line; cells recover the division."""
    document = read_document(pymupdf.open(SYNTHETIC_DIRECTORY / "9.ocr.pdf"))
    statistics = measure(document)
    document = reconstruct_words(document, statistics)
    row_like = [line for line in extracted_lines(document, statistics) if line.row_like]
    assert row_like
    texts = [tuple(cell.text for cell in line.cells) for line in row_like]
    assert any(
        left.startswith("Senior Operations Analyst") and "2023" in right
        for left, right in (pair for pair in texts if len(pair) == 2)
    )
