"""Guards on unsupported-layout detection.

Two questions matter and they pull in opposite directions. A detector that
never fires is useless, and a detector that fires on the documents v1 actually
targets is worse than useless -- it would teach a reader to ignore the warning.
So every check here is paired: a real positive and the full synthetic set as
negatives.

The column fixtures are real two-column resumes. The other four shapes have no
fixture, so each is built here as the smallest PDF that exhibits it; a detector
with no positive case is an assertion that has never been tested.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from restruct.document.stats import measure
from restruct.ingestion.native import read_document
from restruct.layout.unsupported import LayoutWarning, detect_unsupported_layouts
from tests.helpers import (
    SYNTHETIC_DIRECTORY,
    fixture_path,
    UNSUPPORTED_DIRECTORY,
    synthetic_stems,
    tesseract_available,
)

UNSUPPORTED_STEMS = ("3.cols", "4.cols")


def warnings_for(source_path: Path) -> tuple[LayoutWarning, ...]:
    if source_path.suffix.casefold() == ".docx":
        from restruct.ingestion.docx import read_docx

        document = read_docx(source_path)
        return detect_unsupported_layouts(document, measure(document))
    with pymupdf.open(source_path) as pdf:
        document = read_document(pdf)
        return detect_unsupported_layouts(document, measure(document))


def kinds(warnings: tuple[LayoutWarning, ...]) -> set[str]:
    return {warning.kind for warning in warnings}


# -- the fixtures the milestone names ---------------------------------------


@pytest.mark.parametrize("stem", UNSUPPORTED_STEMS)
def test_a_two_column_resume_is_reported(stem: str) -> None:
    warnings = warnings_for(UNSUPPORTED_DIRECTORY / f"{stem}.pdf")
    assert "multiple_columns" in kinds(warnings)


@pytest.mark.parametrize("stem", UNSUPPORTED_STEMS)
def test_every_page_of_a_two_column_resume_is_reported(stem: str) -> None:
    """A warning on page 1 alone would let page 2 be flattened in silence."""
    with pymupdf.open(UNSUPPORTED_DIRECTORY / f"{stem}.pdf") as pdf:
        page_count = pdf.page_count
    columns = [w for w in warnings_for(UNSUPPORTED_DIRECTORY / f"{stem}.pdf")
               if w.kind == "multiple_columns"]
    assert {warning.page for warning in columns} == set(range(1, page_count + 1))


@pytest.mark.parametrize("stem", synthetic_stems())
def test_a_supported_resume_raises_nothing(stem: str) -> None:
    """The whole value of the warning is that these six do not trigger it."""
    if stem.endswith(".ocr") and not tesseract_available():
        pytest.skip("needs tesseract")
    assert warnings_for(fixture_path(stem)) == ()


def test_the_gutter_is_where_the_columns_actually_are() -> None:
    """A count can be right while the box sits somewhere else entirely.

    Both fixtures set a sidebar against a main column that starts at x=244, so
    the gutter has to end at that edge and be wide enough to be a gutter rather
    than a wide word space.
    """
    statistics = measure(read_document(pymupdf.open(UNSUPPORTED_DIRECTORY / "3.cols.pdf")))
    for gutter in statistics.column_gutters:
        assert gutter.width > 20
        assert 230 <= gutter.right <= 250
        assert gutter.left_column_lines >= 4 and gutter.right_column_lines >= 4


# -- the shapes with no fixture ---------------------------------------------


def _page(width: float = 400.0, height: float = 400.0) -> pymupdf.Document:
    """A blank page carrying enough ordinary text to stay on the native path.

    Below twenty native characters ingestion falls back to OCR, which would
    quietly make these tests depend on Tesseract and on what it happened to
    read. The filler sits low on the page, in one column, so it cannot itself
    trigger any of the detectors under test.
    """
    document = pymupdf.open()
    page = document.new_page(width=width, height=height)
    for index in range(6):
        page.insert_text((40, 300 + index * 14), f"Ordinary body text line {index}")
    return document


def _detect(document: pymupdf.Document) -> tuple[LayoutWarning, ...]:
    physical = read_document(document)
    return detect_unsupported_layouts(physical, measure(physical))


def test_vertical_text_is_reported() -> None:
    document = _page()
    page = document[0]
    page.insert_text((50, 200), "Rotated sidebar label", rotate=90)
    page.insert_text((100, 100), "Ordinary horizontal text")
    assert "vertical_text" in kinds(_detect(document))


def test_overlapping_text_boxes_are_reported() -> None:
    document = _page()
    page = document[0]
    page.insert_text((100, 100), "Absolutely positioned text")
    page.insert_text((100, 101), "Absolutely positioned text")
    assert "overlapping_text" in kinds(_detect(document))


def test_ordinary_stacked_lines_are_not_overlapping() -> None:
    """Consecutive lines share ascender and descender space; that is not an
    overlap, and treating it as one would fire on every document."""
    document = _page()
    page = document[0]
    for index in range(6):
        page.insert_text((100, 100 + index * 12), f"Line number {index}")
    assert "overlapping_text" not in kinds(_detect(document))


def test_text_inside_a_graphic_is_reported() -> None:
    document = _page()
    page = document[0]
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 60))
    pixmap.clear_with(200)
    page.insert_image(pymupdf.Rect(40, 40, 160, 100), pixmap=pixmap)
    page.insert_text((50, 75), "Inside the graphic")
    assert "text_in_graphics" in kinds(_detect(document))


@pytest.mark.skipif(not tesseract_available(), reason="needs tesseract")
def test_a_scanned_page_is_not_text_in_a_graphic() -> None:
    """Every line of a scan sits inside the page image by construction. That
    is the OCR path working, not a decorative graphic."""
    warnings = warnings_for(SYNTHETIC_DIRECTORY / "9.ocr.pdf")
    assert "text_in_graphics" not in kinds(warnings)


def test_a_nested_table_is_reported() -> None:
    document = _page()
    page = document[0]
    for y in (100, 140, 180, 220):
        page.draw_line(pymupdf.Point(40, y), pymupdf.Point(360, y))
    # A separator belonging to a table drawn inside one cell of that grid.
    page.draw_line(pymupdf.Point(200, 160), pymupdf.Point(340, 160))
    assert "nested_table" in kinds(_detect(document))


def test_a_flat_table_is_not_a_nested_one() -> None:
    """7.anomaly's certification table is a supported shape and must stay one."""
    warnings = warnings_for(SYNTHETIC_DIRECTORY / "7.anomaly.pdf")
    assert "nested_table" not in kinds(warnings)


def test_heading_underlines_are_not_a_table_grid() -> None:
    """Resume 6 underlines each heading, at a different width every time. Rules
    that share only a left edge are not a grid, and a shorter one between two
    of them is just the next heading."""
    warnings = warnings_for(SYNTHETIC_DIRECTORY / "6.pdf")
    assert "nested_table" not in kinds(warnings)


# -- the conservative parse -------------------------------------------------


def test_row_grouping_refuses_to_cross_a_gutter() -> None:
    """Two columns set side by side align by accident constantly. Joining them
    into a row is the silent flattening this milestone exists to prevent."""
    from restruct.ingestion.native import extracted_lines
    from restruct.layout.rows import _visual_rows

    with pymupdf.open(UNSUPPORTED_DIRECTORY / "3.cols.pdf") as pdf:
        physical = read_document(pdf)
        statistics = measure(physical)
        lines = extracted_lines(physical, statistics)
        rows = _visual_rows(lines, range(len(lines)), statistics)

    for row in rows:
        for _, first in row:
            for _, second in row:
                assert not statistics.separated_by_a_gutter(
                    first.page, first.bbox, second.bbox
                )


def test_a_supported_document_still_groups_rows() -> None:
    """The gutter check must not disable row grouping generally: 7.anomaly's
    tables depend on it."""
    from restruct.ingestion.native import extracted_lines
    from restruct.layout.rows import _visual_rows

    with pymupdf.open(SYNTHETIC_DIRECTORY / "7.anomaly.pdf") as pdf:
        physical = read_document(pdf)
        statistics = measure(physical)
        lines = extracted_lines(physical, statistics)
        rows = _visual_rows(lines, range(len(lines)), statistics)

    assert any(len(row) > 1 for row in rows)
