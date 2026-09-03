"""Guards on document-relative statistics.

These replace fixed thresholds, so the thing to protect is that they stay
*relative*: the same measurement must hold on a document set at 9pt and one set
at 14pt, and must degrade safely on a document too sparse to measure.
"""

from __future__ import annotations

import pymupdf
import pytest

from restruct.document.physical import Document, Page, Span, TextLine
from restruct.document.stats import measure
from restruct.ingestion.native import read_document
from tests.helpers import SYNTHETIC_DIRECTORY


def line(text: str, top: float, *, size: float = 11.0, left: float = 72.0, page: int = 1) -> TextLine:
    span = Span(
        text=text,
        bbox=(left, top, left + 5.0 * len(text), top + size),
        font="Helvetica",
        size=size,
        flags=0,
    )
    return TextLine(page, span.bbox, (span,))


def document_of(*lines: TextLine, height: float = 792.0, pages: int = 1) -> Document:
    by_page: dict[int, list[TextLine]] = {}
    for item in lines:
        by_page.setdefault(item.page, []).append(item)
    return Document(
        pages=tuple(
            Page(number, 612.0, height, lines=tuple(by_page.get(number, ())))
            for number in range(1, pages + 1)
        )
    )


# -- degradation ------------------------------------------------------------


def test_an_empty_document_measures_without_raising() -> None:
    """An unparseable document must not crash the pipeline before it can warn."""
    statistics = measure(Document())
    assert statistics.body_font_size == 0.0
    assert statistics.is_larger_than_body(99.0) is False
    assert statistics.indentation_level(100.0) == 0


# -- typography -------------------------------------------------------------


def test_body_size_is_weighted_by_text_not_by_line_count() -> None:
    """A resume can have more heading lines than body lines."""
    statistics = measure(
        document_of(
            line("H1", 0, size=20.0),
            line("H2", 40, size=20.0),
            line("H3", 80, size=20.0),
            line("a much longer run of ordinary body text", 120, size=10.0),
        )
    )
    assert statistics.body_font_size == 10.0


@pytest.mark.parametrize("body_size", [8.0, 11.0, 14.0])
def test_size_comparison_is_relative_to_the_document(body_size: float) -> None:
    """The same judgement must hold whatever size the document is set at."""
    statistics = measure(
        document_of(
            line("ordinary body text repeated for weight", 0, size=body_size),
            line("HEADING", 40, size=body_size * 1.5),
        )
    )
    assert statistics.is_body_size(body_size)
    assert statistics.is_larger_than_body(body_size * 1.5)
    assert not statistics.is_larger_than_body(body_size)


def test_near_identical_sizes_count_as_the_same_size() -> None:
    """Exporters emit 10.98 and 11.0 for one run of text."""
    statistics = measure(document_of(line("body text here", 0, size=11.0)))
    assert statistics.is_body_size(10.98)


def test_a_mostly_bold_document_cannot_use_boldness_as_a_signal() -> None:
    bold = Span(text="all bold text", bbox=(0, 0, 60, 12), font="Arial-Bold", size=11.0, flags=1 << 4)
    statistics = measure(document_of(TextLine(1, bold.bbox, (bold,))))
    assert statistics.mostly_bold


# -- rhythm -----------------------------------------------------------------

def test_paragraph_gap_is_measured_from_the_documents_own_leading() -> None:
    statistics = measure(
        document_of(*[line(f"body line {index}", index * 14.0) for index in range(6)])
    )
    assert statistics.median_line_gap > 0
    assert statistics.is_paragraph_gap(statistics.median_line_gap)
    assert not statistics.is_paragraph_gap(statistics.median_line_gap * 10)


def test_indentation_levels_cluster_the_lefts_actually_used() -> None:
    statistics = measure(
        document_of(
            line("flush left", 0, left=72.0),
            line("flush left again", 20, left=72.4),   # same level, sub-point drift
            line("indented bullet", 40, left=100.0),
        )
    )
    assert len(statistics.indentation_levels) == 2
    assert statistics.indentation_level(72.2) == 0
    assert statistics.indentation_level(100.0) == 1


# -- page furniture ---------------------------------------------------------


def test_running_footers_are_matched_despite_the_page_number() -> None:
    """The literal text differs on every page; the fixed regex missed that."""
    banner_top = 700.0
    statistics = measure(
        document_of(
            line("Confidential draft | Page 1", banner_top, page=1),
            line("Confidential draft | Page 2", banner_top, page=2),
            line("Confidential draft | Page 3", banner_top, page=3),
            pages=3,
        )
    )
    assert statistics.is_page_furniture(
        "Confidential draft | Page 2", top=banner_top, bottom=banner_top + 11.0,
        page_height=792.0,
    )
    assert not statistics.is_page_furniture(
        "Senior Site Engineer", top=banner_top, bottom=banner_top + 11.0,
        page_height=792.0,
    )


def test_a_bare_year_in_a_table_is_not_page_furniture() -> None:
    """Regression: 7.anomaly's certification table has a Year column, and after
    digit folding a bare "2022" is the same key as a bare page number. Found by
    rendering the pass-1 overlay, where five table cells were flagged red."""
    banner_top = 700.0
    statistics = measure(
        document_of(
            line("Confidential draft | Page 1", banner_top, page=1),
            line("Confidential draft | Page 2", banner_top, page=2),
            line("2022", 400.0, page=2),
            line("2023", 420.0, page=2),
            pages=2,
        )
    )
    assert not statistics.is_page_furniture(
        "2022", top=400.0, bottom=411.0, page_height=792.0
    )


def test_repeated_lines_count_pages_not_occurrences() -> None:
    """Five alike lines on one page are not a running footer."""
    statistics = measure(
        document_of(
            *[line("2020", 700.0 + index, page=1) for index in range(5)],
            line("real content", 100.0, page=2),
            pages=2,
        )
    )
    assert not statistics.repeated_footers


def test_a_single_page_document_has_no_running_furniture() -> None:
    """Nothing repeats on one page, so nothing may be discarded as furniture."""
    statistics = measure(document_of(line("Some line", 700.0)))
    assert not statistics.repeated_headers and not statistics.repeated_footers


# -- against the real fixtures ---------------------------------------------


@pytest.mark.parametrize("stem", ["1", "2", "6", "7.anomaly"])
def test_measurements_are_plausible_on_real_resumes(stem: str) -> None:
    statistics = measure(read_document(pymupdf.open(SYNTHETIC_DIRECTORY / f"{stem}.pdf")))
    assert 6.0 <= statistics.body_font_size <= 16.0
    assert statistics.median_line_height > 0
    assert statistics.median_character_width > 0
    assert 0 < statistics.left_margin < statistics.right_margin
    assert statistics.indentation_levels


def test_the_repeated_banner_in_the_anomaly_fixture_is_found() -> None:
    """7.anomaly repeats a banner on all three pages, numbered per page -- and
    nothing else on the page may be mistaken for it."""
    document = read_document(pymupdf.open(SYNTHETIC_DIRECTORY / "7.anomaly.pdf"))
    statistics = measure(document)
    page = document.page(2)
    flagged = [
        line
        for line in page.lines
        if statistics.is_page_furniture(
            line.text, top=line.bbox[1], bottom=line.bbox[3], page_height=page.height
        )
    ]
    assert len(flagged) == 1
    assert flagged[0].text.startswith("SYNTHETIC RESUME")


def test_separator_rules_are_measured() -> None:
    statistics = measure(read_document(pymupdf.open(SYNTHETIC_DIRECTORY / "6.pdf")))
    assert statistics.horizontal_rules
