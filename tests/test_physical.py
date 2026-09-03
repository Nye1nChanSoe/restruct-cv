"""Guards on the pass-1 physical representation.

These check the two properties the whole design rests on: that one read gives
everything later passes need, and that the native and OCR paths produce the
same types.
"""

from __future__ import annotations

import pymupdf
import pytest

from restruct.document.physical import Document, Page, Span, TextLine, Token
from restruct.ingestion.native import extracted_lines, read_document
from tests.helpers import SYNTHETIC_DIRECTORY, tesseract_available


def span(text: str, *, font: str = "Helvetica", size: float = 11.0, flags: int = 0) -> Span:
    return Span(text=text, bbox=(0, 0, 10 * len(text), 12), font=font, size=size, flags=flags)


# -- style detection --------------------------------------------------------


def test_bold_is_detected_from_the_flag() -> None:
    assert span("x", flags=1 << 4).bold


@pytest.mark.parametrize("font", ["Arial-Bold", "Helvetica-Black", "opensans-bold"])
def test_bold_is_detected_from_the_font_name(font: str) -> None:
    """Some exporters encode weight only in the name and never set the flag."""
    assert span("x", font=font, flags=0).bold


def test_regular_text_is_not_bold() -> None:
    assert not span("x", font="Helvetica", flags=0).bold


def test_character_width_is_measured_not_assumed() -> None:
    """Pass 2 needs spacing relative to the actual font, not a constant."""
    assert span("abcde").character_width == pytest.approx(10.0)
    assert span("").character_width == 0.0


# -- line properties --------------------------------------------------------


def test_line_size_is_the_largest_span() -> None:
    """A line reads as a heading on its largest run, not its average."""
    line = TextLine(1, (0, 0, 100, 12), (span("small", size=9), span("BIG", size=18)))
    assert line.size == 18


def test_line_is_bold_if_any_span_is() -> None:
    line = TextLine(1, (0, 0, 100, 12), (span("a"), span("b", flags=1 << 4)))
    assert line.bold


def test_vertical_text_is_recognised_as_not_horizontal() -> None:
    """Vertical writing is out of scope and must be detectable, not parsed."""
    assert TextLine(1, (0, 0, 10, 100), (span("x"),), direction=(0.0, 1.0)).is_horizontal is False
    assert TextLine(1, (0, 0, 100, 10), (span("x"),)).is_horizontal is True


# -- document container -----------------------------------------------------


def test_document_flattens_lines_in_page_order() -> None:
    document = Document(
        pages=(
            Page(1, 612, 792, lines=(TextLine(1, (0, 0, 1, 1), (span("first"),)),)),
            Page(2, 612, 792, lines=(TextLine(2, (0, 0, 1, 1), (span("second"),)),)),
        )
    )
    assert [line.text for line in document.lines] == ["first", "second"]
    assert document.page(2).number == 2
    assert document.used_ocr is False


# -- against the real fixtures ---------------------------------------------


@pytest.mark.parametrize("stem", ["1", "2", "6", "7.anomaly"])
def test_native_pages_carry_character_geometry(stem: str) -> None:
    """One read must yield per-character boxes; pass 2 has no second chance."""
    document = read_document(pymupdf.open(SYNTHETIC_DIRECTORY / f"{stem}.pdf"))
    spans = [s for line in document.lines for s in line.spans]
    assert spans, stem
    assert all(s.granularity == "character" for s in spans)
    tokens = [token for s in spans for token in s.tokens]
    assert len(tokens) > 500, f"{stem}: only {len(tokens)} character boxes"
    assert all(len(token.text) <= 1 or token.synthetic for token in tokens[:200])


@pytest.mark.parametrize("stem", ["1", "2", "6", "7.anomaly"])
def test_span_text_equals_its_characters(stem: str) -> None:
    """rawdict replaced dict as the single read; the text must not drift."""
    document = read_document(pymupdf.open(SYNTHETIC_DIRECTORY / f"{stem}.pdf"))
    for line in document.lines:
        for item in line.spans:
            assert item.text == "".join(token.text for token in item.tokens)


def test_drawn_rules_are_captured() -> None:
    """Separator lines are layout evidence pass 4 needs; nothing else records them."""
    document = read_document(pymupdf.open(SYNTHETIC_DIRECTORY / "6.pdf"))
    rules = [rule for page in document.pages for rule in page.horizontal_rules]
    assert rules
    assert all(rule.orientation == "horizontal" and rule.length > 10 for rule in rules)


def test_scanned_pages_are_recorded_as_images() -> None:
    document = read_document(pymupdf.open(SYNTHETIC_DIRECTORY / "9.ocr.pdf"))
    assert document.pages[0].images


def test_page_geometry_is_recorded() -> None:
    document = read_document(pymupdf.open(SYNTHETIC_DIRECTORY / "1.pdf"))
    page = document.page(1)
    assert page.width > 0 and page.height > 0
    assert page.rotation == 0


@pytest.mark.skipif(not tesseract_available(), reason="needs tesseract")
def test_ocr_converges_on_the_same_types() -> None:
    """Downstream code must not be able to tell which path produced a page."""
    document = read_document(pymupdf.open(SYNTHETIC_DIRECTORY / "9.ocr.pdf"))
    assert document.used_ocr
    lines = document.lines
    assert lines and all(isinstance(line, TextLine) for line in lines)

    spans = [s for line in lines for s in line.spans]
    # The one declared difference: OCR resolves to words, not characters.
    assert all(s.granularity == "word" for s in spans)
    assert all(isinstance(token, Token) for s in spans for token in s.tokens)
    # And OCR alone reports confidence.
    assert any(token.confidence is not None for s in spans for token in s.tokens)


@pytest.mark.skipif(not tesseract_available(), reason="needs tesseract")
def test_ocr_line_size_resists_a_single_tall_glyph() -> None:
    """Median word height, not maximum: one tall glyph must not make a heading."""
    document = read_document(pymupdf.open(SYNTHETIC_DIRECTORY / "9.ocr.pdf"))
    for line in document.lines:
        for item in line.spans:
            heights = [token.bbox[3] - token.bbox[1] for token in item.tokens]
            if len(heights) > 2:
                assert item.size <= max(heights)


@pytest.mark.parametrize("stem", ["1", "6"])
def test_extracted_lines_bridge_drops_only_blank_lines(stem: str) -> None:
    """The bridge feeding the existing parsers must not lose content."""
    document = read_document(pymupdf.open(SYNTHETIC_DIRECTORY / f"{stem}.pdf"))
    bridged = extracted_lines(document)
    non_blank = [line for line in document.lines if line.text.strip()]
    assert len(bridged) == len(non_blank)
    assert [line.text for line in bridged] == [line.text.strip() for line in non_blank]
