"""Guards on pass-2 word reconstruction.

The property under test is that grouping uses *measured* evidence rather than a
spacing constant: the same layout must reconstruct identically whether it is set
at 8pt or 16pt, and a PDF that emits no space glyphs must still yield words.
"""

from __future__ import annotations

import pymupdf
import pytest

from restruct.document.physical import Document, Page, Span, TextLine, Token
from restruct.document.stats import measure
from restruct.ingestion.native import read_document
from restruct.layout.words import reconstruct_words, words_in_line
from tests.helpers import SYNTHETIC_DIRECTORY, tesseract_available


def token(text: str, left: float, width: float, *, baseline: float = 20.0) -> Token:
    return Token(text=text, bbox=(left, 10.0, left + width, 22.0), origin=(left, baseline))


def line_of(*tokens: Token, font: str = "Helvetica", size: float = 10.0, flags: int = 0) -> TextLine:
    span = Span(
        text="".join(t.text for t in tokens),
        bbox=(tokens[0].bbox[0], 10.0, tokens[-1].bbox[2], 22.0),
        font=font,
        size=size,
        flags=flags,
        tokens=tokens,
    )
    return TextLine(1, span.bbox, (span,))


def statistics_for(*lines: TextLine):
    return measure(Document(pages=(Page(1, 612.0, 792.0, lines=lines),)))


def glyphs(text: str, *, start: float = 0.0, advance: float = 5.0) -> list[Token]:
    """Tokens tiling contiguously, the way PyMuPDF reports real glyph boxes."""
    tokens, left = [], start
    for character in text:
        tokens.append(token(character, left, advance))
        left += advance
    return tokens


# -- the primary signal -----------------------------------------------------


def test_a_space_glyph_splits_words_and_is_not_kept() -> None:
    line = line_of(*glyphs("ab cd"))
    words = words_in_line(line, statistics_for(line))
    assert [word.text for word in words] == ["ab", "cd"]


def test_kerning_never_splits_a_word() -> None:
    """Overlapping boxes are the strongest evidence of a single word."""
    tokens = [token("A", 0, 5), token("V", 4.5, 5)]      # boxes overlap
    line = line_of(*tokens)
    assert [w.text for w in words_in_line(line, statistics_for(line))] == ["AV"]


def test_a_wide_gap_splits_words_without_any_space_glyph() -> None:
    """Some exporters position words by advance and emit no spaces at all."""
    reference = line_of(*glyphs("reference text for metrics"))
    tokens = [*glyphs("ab"), token("c", 40.0, 5), token("d", 45.0, 5)]
    line = line_of(*tokens)
    words = words_in_line(line, statistics_for(reference, line))
    assert [word.text for word in words] == ["ab", "cd"]


def test_a_baseline_shift_splits_a_superscript_off() -> None:
    tokens = [*glyphs("x"), token("2", 5.0, 5, baseline=14.0)]
    line = line_of(*tokens)
    assert [w.text for w in words_in_line(line, statistics_for(line))] == ["x", "2"]


# -- style is weak evidence -------------------------------------------------


def test_a_font_change_alone_does_not_split_a_word() -> None:
    """Fonts switch mid-word routinely; splitting on that would break names."""
    first = Span(text="Node", bbox=(0, 10, 20, 22), font="Helvetica", size=10.0,
                 flags=0, tokens=tuple(glyphs("Node")))
    second = Span(text=".js", bbox=(20, 10, 35, 22), font="Courier", size=10.0,
                  flags=0, tokens=tuple(glyphs(".js", start=20.0)))
    line = TextLine(1, (0, 10, 35, 22), (first, second))
    assert [w.text for w in words_in_line(line, statistics_for(line))] == ["Node.js"]


def test_a_style_change_with_a_visible_gap_does_split() -> None:
    reference = line_of(*glyphs("reference text for metrics"))
    first = Span(text="Label", bbox=(0, 10, 25, 22), font="Arial-Bold", size=10.0,
                 flags=1 << 4, tokens=tuple(glyphs("Label")))
    second = Span(text="value", bbox=(27, 10, 52, 22), font="Helvetica", size=10.0,
                  flags=0, tokens=tuple(glyphs("value", start=27.0)))
    line = TextLine(1, (0, 10, 52, 22), (first, second))
    words = words_in_line(line, statistics_for(reference, line))
    assert [word.text for word in words] == ["Label", "value"]


# -- relativity -------------------------------------------------------------


@pytest.mark.parametrize("scale", [0.8, 1.0, 1.6])
def test_the_same_layout_reconstructs_identically_at_any_size(scale: float) -> None:
    """A fixed pixel threshold would fail this; a measured one must not."""
    tokens, left = [], 0.0
    for character in "hello world again":
        tokens.append(token(character, left, 5.0 * scale))
        left += 5.0 * scale
    line = line_of(*tokens, size=10.0 * scale)
    words = words_in_line(line, statistics_for(line))
    assert [word.text for word in words] == ["hello", "world", "again"]


# -- provenance -------------------------------------------------------------


def test_a_word_keeps_the_tokens_it_was_built_from() -> None:
    """Grouping must stay reversible, like a claimed span's offsets."""
    line = line_of(*glyphs("abc de"))
    first = words_in_line(line, statistics_for(line))[0]
    assert "".join(t.text for t in first.tokens) == first.text
    assert len(first.tokens) == 3


def test_words_inherit_style_from_their_span() -> None:
    line = line_of(*glyphs("Bold"), font="Arial-Bold", flags=1 << 4)
    word = words_in_line(line, statistics_for(line))[0]
    assert word.bold and word.size == 10.0


def test_vertical_text_yields_no_words() -> None:
    """Out of v1 scope: better nothing than a confidently wrong reading order."""
    span = Span(text="abc", bbox=(0, 0, 10, 40), font="Helvetica", size=10.0,
                flags=0, tokens=tuple(glyphs("abc")))
    line = TextLine(1, (0, 0, 10, 40), (span,), direction=(0.0, 1.0))
    assert words_in_line(line, statistics_for(line)) == ()


# -- against the real fixtures ---------------------------------------------


@pytest.mark.parametrize("stem", ["1", "2", "6", "7.anomaly"])
def test_reconstruction_neither_loses_nor_invents_characters(stem: str) -> None:
    document = read_document(pymupdf.open(SYNTHETIC_DIRECTORY / f"{stem}.pdf"))
    document = reconstruct_words(document, measure(document))
    for line in document.lines:
        from_words = "".join(word.text for word in line.words)
        from_spans = "".join(line.text.split())
        assert from_words == from_spans, line.text[:60]


def test_a_bullet_glyph_separates_from_the_word_after_it() -> None:
    """6.pdf sets its bullet in OpenSymbol with a real 3.3pt gap; pass 3 needs
    the marker as its own unit to recognise the bullet."""
    document = read_document(pymupdf.open(SYNTHETIC_DIRECTORY / "6.pdf"))
    document = reconstruct_words(document, measure(document))
    bulleted = [
        line for line in document.lines
        if line.words and line.words[0].text == ""
    ]
    assert bulleted
    assert bulleted[0].words[1].text.isalpha()


@pytest.mark.skipif(not tesseract_available(), reason="needs tesseract")
def test_ocr_words_pass_through_with_their_confidence() -> None:
    """Tesseract already grouped these; regrouping would only lose fidelity."""
    document = read_document(pymupdf.open(SYNTHETIC_DIRECTORY / "9.ocr.pdf"))
    document = reconstruct_words(document, measure(document))
    words = [word for line in document.lines for word in line.words]
    assert words
    assert all(word.source == "word" and word.used_ocr for word in words)
    assert any(word.confidence is not None for word in words)
    assert not any(" " in word.text for word in words)
