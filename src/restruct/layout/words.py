"""Pass 2: reconstruct words from characters or OCR tokens.

The rule is deliberately not one spacing constant. A PDF may separate words with
a space glyph, or with nothing but an advance; the same absolute gap means
different things at 8pt and at 14pt; and a font change mid-run is normal inside
a single word. So the decision uses several kinds of evidence, each measured
against the current line, font and document rather than against a magic number.

Every word keeps the tokens it was built from, so a later pass can re-examine or
undo the grouping -- the same principle that keeps character offsets on a
claimed span.
"""

from __future__ import annotations

from dataclasses import replace

import pymupdf

from restruct.configs import SETTINGS
from restruct.document.physical import (
    Document,
    LinkAnnotation,
    Page,
    Span,
    TextLine,
    Token,
    Word,
)
from restruct.document.stats import DocumentStatistics
from restruct.geometry import union

# A baseline shift beyond this fraction of the font size is a superscript or a
# different line, not a continuation of the current word.
_BASELINE_TOLERANCE = 0.3

# A style change only breaks a word if the glyphs are also visibly apart. Fonts
# switch mid-word routinely -- "Node.js" can arrive as two spans -- so style
# alone is weak evidence.
_STYLE_CHANGE_GAP_RATIO = 0.25

# Share of a word's box that must fall inside a link rectangle to claim its URL.
_LINK_OVERLAP_RATIO = 0.5


def _breaks_word(
    previous: Token,
    current: Token,
    previous_span: Span,
    current_span: Span,
    statistics: DocumentStatistics,
) -> bool:
    """Whether ``current`` starts a new word rather than continuing one."""
    gap = current.bbox[0] - previous.bbox[2]

    # Overlapping boxes are kerning, which is the strongest evidence of one word.
    if gap < 0:
        return False

    # A different baseline means a superscript or a stray glyph.
    if previous.origin and current.origin:
        reference_size = current_span.size or statistics.body_font_size
        if reference_size > 0:
            shift = abs(current.origin[1] - previous.origin[1])
            if shift > reference_size * _BASELINE_TOLERANCE:
                return True

    # Prefer this span's own measured advance over the document median, so a
    # heading set in a wide face is judged on its own metrics.
    threshold = statistics.word_gap_threshold
    span_width = current_span.character_width
    if span_width > 0:
        threshold = min(threshold, span_width * 0.5) if threshold > 0 else span_width * 0.5
    if threshold > 0 and gap >= threshold:
        return True

    style_changed = (
        previous_span.font != current_span.font
        or abs(previous_span.size - current_span.size) > 0.1
        or previous_span.bold != current_span.bold
    )
    if style_changed and span_width > 0 and gap >= span_width * _STYLE_CHANGE_GAP_RATIO:
        return True
    return False


def _word_from_tokens(
    tokens: list[Token],
    span: Span,
    page_number: int,
    *,
    used_ocr: bool,
) -> Word | None:
    text = "".join(token.text for token in tokens)
    if not text.strip():
        return None
    return Word(
        text=text,
        page=page_number,
        bbox=tuple(float(value) for value in union(token.bbox for token in tokens)),
        tokens=tuple(tokens),
        font=span.font,
        size=span.size,
        flags=span.flags,
        source="character",
        used_ocr=used_ocr,
    )


def _ocr_words(line: TextLine) -> list[Word]:
    """Words the source already grouped, for the OCR path."""
    words: list[Word] = []
    for span in line.spans:
        for token in span.tokens:
            if not token.text.strip():
                continue
            words.append(
                Word(
                    text=token.text,
                    page=line.page,
                    bbox=token.bbox,
                    tokens=(token,),
                    font=span.font,
                    size=span.size,
                    flags=span.flags,
                    source="word",
                    used_ocr=True,
                    confidence=token.confidence,
                )
            )
    return words


def words_in_line(line: TextLine, statistics: DocumentStatistics) -> tuple[Word, ...]:
    """Group one line's tokens into words.

    Vertical text is out of v1 scope, so a non-horizontal line yields nothing
    rather than a confidently wrong reading order.
    """
    if not line.is_horizontal:
        return ()

    # OCR resolves to words already; regrouping them would only lose fidelity.
    if any(span.granularity == "word" for span in line.spans):
        return tuple(_ocr_words(line))

    words: list[Word] = []
    buffer: list[Token] = []
    buffer_span: Span | None = None
    previous_token: Token | None = None
    previous_span: Span | None = None

    def flush() -> None:
        nonlocal buffer, buffer_span
        if buffer and buffer_span is not None:
            word = _word_from_tokens(buffer, buffer_span, line.page, used_ocr=line.used_ocr)
            if word is not None:
                words.append(word)
        buffer, buffer_span = [], None

    for span in line.spans:
        for token in span.tokens:
            # An explicit space glyph is the clearest break there is, and is how
            # most exporters mark one. It never becomes part of a word.
            if token.text.isspace() or not token.text:
                flush()
                previous_token, previous_span = None, None
                continue
            if (
                previous_token is not None
                and previous_span is not None
                and _breaks_word(previous_token, token, previous_span, span, statistics)
            ):
                flush()
            if not buffer:
                buffer_span = span
            buffer.append(token)
            previous_token, previous_span = token, span
    flush()
    return tuple(words)


def _link_for(word: Word, links: tuple[LinkAnnotation, ...]) -> str | None:
    """The link annotation covering this word, if one does.

    Matched by geometry rather than by text, because the visible text and the
    annotation target routinely disagree.
    """
    if not links:
        return None
    box = pymupdf.Rect(word.bbox)
    if box.is_empty:
        return None
    tolerance = SETTINGS.url.annotation_bbox_tolerance
    for link in links:
        rectangle = pymupdf.Rect(link.bbox)
        overlap = box & pymupdf.Rect(
            rectangle.x0 - tolerance,
            rectangle.y0 - tolerance,
            rectangle.x1 + tolerance,
            rectangle.y1 + tolerance,
        )
        if overlap.is_empty or overlap.get_area() <= 0:
            continue
        if overlap.get_area() >= box.get_area() * _LINK_OVERLAP_RATIO:
            return link.uri
    return None


def reconstruct_words(document: Document, statistics: DocumentStatistics) -> Document:
    """Return the document with every line's words filled in.

    Words are attached to the shared representation rather than recomputed per
    section, so the grouping cost is paid once per document.
    """
    pages: list[Page] = []
    for page in document.pages:
        lines = tuple(
            replace(
                line,
                words=tuple(
                    replace(word, url=_link_for(word, page.links))
                    for word in words_in_line(line, statistics)
                ),
            )
            for line in page.lines
        )
        pages.append(replace(page, lines=lines))
    return replace(document, pages=tuple(pages))
