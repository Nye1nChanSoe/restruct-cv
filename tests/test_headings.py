"""Guards on how a section measures its own body, and on what may be a subheading.

The bug behind both: a section whose blocks are set at two sizes measured its
body from the smaller one, so every line of the larger block read as "bigger
than body" and a wrapped bullet tail became a subheading. Two independent
guards were added, and each is tested here on its own -- either alone must
catch it, because the value of the pair is that neither has to be right.
"""

from __future__ import annotations

from restruct.document.physical import Document, Page, Span, TextLine
from restruct.document.stats import measure
from restruct.document.types import ExtractedLine
from restruct.structure.headings import _looks_like_subheading, _section_body_style


def line(
    text: str,
    *,
    top: float,
    left: float = 72.0,
    size: float = 11.0,
    bold: bool = False,
    page: int = 1,
) -> ExtractedLine:
    return ExtractedLine(
        page=page,
        text=text,
        bbox=(left, top, left + 5.0 * len(text), top + size),
        size=size,
        bold=bold,
        used_ocr=False,
    )


def statistics_for(*lines: ExtractedLine, height: float = 792.0):
    """Measure a document holding exactly these lines.

    The indent guard asks for indentation *levels*, which are clustered from
    the document's own left edges, so the statistics have to be measured from
    the same lines rather than stubbed.
    """
    spans = [
        (
            item,
            Span(
                text=item.text,
                bbox=item.bbox,
                font="Helvetica",
                size=item.size,
                flags=1 << 4 if item.bold else 0,
            ),
        )
        for item in lines
    ]
    by_page: dict[int, list[TextLine]] = {}
    for item, span in spans:
        by_page.setdefault(item.page, []).append(
            TextLine(item.page, item.bbox, (span,))
        )
    document = Document(
        pages=tuple(
            Page(number, 612.0, height, lines=tuple(page_lines))
            for number, page_lines in sorted(by_page.items())
        )
    )
    return measure(document)


# The shape of the real failure: one section, two projects, the first set at
# 13pt and the second at 11pt, each with a wrapped bullet. The wraps sit at a
# deeper indent than the bullets they continue.
def _mixed_size_section() -> list[ExtractedLine]:
    return [
        line("Open-LinkedOut", top=120.0, size=16.0, bold=True),
        line("Local-first job discovery and matching tool, runs on-device.", top=150.0, size=13.0),
        line("Single Node process: HTTP API, scheduler, three worker loops, live", top=280.0, size=13.0, left=90.0),
        line("SSE updates", top=303.0, size=13.0, left=108.0),
        line("Restruct-CV", top=337.0, size=16.0, bold=True),
        line("Local resume-to-JSON extractor, runs on-device.", top=366.0, size=13.0),
        line("Reads PDF, scanned PDF via OCR, DOCX and PNG into one", top=388.0, size=11.0, left=90.0),
        line("shared document model.", top=405.0, size=11.0, left=108.0),
        line("Writes one published JSON schema, sixteen sections in fixed order", top=422.0, size=11.0, left=90.0),
        line("means absent key.", top=439.0, size=11.0, left=108.0),
    ]


# -- 1a: the body-size estimate ---------------------------------------------


def test_a_wrapped_continuation_is_not_a_body_size_reference() -> None:
    """Regression: the wraps of the 11pt block were the only lines the prose
    filter accepted -- short, and ending in the full stop their parent's
    sentence happened to end with -- so they outvoted the 13pt block and the
    section measured its body as 11pt."""
    lines = _mixed_size_section()
    body_size, _, _ = _section_body_style(lines, statistics_for(*lines))
    assert body_size == 13.0


def test_the_body_size_still_comes_from_prose_where_there_is_prose() -> None:
    """The fragment exclusion must not empty the reference set on an ordinary
    single-size section."""
    lines = [
        line("EXPERIENCE", top=100.0, size=16.0, bold=True),
        line("Delivered custom web and backend solutions for small businesses.", top=130.0),
        line("Managed the full project lifecycle from gathering to deployment.", top=150.0),
    ]
    body_size, _, _ = _section_body_style(lines, statistics_for(*lines))
    assert body_size == 11.0


def test_the_section_indent_level_is_its_shallowest() -> None:
    lines = _mixed_size_section()
    _, _, body_indent_level = _section_body_style(lines, statistics_for(*lines))
    assert body_indent_level == 0


# -- 1b: the indent guard ---------------------------------------------------


def test_a_line_indented_past_the_whole_section_is_not_a_subheading() -> None:
    """The independent half. Given the *old*, wrong body size of 11pt, the
    wrapped tail still clears the size contrast -- so this guard alone has to
    reject it, on position, whatever the typography says."""
    lines = _mixed_size_section()
    statistics = statistics_for(*lines)
    wrapped_tail = next(item for item in lines if item.text == "SSE updates")
    assert not _looks_like_subheading(
        wrapped_tail,
        body_size=11.0,
        body_bold=False,
        body_indent_level=0,
        statistics=statistics,
    )
    # Without the guard, the same line and the same wrong body size accept it.
    assert _looks_like_subheading(
        wrapped_tail,
        body_size=11.0,
        body_bold=False,
        body_indent_level=9,
        statistics=statistics,
    )


def test_a_real_subheading_at_the_section_indent_is_still_accepted() -> None:
    """The guard must only ever remove candidates deeper than every other line
    in the section, or it would reject the headings it exists to find."""
    lines = _mixed_size_section()
    statistics = statistics_for(*lines)
    heading = next(item for item in lines if item.text == "Restruct-CV")
    assert _looks_like_subheading(
        heading,
        body_size=13.0,
        body_bold=False,
        body_indent_level=0,
        statistics=statistics,
    )


def test_a_subheading_indented_with_its_content_is_still_accepted() -> None:
    """A document may indent a whole labelled block. The guard compares against
    the section's own shallowest line, so an indented subheading above
    still-deeper content is unaffected."""
    lines = [
        line("Programming", top=130.0, size=13.0, bold=True, left=90.0),
        line("Go, Python, TypeScript, C++ (graphics)", top=150.0, left=108.0),
        line("Databases", top=170.0, size=13.0, bold=True, left=90.0),
        line("PostgreSQL, SQLite, Redis, pgvector", top=190.0, left=108.0),
    ]
    statistics = statistics_for(*lines)
    body_size, body_bold, body_indent_level = _section_body_style(lines, statistics)
    assert _looks_like_subheading(
        lines[0],
        body_size=body_size,
        body_bold=body_bold,
        body_indent_level=body_indent_level,
        statistics=statistics,
    )
    assert not _looks_like_subheading(
        lines[1],
        body_size=body_size,
        body_bold=body_bold,
        body_indent_level=body_indent_level,
        statistics=statistics,
    )
