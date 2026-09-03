"""Visual debug for passes 1 to 3.

These passes reconstruct physical structure, and their output is geometry. A
JSON dump of it is close to unreadable, and a numeric test can confirm a count
without revealing that every box is ten points too low. So passes 1-3 render
images and no JSON, which is the split the design brief asks for.

Each overlay carries a legend, so an image says what its colours mean without
anyone having to read this file.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from restruct.debug import canvas
from restruct.debug.colors import LINE_STYLES, PHYSICAL_STYLES, WORD_STYLES
from restruct.document.physical import Document, Page
from restruct.document.stats import DocumentStatistics


def render_physical(
    pdf_document: pymupdf.Document,
    document: Document,
    statistics: DocumentStatistics,
    output_directory: Path,
) -> None:
    """Pass 1: what the page physically contains.

    Spans, drawn rules, image regions, link rectangles, and the lines the
    statistics identified as running headers or footers.
    """
    for page in document.pages:
        image, draw = canvas.page_canvas(pdf_document, page.number)
        counts = dict.fromkeys(PHYSICAL_STYLES, 0)

        for line in page.lines:
            furniture = statistics.is_page_furniture(
                line.text,
                top=line.bbox[1],
                bottom=line.bbox[3],
                page_height=page.height,
            )
            style = PHYSICAL_STYLES["furniture" if furniture else "span"]
            counts["furniture" if furniture else "span"] += 1
            for span in line.spans:
                canvas.outline(draw, span.bbox, style, width=1)
            if furniture:
                canvas.outline(draw, line.bbox, style, width=3, label="furniture")

        for rule in page.rules:
            canvas.outline(draw, rule.bbox, PHYSICAL_STYLES["rule"], width=3)
            counts["rule"] += 1
        for region in page.images:
            canvas.outline(draw, region.bbox, PHYSICAL_STYLES["image"], width=3, label="image")
            counts["image"] += 1
        for link in page.links:
            canvas.outline(draw, link.bbox, PHYSICAL_STYLES["link"], width=3, label="link")
            counts["link"] += 1

        canvas.legend(
            draw,
            [(PHYSICAL_STYLES[name], count) for name, count in counts.items() if count],
            title=f"pass 1 - physical  |  page {page.number}"
            + ("  (OCR)" if page.used_ocr else ""),
        )
        canvas.save(image, output_directory, page.number)


def render_words(
    pdf_document: pymupdf.Document,
    document: Document,
    output_directory: Path,
) -> None:
    """Pass 2: the reconstructed words.

    Native and OCR words are coloured differently, because they were grouped by
    different evidence and a reader should be able to tell which.
    """
    for page in document.pages:
        image, draw = canvas.page_canvas(pdf_document, page.number)
        counts = dict.fromkeys(WORD_STYLES, 0)

        for line in page.lines:
            for word in line.words:
                if word.url:
                    name = "linked_word"
                elif word.used_ocr:
                    name = "ocr_word"
                else:
                    name = "word"
                counts[name] += 1
                canvas.outline(draw, word.bbox, WORD_STYLES[name], width=1)

        canvas.legend(
            draw,
            [(WORD_STYLES[name], count) for name, count in counts.items() if count],
            title=f"pass 2 - words  |  page {page.number}",
        )
        canvas.save(image, output_directory, page.number)


def _draw_line_layer(draw, page: Page, statistics: DocumentStatistics) -> dict[str, int]:
    """Lines, their baselines, and the cells a wide gap divided them into."""
    from restruct.layout.lines import cells_in_line

    counts = dict.fromkeys(LINE_STYLES, 0)
    for line in page.lines:
        if not line.text.strip():
            continue
        canvas.outline(draw, line.bbox, LINE_STYLES["line"], width=1)
        counts["line"] += 1

        # The baseline is what row grouping compares, so draw it explicitly:
        # a row that groups wrongly is obvious the moment you see two baselines
        # that should coincide and do not.
        baseline = _baseline_of(line)
        canvas.rule_line(
            draw,
            (line.bbox[0], line.bbox[1], line.bbox[2], baseline),
            LINE_STYLES["baseline"],
            width=1,
        )
        counts["baseline"] += 1

        cells = cells_in_line(line, statistics)
        if len(cells) > 1:
            for index, cell in enumerate(cells):
                canvas.outline(
                    draw,
                    cell.bbox,
                    LINE_STYLES["cell"],
                    width=3,
                    label=f"cell {index + 1}/{len(cells)}",
                )
                counts["cell"] += 1
    return counts


def _baseline_of(line) -> float:
    from restruct.layout.lines import line_baseline

    return line_baseline(line)


def render_lines(
    pdf_document: pymupdf.Document,
    document: Document,
    statistics: DocumentStatistics,
    output_directory: Path,
) -> None:
    """Pass 3: lines, baselines and cells."""
    for page in document.pages:
        image, draw = canvas.page_canvas(pdf_document, page.number)
        counts = _draw_line_layer(draw, page, statistics)
        canvas.legend(
            draw,
            [(LINE_STYLES[name], count) for name, count in counts.items() if count],
            title=f"pass 3 - lines and cells  |  page {page.number}",
        )
        canvas.save(image, output_directory, page.number)


def render_stage_overlays(
    pdf_document: pymupdf.Document,
    document: Document,
    statistics: DocumentStatistics,
    debug_directory: Path,
    stages: tuple[int, ...] = (1, 2, 3),
) -> None:
    """Render the requested reconstruction passes as images."""
    if 1 in stages:
        render_physical(pdf_document, document, statistics, debug_directory / "pass-1-physical")
    if 2 in stages:
        render_words(pdf_document, document, debug_directory / "pass-2-words")
    if 3 in stages:
        render_lines(pdf_document, document, statistics, debug_directory / "pass-3-lines")
