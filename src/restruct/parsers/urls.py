"""URL evidence: link annotations first, then visible text.

A PDF link annotation carries the real target, which the visible text often
abbreviates, so annotations are claimed before the regex fills in the rest.
Both the visible text and the annotation target are preserved.
"""

from __future__ import annotations

from typing import Any

import pymupdf

from restruct.configs import SETTINGS
from restruct.document.types import ExtractedLine, HeaderEntityMatch, append_regex_matches
from restruct.geometry import resolve_span_box, rounded
from restruct.patterns.contacts import URL_RE


def _annotation_text_span(
    line: ExtractedLine,
    annotation_rectangle: pymupdf.Rect,
    page_words: list[tuple[Any, ...]],
) -> tuple[str, int, int] | None:
    """Map the visible words under a PDF link annotation back to one line."""
    line_rectangle = pymupdf.Rect(line.bbox)
    selected_words = [
        str(word[4])
        for word in page_words
        if len(word) >= 5
        and pymupdf.Rect(word[:4]).intersects(annotation_rectangle)
        and pymupdf.Rect(word[:4]).intersects(line_rectangle)
    ]
    if selected_words:
        folded_line = line.text.casefold()
        cursor = 0
        positions: list[tuple[int, int]] = []
        for word in selected_words:
            position = folded_line.find(word.casefold(), cursor)
            if position < 0:
                position = folded_line.find(word.casefold())
            if position < 0:
                continue
            positions.append((position, position + len(word)))
            cursor = position + len(word)
        if positions:
            start = min(position[0] for position in positions)
            end = max(position[1] for position in positions)
            return line.text[start:end], start, end

    intersection = line_rectangle & annotation_rectangle
    if intersection.is_empty or line_rectangle.width <= 0 or not line.text:
        return None
    start = round(
        len(line.text)
        * max(0.0, intersection.x0 - line_rectangle.x0)
        / line_rectangle.width
    )
    end = round(
        len(line.text)
        * min(line_rectangle.width, intersection.x1 - line_rectangle.x0)
        / line_rectangle.width
    )
    raw_text = line.text[start:end]
    visible_text = raw_text.strip()
    if not visible_text:
        return None
    start += len(raw_text) - len(raw_text.lstrip())
    return visible_text, start, start + len(visible_text)

def _annotation_url_matches(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    line_indexes: list[int],
) -> list[HeaderEntityMatch]:
    """Match PDF link rectangles to visible lines with a small bbox tolerance."""
    indexes_by_page: dict[int, list[int]] = {}
    for line_index in line_indexes:
        indexes_by_page.setdefault(lines[line_index].page, []).append(line_index)

    matches: list[HeaderEntityMatch] = []
    tolerance = SETTINGS.url.annotation_bbox_tolerance
    for page_number, page_line_indexes in indexes_by_page.items():
        page = document[page_number - 1]
        page_words = list(page.get_text("words", sort=True))
        for link in page.get_links():
            url = str(link.get("uri") or "").strip()
            if not url or url.casefold().startswith(("mailto:", "tel:")):
                continue
            raw_rectangle = link.get("from")
            if raw_rectangle is None:
                continue
            annotation_rectangle = pymupdf.Rect(raw_rectangle)
            matching_rectangle = pymupdf.Rect(
                annotation_rectangle.x0 - tolerance,
                annotation_rectangle.y0 - tolerance,
                annotation_rectangle.x1 + tolerance,
                annotation_rectangle.y1 + tolerance,
            )
            exact_candidates: list[tuple[int, float]] = []
            tolerant_candidates: list[tuple[int, float]] = []
            for line_index in page_line_indexes:
                line = lines[line_index]
                line_rectangle = pymupdf.Rect(line.bbox)
                center_distance = abs(
                    (line_rectangle.y0 + line_rectangle.y1)
                    - (annotation_rectangle.y0 + annotation_rectangle.y1)
                )
                if line_rectangle.intersects(annotation_rectangle):
                    exact_candidates.append((line_index, center_distance))
                elif line_rectangle.intersects(matching_rectangle):
                    tolerant_candidates.append((line_index, center_distance))

            candidates = sorted(
                exact_candidates or tolerant_candidates,
                key=lambda candidate: candidate[1],
            )[:1]
            for line_index, _ in candidates:
                line = lines[line_index]
                line_rectangle = pymupdf.Rect(line.bbox)
                span = _annotation_text_span(
                    line,
                    matching_rectangle,
                    page_words,
                )
                if span is None:
                    continue
                text, start, end = span
                matches.append(
                    HeaderEntityMatch(
                        kind="url",
                        text=text,
                        line_index=line_index,
                        start=start,
                        end=end,
                        detection_method="pdf_annotation",
                        url=url,
                        bbox=tuple(
                            float(value)
                            for value in (
                                line_rectangle
                                & (
                                    annotation_rectangle
                                    if line_rectangle.intersects(annotation_rectangle)
                                    else matching_rectangle
                                )
                            )
                        ),
                    )
                )
    return matches

def _url_matches_for_lines(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    line_indexes: list[int],
) -> list[HeaderEntityMatch]:
    """Prefer annotation targets, then fill remaining visible URLs with regex."""
    matches = _annotation_url_matches(document, lines, line_indexes)
    for line_index in line_indexes:
        append_regex_matches(
            matches,
            line_index=line_index,
            text=lines[line_index].text,
            kind="url",
            pattern=URL_RE,
        )
    return matches

def _url_entity_value(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    match: HeaderEntityMatch,
) -> dict[str, Any]:
    """Serialize one URL match consistently for headers and later sections."""
    line = lines[match.line_index]
    return {
        "type": "url",
        "text": match.text,
        "url": match.url or match.text,
        "page": line.page,
        "bbox": (
            rounded(match.bbox)
            if match.bbox is not None
            else resolve_span_box(
                document,
                line,
                match.text,
                match.start,
                match.end,
            )
        ),
        "detectionMethod": match.detection_method,
    }
