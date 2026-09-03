"""Generic parser for the record-shaped supplementary sections.

Projects, certifications, awards and the rest share one shape: an optional
title, optional dates, and body text. Unrecognised sections land in ``others``
with their original heading preserved rather than being forced into a
destination they do not belong to.
"""
from __future__ import annotations

import re
from typing import Any

import pymupdf

from restruct.configs import SETTINGS
from restruct.document.stats import DocumentStatistics
from restruct.document.types import DetectedHeading, ExtractedLine
from restruct.geometry import resolve_span_box, rounded
from restruct.layout.blocks import (
    append_paragraph,
    continues_block,
    extend_block,
    last_block_text,
)
from restruct.layout.rows import _row_value, _visual_rows
from restruct.model import EmbeddingModel, classify_profile_attribute_labels
from restruct.patterns.bullets import BULLET_RE
from restruct.patterns.dates import DATE_RANGE_RE, SINGLE_YEAR_RE
from restruct.patterns.layout import PAGE_FOOTER_RE
from restruct.patterns.personal import ATTRIBUTE_INLINE_RE
from restruct.structure.compound import routed_logical_sections
from restruct.structure.separators import is_parenthetically_complete
from restruct.structure.headings import (
    _looks_like_subheading,
    _section_body_style,
)
from restruct.structure.keyvalue import _skill_inline_parts, _skill_subheading_value


_SUPPLEMENTARY_SECTION_TYPES = (
    "certifications",
    "licenses",
    "tools_equipment",
    "languages",
    "volunteering",
    "awards",
    "publications",
    "references",
    "interests",
    "others",
)

def _grouped_section_date_matches(text: str) -> list[re.Match[str]]:
    ranges = list(DATE_RANGE_RE.finditer(text))
    singles = [
        match
        for match in SINGLE_YEAR_RE.finditer(text)
        if not any(
            date_range.start() < match.end() and match.start() < date_range.end()
            for date_range in ranges
        )
    ]
    return sorted([*ranges, *singles], key=lambda match: match.start())

def _grouped_section_entry() -> dict[str, Any]:
    return {
        "detectionMethod": "geometry_reconstruction",
        "subheadingLines": [],
        "metadataRows": [],
        "dates": [],
        "urls": [],
        "attributes": [],
        "paragraphs": [],
        "bullets": [],
        "_lastType": None,
        "_lastLineBbox": None,
    }

def _build_grouped_section_debug(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
    url_entities_by_line: dict[int, list[dict[str, Any]]],
    section_type: str,
    statistics: DocumentStatistics,
    occurrence: int = 0,
    semantic_model: EmbeddingModel | None = None,
) -> dict[str, Any] | None:
    """Group titles, dates, paragraphs, bullets, and URLs for a minor section."""
    routed = routed_logical_sections(lines, headings)
    positions = [
        index
        for index, item in enumerate(routed)
        if item.section_type == section_type
    ]
    if occurrence >= len(positions):
        return None
    position = positions[occurrence]
    logical = routed[position]
    heading = logical.heading
    heading_line = lines[heading.line_index]
    # Explicit indexes rather than a range: a split compound heading gives its
    # destinations interleaved lines, not two contiguous halves.
    line_range = logical.line_indexes
    rows = _visual_rows(lines, line_range, statistics)
    body_size, body_bold = _section_body_style([lines[index] for index in line_range])
    titled_section = section_type in {
        "projects",
        "certifications",
        "licenses",
        "volunteering",
        "awards",
        "publications",
        "references",
    }
    entries: list[dict[str, Any]] = []
    visible_rows: list[list[tuple[int, ExtractedLine]]] = []
    current: dict[str, Any] | None = None

    for row in rows:
        if all(
            PAGE_FOOTER_RE.search(line.text) is not None
            and line.bbox[1] >= document[line.page - 1].rect.height * 0.88
            for _, line in row
        ):
            continue
        visible_rows.append(row)

        # Unknown sections frequently contain labelled groups such as
        # ``Availability: ...`` or a short left label paired with a value on
        # the right. Detect those groups before the generic paragraph fallback,
        # using the same precedence as the skills parser.
        if section_type == "others" and len(row) >= 2:
            left_index, left = row[0]
            right_cells = row[1:]
            left_is_label = (
                len(left.text.split()) <= 7
                and len(left.text) <= 50
                and (
                    left.bold
                    or _looks_like_subheading(
                        left,
                        body_size=body_size,
                        body_bold=body_bold,
                    )
                    or pymupdf.Rect(right_cells[0][1].bbox).x0
                    > pymupdf.Rect(left.bbox).x1 + left.size
                )
            )
            if left_is_label:
                stripped = left.text.strip().rstrip(":-–—").strip()
                attribute_type: str | None = None
                confidence = 0.0
                if semantic_model is not None:
                    attribute_type, confidence = classify_profile_attribute_labels(
                        semantic_model,
                        [stripped],
                    )[0]
                start = left.text.find(stripped)
                current = _grouped_section_entry()
                entries.append(current)
                current["subheadingLines"].append(
                    _skill_subheading_value(
                        document,
                        left,
                        stripped,
                        start,
                        start + len(stripped),
                        "geometry_table_cell",
                    )
                )
                left_urls = url_entities_by_line.get(left_index, [])
                current["urls"].extend(left_urls)
                current["metadataRows"].append(_row_value(row))
                for line_index, line in right_cells:
                    line_urls = url_entities_by_line.get(line_index, [])
                    current["urls"].extend(line_urls)
                    if attribute_type is not None:
                        current["attributes"].append({
                            "type": attribute_type,
                            "text": line.text,
                            "page": line.page,
                            "bbox": rounded(line.bbox),
                            "confidence": round(confidence, 4),
                            "detectionMethod": (
                                "label_pattern"
                                if confidence >= 0.9999
                                else "minilm_label"
                            ),
                        })
                    else:
                        append_paragraph(
                            current,
                            text=line.text,
                            page=line.page,
                            bbox=rounded(line.bbox),
                            entities=line_urls,
                            statistics=statistics,
                        )
                continue

        if section_type == "others":
            inline_parts: list[
                tuple[
                    int,
                    ExtractedLine,
                    re.Match[str] | None,
                    tuple[str, int, int, str, int, int, str],
                ]
            ] = []
            for line_index, line in row:
                bullet = BULLET_RE.match(line.text)
                content_start = bullet.end() if bullet else 0
                leading_space = len(line.text[content_start:]) - len(
                    line.text[content_start:].lstrip()
                )
                content_start += leading_space
                content = line.text[content_start:].strip()
                attribute_inline = ATTRIBUTE_INLINE_RE.match(content)
                inline = (
                    (
                        attribute_inline.group("label").strip(),
                        content_start + attribute_inline.start("label"),
                        content_start + attribute_inline.end("label"),
                        attribute_inline.group("body").strip(),
                        content_start + attribute_inline.start("body"),
                        content_start + attribute_inline.end("body"),
                        "profile_attribute_delimiter",
                    )
                    if attribute_inline is not None
                    else _skill_inline_parts(
                        content,
                        content_start,
                        allow_single_dash_body=bullet is None,
                    )
                )
                if inline is not None:
                    inline_parts.append((line_index, line, bullet, inline))

            if inline_parts:
                for line_index, line, bullet, inline in inline_parts:
                    (
                        label,
                        label_start,
                        label_end,
                        body,
                        body_start,
                        body_end,
                        method,
                    ) = inline
                    current = _grouped_section_entry()
                    entries.append(current)
                    current["subheadingLines"].append(
                        _skill_subheading_value(
                            document,
                            line,
                            label,
                            label_start,
                            label_end,
                            method,
                        )
                    )
                    current["metadataRows"].append(_row_value(row))
                    line_urls = url_entities_by_line.get(line_index, [])
                    body_value: dict[str, Any] = {
                        "text": body,
                        "page": line.page,
                        "bbox": resolve_span_box(
                            document,
                            line,
                            body,
                            body_start,
                            body_end,
                        ),
                        "detectionMethod": (
                            "bullet_marker" if bullet else "geometry_default"
                        ),
                    }
                    if line_urls:
                        body_value["entities"] = line_urls
                    current["urls"].extend(line_urls)
                    attribute_type: str | None = None
                    confidence = 0.0
                    if semantic_model is not None:
                        attribute_type, confidence = classify_profile_attribute_labels(
                            semantic_model,
                            [label],
                        )[0]
                        if attribute_type is not None:
                            current["attributes"].append({
                                "type": attribute_type,
                                "text": body,
                                "page": line.page,
                                "bbox": body_value["bbox"],
                                "confidence": round(confidence, 4),
                                "detectionMethod": (
                                    "label_pattern"
                                    if confidence >= 0.9999
                                    else "minilm_label"
                                ),
                            })
                    if attribute_type is None:
                        target = "bullets" if bullet else "paragraphs"
                        current[target].append(body_value)
                        current["_lastType"] = "bullet" if bullet else "paragraph"
                    current["_lastLineBbox"] = [
                        round(number, 2) for number in line.bbox
                    ]
                continue

        dates_by_line = {
            line_index: _grouped_section_date_matches(line.text)
            for line_index, line in row
        }
        row_has_date = any(dates_by_line.values())
        title_cells: list[tuple[int, ExtractedLine]] = []
        for line_index, line in row:
            bullet = BULLET_RE.match(line.text)
            matches = dates_by_line[line_index]
            date_only = bool(matches) and all(
                not character.strip(" ()[]{}|,;:-–—")
                for character in (
                    line.text[:matches[0].start()],
                    line.text[matches[-1].end():],
                )
            )
            if bullet or date_only:
                continue
            typographic_title = _looks_like_subheading(
                line,
                body_size=body_size,
                body_bold=body_bold,
            )
            row_title = row_has_date and len(row) >= 2
            first_title = (
                titled_section
                and
                current is None
                and len(line.text.split()) <= SETTINGS.section_router.maximum_subheading_words
                and not line.text.rstrip().endswith((".", ";", ":"))
            )
            if typographic_title or row_title or first_title:
                title_cells.append((line_index, line))

        if title_cells:
            should_continue_title = bool(
                current is not None
                and current["subheadingLines"]
                and not current["dates"]
                and not current["paragraphs"]
                and not current["bullets"]
                and title_cells[0][1].page == current["subheadingLines"][-1]["page"]
                and pymupdf.Rect(title_cells[0][1].bbox).y0
                - pymupdf.Rect(current["subheadingLines"][-1]["bbox"]).y1
                <= max(title_cells[0][1].size, 1.0) * 0.65
            )
            if not should_continue_title:
                current = _grouped_section_entry()
                entries.append(current)
            for line_index, line in title_cells:
                value: dict[str, Any] = {
                    "text": line.text,
                    "page": line.page,
                    "bbox": rounded(line.bbox),
                    "detectionMethod": "geometry_typography",
                }
                line_urls = url_entities_by_line.get(line_index, [])
                if line_urls:
                    value["entities"] = line_urls
                    current["urls"].extend(line_urls)
                current["subheadingLines"].append(value)
            current["metadataRows"].append(_row_value(row))

        if row_has_date:
            if current is None:
                current = _grouped_section_entry()
                entries.append(current)
            for line_index, line in row:
                for match in dates_by_line[line_index]:
                    current["dates"].append({
                        "text": match.group(0),
                        "page": line.page,
                        "bbox": resolve_span_box(
                            document,
                            line,
                            match.group(0),
                            match.start(),
                            match.end(),
                        ),
                        "detectionMethod": (
                            "date_regex" if DATE_RANGE_RE.fullmatch(match.group(0)) else "year_regex"
                        ),
                    })
            if not title_cells:
                current["metadataRows"].append(_row_value(row))

        title_indexes = {line_index for line_index, _ in title_cells}
        for line_index, line in row:
            if line_index in title_indexes:
                continue
            matches = dates_by_line[line_index]
            if matches:
                residual = line.text
                for match in reversed(matches):
                    residual = residual[:match.start()] + residual[match.end():]
                if not residual.strip(" ()[]{}|,;:-–—"):
                    continue
            if current is None:
                current = _grouped_section_entry()
                entries.append(current)
            line_urls = url_entities_by_line.get(line_index, [])
            rounded_bbox = rounded(line.bbox)
            bullet = BULLET_RE.match(line.text)
            if bullet:
                value: dict[str, Any] = {
                    "text": line.text[bullet.end():].strip(),
                    "page": line.page,
                    "bbox": rounded_bbox,
                    "detectionMethod": "bullet_marker",
                }
                if line_urls:
                    value["entities"] = line_urls
                current["bullets"].append(value)
                current["urls"].extend(line_urls)
                current["_lastType"] = "bullet"
                current["_lastLineBbox"] = rounded_bbox
                continue
            if current["_lastType"] == "bullet" and current["_lastLineBbox"] is not None:
                if continues_block(
                    current["_lastLineBbox"],
                    line.bbox,
                    same_page=True,
                    require_horizontal_overlap=False,
                    statistics=statistics,
                    previous_text=last_block_text(current),
                ):
                    extend_block(
                        current["bullets"][-1],
                        text=line.text,
                        box=line.bbox,
                        entities=line_urls,
                    )
                    current["urls"].extend(line_urls)
                    current["_lastLineBbox"] = rounded_bbox
                    continue
            # A heading that has opened a bracket and not closed it has not
            # finished naming what it names. The line under it completes that
            # phrase rather than starting a paragraph of its own -- the
            # author's punctuation is better evidence than the typography,
            # which only sees a line that is not set like a heading.
            #
            # The open bracket is not always on the most recent heading: in a
            # multi-column table the cells of one row all arrive first, so the
            # heading left hanging can be several back. Take the last one that
            # is still open, which is the phrase this line can be completing.
            unfinished = next(
                (
                    value
                    for value in reversed(current["subheadingLines"])
                    if not is_parenthetically_complete(value["text"])
                ),
                None,
            )
            if unfinished is not None and continues_block(
                unfinished["bbox"],
                line.bbox,
                same_page=unfinished["page"] == line.page,
                require_horizontal_overlap=False,
                statistics=statistics,
                previous_text=unfinished["text"],
            ):
                extend_block(
                    unfinished,
                    text=line.text,
                    box=line.bbox,
                    entities=line_urls,
                )
                current["urls"].extend(line_urls)
                current["_lastLineBbox"] = rounded_bbox
                continue

            append_paragraph(
                current,
                text=line.text,
                page=line.page,
                bbox=rounded_bbox,
                entities=line_urls,
                statistics=statistics,
            )
            current["urls"].extend(line_urls)

    for entry in entries:
        entry.pop("_lastType", None)
        entry.pop("_lastLineBbox", None)
    next_section = routed[position + 1] if position + 1 < len(routed) else None
    return {
        "sectionType": section_type,
        "heading": {
            "text": heading_line.text,
            "page": heading_line.page,
            "bbox": rounded(heading_line.bbox),
            "similarity": round(heading.similarity, 4),
            "detectionMethod": (
                "geometry_unknown_boundary"
                if heading.line_index not in {item.line_index for item in headings}
                else "geometry_semantic"
            ),
        },
        "rows": [_row_value(row) for row in visible_rows],
        "entries": entries,
        "stoppedAtSection": (
            {
                "sectionType": next_section.section_type,
                "text": lines[next_section.heading.line_index].text,
            }
            if next_section else None
        ),
    }

def build_projects_debug(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
    url_entities_by_line: dict[int, list[dict[str, Any]]],
    statistics: DocumentStatistics,
) -> dict[str, Any] | None:
    return _build_grouped_section_debug(
        document,
        lines,
        headings,
        url_entities_by_line,
        "projects",
        statistics,
    )

def build_supplementary_sections_debug(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
    url_entities_by_line: dict[int, list[dict[str, Any]]],
    semantic_model: EmbeddingModel,
    statistics: DocumentStatistics,
) -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    for section_type in _SUPPLEMENTARY_SECTION_TYPES:
        if section_type == "others":
            other_sections: list[dict[str, Any]] = []
            occurrence = 0
            while True:
                section = _build_grouped_section_debug(
                    document,
                    lines,
                    headings,
                    url_entities_by_line,
                    section_type,
                    statistics,
                    occurrence,
                    semantic_model,
                )
                if section is None:
                    break
                other_sections.append(section)
                occurrence += 1
            if other_sections:
                sections[section_type] = {
                    "sectionType": "others",
                    "sections": other_sections,
                }
            continue
        section = _build_grouped_section_debug(
            document,
            lines,
            headings,
            url_entities_by_line,
            section_type,
            statistics,
        )
        if section is not None:
            sections[section_type] = section
    return sections
