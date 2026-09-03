"""Skill groups: a label and the skills it introduces, however delimited."""
from __future__ import annotations

from typing import Any

import pymupdf

from restruct.document.stats import DocumentStatistics
from restruct.document.types import DetectedHeading, ExtractedLine
from restruct.geometry import resolve_span_box, rounded
from restruct.layout.blocks import append_paragraph, continues_block, extend_block
from restruct.layout.rows import _row_value, _visual_rows
from restruct.patterns.bullets import BULLET_RE
from restruct.patterns.layout import PAGE_FOOTER_RE
from restruct.structure.headings import (
    _looks_like_subheading,
    _routed_section_headings,
    _section_body_style,
)
from restruct.structure.keyvalue import _skill_inline_parts, _skill_subheading_value


def _new_skill_group(
    groups: list[dict[str, Any]],
    subheading: dict[str, Any] | None,
) -> dict[str, Any]:
    group: dict[str, Any] = {
        "subheading": subheading,
        "paragraphs": [],
        "bullets": [],
        "urls": [],
        "_lastType": None,
        "_lastLineBbox": None,
    }
    groups.append(group)
    return group

def build_skills_debug(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
    url_entities_by_line: dict[int, list[dict[str, Any]]],
    statistics: DocumentStatistics,
) -> dict[str, Any] | None:
    """Preserve skill prose and attach vertical bullets to geometric groups."""
    routed = _routed_section_headings(lines, headings)
    position = next((i for i, item in enumerate(routed) if item.section_type == "skills"), None)
    if position is None:
        return None
    heading = routed[position]
    end = routed[position + 1].line_index if position + 1 < len(routed) else len(lines)
    heading_line = lines[heading.line_index]
    line_range = range(heading.line_index + 1, end)
    rows = _visual_rows(lines, line_range, statistics)
    body_size, body_bold = _section_body_style([lines[index] for index in line_range])
    groups: list[dict[str, Any]] = []
    routed_rows: list[list[tuple[int, ExtractedLine]]] = []
    current: dict[str, Any] | None = None

    for row in rows:
        if all(
            PAGE_FOOTER_RE.search(line.text) is not None
            and line.bbox[1] >= document[line.page - 1].rect.height * 0.88
            for _, line in row
        ):
            continue
        routed_rows.append(row)

        # A short left cell paired with right-side content is a table-like group row.
        if len(row) >= 2:
            left_index, left = row[0]
            right_cells = row[1:]
            left_is_label = (
                len(left.text.split()) <= 7
                and len(left.text) <= 50
                and (
                    left.bold
                    or _looks_like_subheading(left, body_size=body_size, body_bold=body_bold)
                    or pymupdf.Rect(right_cells[0][1].bbox).x0
                    > pymupdf.Rect(left.bbox).x1 + left.size
                )
            )
            if left_is_label:
                stripped = left.text.strip().rstrip(":-–—").strip()
                start = left.text.find(stripped)
                current = _new_skill_group(
                    groups,
                    _skill_subheading_value(
                        document,
                        left,
                        stripped,
                        start,
                        start + len(stripped),
                        "geometry_table_cell",
                    ),
                )
                current["urls"].extend(url_entities_by_line.get(left_index, []))
                for line_index, line in right_cells:
                    line_urls = url_entities_by_line.get(line_index, [])
                    append_paragraph(
                        current,
                        text=line.text,
                        page=line.page,
                        bbox=rounded(line.bbox),
                        entities=line_urls,
                        statistics=statistics,
                    )
                    current["urls"].extend(line_urls)
                continue

        handled_row = False
        for line_index, line in row:
            line_urls = url_entities_by_line.get(line_index, [])
            bullet_match = BULLET_RE.match(line.text)
            content_start = bullet_match.end() if bullet_match else 0
            content = line.text[content_start:].strip()
            leading_space = len(line.text[content_start:]) - len(line.text[content_start:].lstrip())
            content_start += leading_space
            inline = _skill_inline_parts(
                content,
                content_start,
                allow_single_dash_body=bullet_match is None,
            )
            if inline is not None:
                label, label_start, label_end, body, body_start, body_end, method = inline
                current = _new_skill_group(
                    groups,
                    _skill_subheading_value(
                        document,
                        line,
                        label,
                        label_start,
                        label_end,
                        method,
                    ),
                )
                body_value: dict[str, Any] = {
                    "text": body,
                    "page": line.page,
                    "bbox": resolve_span_box(document, line, body, body_start, body_end),
                    "detectionMethod": "bullet_marker" if bullet_match else "geometry_default",
                }
                if line_urls:
                    body_value["entities"] = line_urls
                target = "bullets" if bullet_match else "paragraphs"
                current[target].append(body_value)
                current["urls"].extend(line_urls)
                current["_lastType"] = "bullet" if bullet_match else "paragraph"
                current["_lastLineBbox"] = rounded(line.bbox)
                handled_row = True
                continue

            standalone_subheading = (
                not bullet_match
                and len(row) == 1
                and (
                    (
                        line.text.strip().endswith((":", "-", "–", "—"))
                        and len(line.text.split()) <= 7
                    )
                    or _looks_like_subheading(
                        line,
                        body_size=body_size,
                        body_bold=body_bold,
                    )
                )
            )
            if standalone_subheading:
                stripped = line.text.strip().rstrip(":-–—").strip()
                start = line.text.find(stripped)
                current = _new_skill_group(
                    groups,
                    _skill_subheading_value(
                        document,
                        line,
                        stripped,
                        start,
                        start + len(stripped),
                        "geometry_typography",
                    ),
                )
                current["urls"].extend(line_urls)
                handled_row = True
                continue

            current = current or _new_skill_group(groups, None)
            rounded_bbox = rounded(line.bbox)
            if bullet_match:
                value: dict[str, Any] = {
                    "text": content,
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
            elif current["_lastType"] == "bullet" and current["_lastLineBbox"] is not None:
                if continues_block(
                    current["_lastLineBbox"],
                    line.bbox,
                    same_page=True,
                    require_horizontal_overlap=False,
                    statistics=statistics,
                ):
                    extend_block(
                        current["bullets"][-1],
                        text=line.text,
                        box=line.bbox,
                        entities=line_urls,
                    )
                    current["urls"].extend(line_urls)
                    current["_lastLineBbox"] = rounded_bbox
                else:
                    append_paragraph(
                        current,
                        text=line.text,
                        page=line.page,
                        bbox=rounded_bbox,
                        entities=line_urls,
                        statistics=statistics,
                    )
                    current["urls"].extend(line_urls)
            else:
                append_paragraph(
                    current,
                    text=line.text,
                    page=line.page,
                    bbox=rounded_bbox,
                    entities=line_urls,
                    statistics=statistics,
                )
                current["urls"].extend(line_urls)
            handled_row = True
        if not handled_row:
            continue

    for group in groups:
        group.pop("_lastType", None)
        group.pop("_lastLineBbox", None)
    next_heading = routed[position + 1] if position + 1 < len(routed) else None
    return {
        "sectionType": "skills",
        "heading": {
            "text": heading_line.text,
            "page": heading_line.page,
            "bbox": rounded(heading_line.bbox),
            "similarity": round(heading.similarity, 4),
            "detectionMethod": "geometry_semantic",
        },
        "rows": [_row_value(row) for row in routed_rows],
        "groups": groups,
        "stoppedAtSection": (
            {"sectionType": next_heading.section_type, "text": lines[next_heading.line_index].text}
            if next_heading else None
        ),
    }
