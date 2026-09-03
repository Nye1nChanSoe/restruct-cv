"""Physical sections: a confirmed heading and the blocks beneath it."""
from __future__ import annotations

from typing import Any


from restruct.document.types import DetectedHeading, ExtractedLine
from restruct.geometry import rounded
from restruct.layout.blocks import continues_block, extend_block
from restruct.patterns.bullets import BULLET_RE
from restruct.structure.headings import (
    _looks_like_subheading,
    _routed_section_headings,
    _section_body_style,
    first_header_boundary,
)


def _content_blocks(
    lines: list[ExtractedLine],
    line_indexes: list[int],
    url_entities_by_line: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Route lines into subheadings, paragraphs, or reconstructed bullets."""
    content_lines = [lines[index] for index in line_indexes]
    body_size, body_bold = _section_body_style(content_lines)
    blocks: list[dict[str, Any]] = []

    for line_index in line_indexes:
        line = lines[line_index]
        line_url_entities = url_entities_by_line.get(line_index, [])
        bullet_match = BULLET_RE.match(line.text)
        if bullet_match is not None:
            bullet_text = line.text[bullet_match.end():].strip()
            if not bullet_text:
                continue
            block = {
                "type": "bullet",
                "text": bullet_text,
                "page": line.page,
                "bbox": rounded(line.bbox),
                "detectionMethod": "bullet_marker",
                "_lastLineBbox": rounded(line.bbox),
            }
            if line_url_entities:
                block["entities"] = line_url_entities
            blocks.append(block)
            continue

        role = (
            "subheading"
            if _looks_like_subheading(
                line,
                body_size=body_size,
                body_bold=body_bold,
            )
            else "paragraph"
        )
        rounded_bbox = rounded(line.bbox)
        if role == "paragraph" and blocks and blocks[-1]["type"] == "bullet":
            previous = blocks[-1]
            if continues_block(
                previous["_lastLineBbox"],
                line.bbox,
                same_page=previous["page"] == line.page,
                require_horizontal_overlap=True,
            ):
                extend_block(
                    previous,
                    text=line.text,
                    box=line.bbox,
                    entities=line_url_entities,
                )
                previous["_lastLineBbox"] = rounded_bbox
                continue

        if role == "paragraph" and blocks and blocks[-1]["type"] == "paragraph":
            previous = blocks[-1]
            if continues_block(
                previous["_lastLineBbox"],
                line.bbox,
                same_page=previous["page"] == line.page,
                require_horizontal_overlap=True,
            ):
                extend_block(
                    previous,
                    text=line.text,
                    box=line.bbox,
                    entities=line_url_entities,
                )
                previous["_lastLineBbox"] = rounded_bbox
                continue

        block: dict[str, Any] = {
            "type": role,
            "text": line.text,
            "page": line.page,
            "bbox": rounded_bbox,
            "detectionMethod": (
                "geometry_typography"
                if role == "subheading"
                else "geometry_default"
            ),
            "_lastLineBbox": rounded_bbox,
        }
        if line_url_entities:
            block["entities"] = line_url_entities
        blocks.append(block)

    for block in blocks:
        block.pop("_lastLineBbox", None)
    return blocks

def build_sections(
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
    url_entities_by_line: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Use MiniLM-confirmed headings as boundaries and geometry within sections."""
    first_boundary = first_header_boundary(lines, headings)
    if first_boundary is None:
        return []
    routed_headings = _routed_section_headings(
        lines,
        headings,
        minimum_line_index=first_boundary.line_index,
    )
    semantic_heading_indexes = {heading.line_index for heading in headings}

    sections: list[dict[str, Any]] = []
    for position, heading in enumerate(routed_headings):
        next_heading_index = (
            routed_headings[position + 1].line_index
            if position + 1 < len(routed_headings)
            else len(lines)
        )
        heading_line = lines[heading.line_index]
        content_indexes = list(range(heading.line_index + 1, next_heading_index))
        heading_value: dict[str, Any] = {
            "text": heading_line.text,
            "page": heading_line.page,
            "bbox": rounded(heading_line.bbox),
            "similarity": round(heading.similarity, 4),
            "detectionMethod": (
                "geometry_semantic"
                if heading.line_index in semantic_heading_indexes
                else "geometry_unknown_boundary"
            ),
        }
        heading_url_entities = url_entities_by_line.get(heading.line_index, [])
        if heading_url_entities:
            heading_value["entities"] = heading_url_entities
        sections.append(
            {
                "sectionType": heading.section_type,
                "heading": heading_value,
                "content": _content_blocks(
                    lines,
                    content_indexes,
                    url_entities_by_line,
                ),
            }
        )
    return sections

def summary_debug_value(
    sections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not sections or sections[0]["sectionType"] != "summary":
        return None
    summary = dict(sections[0])
    summary["stoppedAtSection"] = (
        {
            "sectionType": sections[1]["sectionType"],
            "heading": sections[1]["heading"],
        }
        if len(sections) > 1
        else None
    )
    return summary
