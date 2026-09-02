"""Geometry-aware section and summary routing."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageDraw

from extractor_v1.configs import SETTINGS
from extractor_v1.model import DetectedHeading, ExtractedLine


def first_header_boundary(
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
) -> DetectedHeading | None:
    """Return the first high-confidence section heading on page one."""
    exact_references = {
        reference.casefold().strip()
        for references in SETTINGS.section_references.values()
        for reference in references
    }
    for heading in headings:
        line = lines[heading.line_index]
        if line.page != 1 or heading.line_index == 0:
            continue
        exact_match = line.text.casefold().strip() in exact_references
        strong_semantic_match = (
            heading.similarity
            >= SETTINGS.header_profile.boundary_similarity_threshold
            and heading.similarity - heading.runner_up_similarity
            >= SETTINGS.header_profile.boundary_winner_margin
        )
        if exact_match or strong_semantic_match:
            return heading
    return None


def _section_body_style(
    content_lines: list[ExtractedLine],
) -> tuple[float, bool]:
    prose_lines = [
        line
        for line in content_lines
        if len(line.text.split()) > SETTINGS.section_router.maximum_subheading_words
        or line.text.rstrip().endswith((".", ",", ";", ":"))
    ]
    reference_lines = prose_lines or content_lines
    sizes = [line.size for line in reference_lines if line.size > 0]
    body_size = statistics.median(sizes) if sizes else 0.0
    body_bold = bool(reference_lines) and sum(
        line.bold for line in reference_lines
    ) > len(reference_lines) / 2
    return body_size, body_bold


def _looks_like_subheading(
    line: ExtractedLine,
    *,
    body_size: float,
    body_bold: bool,
) -> bool:
    text = line.text.strip()
    if not text or text.endswith((".", ",", ";", ":")):
        return False
    if len(text) > SETTINGS.section_router.maximum_subheading_characters:
        return False
    if len(text.split()) > SETTINGS.section_router.maximum_subheading_words:
        return False
    size_contrast = (
        body_size > 0
        and line.size
        >= body_size * SETTINGS.section_router.subheading_font_size_multiplier
    )
    bold_contrast = line.bold and not body_bold
    return size_contrast or bold_contrast


def _content_blocks(
    lines: list[ExtractedLine],
    line_indexes: list[int],
    url_entities_by_line: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Route lines into typographic subheadings or combined paragraph blocks."""
    content_lines = [lines[index] for index in line_indexes]
    body_size, body_bold = _section_body_style(content_lines)
    blocks: list[dict[str, Any]] = []

    for line_index in line_indexes:
        line = lines[line_index]
        line_url_entities = url_entities_by_line.get(line_index, [])
        role = (
            "subheading"
            if _looks_like_subheading(
                line,
                body_size=body_size,
                body_bold=body_bold,
            )
            else "paragraph"
        )
        rounded_bbox = [round(value, 2) for value in line.bbox]
        if role == "paragraph" and blocks and blocks[-1]["type"] == "paragraph":
            previous = blocks[-1]
            previous_box = pymupdf.Rect(previous["_lastLineBbox"])
            current_box = pymupdf.Rect(line.bbox)
            vertical_gap = current_box.y0 - previous_box.y1
            horizontal_overlap = max(
                0.0,
                min(previous_box.x1, current_box.x1)
                - max(previous_box.x0, current_box.x0),
            )
            maximum_gap = max(
                previous_box.height,
                current_box.height,
            ) * SETTINGS.section_router.paragraph_gap_multiplier
            if (
                previous["page"] == line.page
                and -2.0 <= vertical_gap <= maximum_gap
                and horizontal_overlap > 0
            ):
                previous["text"] += "\n" + line.text
                previous["bbox"] = [
                    round(value, 2)
                    for value in (pymupdf.Rect(previous["bbox"]) | current_box)
                ]
                previous["_lastLineBbox"] = rounded_bbox
                if line_url_entities:
                    previous.setdefault("entities", []).extend(line_url_entities)
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
    routed_headings = sorted(
        (
            heading
            for heading in headings
            if heading.line_index >= first_boundary.line_index
        ),
        key=lambda heading: heading.line_index,
    )

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
            "bbox": [round(value, 2) for value in heading_line.bbox],
            "similarity": round(heading.similarity, 4),
            "detectionMethod": "geometry_semantic",
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


def write_summary_debug(
    pdf_path: Path,
    summary: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if summary is None:
        return
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(
        json.dumps(
            {
                "source": pdf_path.name,
                "model": SETTINGS.model.name,
                "modelRevision": SETTINGS.model.revision,
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _pixel_box(bbox: list[float] | tuple[float, ...]) -> tuple[int, int, int, int]:
    return tuple(
        round(value * SETTINGS.debug.scale) for value in bbox
    )  # type: ignore[return-value]


def render_summary_debug_images(
    document: pymupdf.Document,
    summary: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if summary is None:
        return
    output_directory.mkdir(parents=True, exist_ok=True)
    items = [
        {
            "type": "section_heading",
            "page": summary["heading"]["page"],
            "bbox": summary["heading"]["bbox"],
        },
        *summary["content"],
    ]
    items_by_page: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        items_by_page.setdefault(int(item["page"]), []).append(item)

    colors = {
        "section_heading": SETTINGS.section_colors["summary"],
        "subheading": "#EF6C00",
        "paragraph": "#546E7A",
    }
    matrix = pymupdf.Matrix(SETTINGS.debug.scale, SETTINGS.debug.scale)
    for page_number, page_items in sorted(items_by_page.items()):
        page = document[page_number - 1]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        draw = ImageDraw.Draw(image)
        for item in page_items:
            item_type = str(item["type"])
            color = colors[item_type]
            item_box = _pixel_box(item["bbox"])
            draw.rectangle(
                item_box,
                outline=color,
                width=(
                    SETTINGS.debug.heading_stroke_width
                    if item_type == "section_heading"
                    else SETTINGS.debug.content_stroke_width
                ),
            )
            draw.text(
                (
                    item_box[0] + SETTINGS.debug.label_x_padding,
                    max(0, item_box[1] - SETTINGS.debug.label_y_offset),
                ),
                item_type,
                fill=color,
            )
        image.save(output_directory / f"page-{page_number}.png")
