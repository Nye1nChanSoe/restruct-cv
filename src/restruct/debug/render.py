"""Pillow debug rendering.

Every overlay is drawn through one renderer so a box means the same thing in
every section, and so model-backed detections stay visually distinct from
deterministic reconstruction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf

from restruct.configs import SETTINGS
from restruct.debug.colors import (
    COMBINED_OUTLINE,
    EDUCATION_STYLES,
    EXPERIENCE_STYLES,
    GROUPED_STYLES,
    ItemStyle,
    PROJECTS_STYLES,
    SKILLS_STYLES,
    SUMMARY_STYLES,
    colors_of,
    labels_of,
    profile_attribute_color,
)
from restruct.geometry import pixel_box
from restruct.debug import canvas
from restruct.debug.canvas import stroke_width
from restruct.structure.resolver import is_model_backed


def _render_entry_debug_images(
    document: pymupdf.Document,
    items: list[dict[str, Any]],
    output_directory: Path,
    colors: dict[str, str],
    labels: dict[str, str],
    metadata_types: set[str],
) -> None:
    items_by_page: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        items_by_page.setdefault(int(item["page"]), []).append(item)
    for page_number, page_items in sorted(items_by_page.items()):
        image, draw = canvas.page_canvas(document, page_number)
        page_item_boxes = [pixel_box(item["bbox"]) for item in page_items]
        counts: dict[str, int] = {}
        styles: dict[str, ItemStyle] = {}
        placed_label_boxes: list[tuple[int, int, int, int]] = []
        for item_index, item in enumerate(page_items):
            item_type = item["type"]
            box = pixel_box(item["bbox"])
            color = str(item.get("_debugColor") or colors[item_type])
            draw.rectangle(
                box,
                outline=color,
                width=stroke_width(
                    item_type,
                    str(item.get("detectionMethod", "")),
                ),
            )
            label = str(item.get("_debugLabel") or labels[item_type])
            counts[label] = counts.get(label, 0) + 1
            styles[label] = ItemStyle(color, label)
            measured_label_box = draw.textbbox((0, 0), label)
            label_width = measured_label_box[2] - measured_label_box[0]
            label_height = measured_label_box[3] - measured_label_box[1]
            if item_type in metadata_types:
                label_position = (
                    min(
                        image.width - label_width,
                        box[2] + SETTINGS.debug.label_x_padding,
                    ),
                    max(0, (box[1] + box[3] - label_height) // 2),
                )
            elif item_type in {"date", "location"}:
                label_position = (
                    max(0, min(image.width - label_width, box[2] - label_width)),
                    min(image.height - label_height, box[3] + 2),
                )
            else:
                label_position = (
                    box[0] + SETTINGS.debug.label_x_padding,
                    max(0, box[1] - SETTINGS.debug.label_y_offset),
                )
            placed_label_boxes.append(
                canvas.place_label(
                    draw,
                    position=label_position,
                    text=label,
                    color=color,
                    avoid=[
                        other_box
                        for other_index, other_box in enumerate(page_item_boxes)
                        if other_index != item_index
                    ]
                    + placed_label_boxes,
                    fallback=(
                        max(0, box[0] - label_width - SETTINGS.debug.label_x_padding),
                        max(
                            0,
                            min(
                                image.height - label_height,
                                (box[1] + box[3] - label_height) // 2,
                            ),
                        ),
                    ),
                )
            )
        canvas.legend(
            draw,
            [(styles[name], count) for name, count in counts.items()],
            title=f"pass 5 - {output_directory.name}  |  page {page_number}",
        )
        canvas.save(image, output_directory, page_number)


def render_combined_debug_images(
    document: pymupdf.Document,
    header_profile: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    experience: dict[str, Any] | None,
    education: dict[str, Any] | None,
    skills: dict[str, Any] | None,
    projects: dict[str, Any] | None,
    supplementary_sections: dict[str, dict[str, Any]],
    output_directory: Path,
) -> None:
    """Render all implemented section overlays together on each source page."""
    items: list[dict[str, Any]] = []

    def add(
        item_type: str,
        value: dict[str, Any],
        color: str,
        label: str,
    ) -> None:
        items.append(
            {
                **value,
                "type": item_type,
                "_debugColor": color,
                "_debugLabel": label,
            }
        )

    if header_profile is not None:
        add(
            "header_profile",
            {
                "page": header_profile["page"],
                "bbox": header_profile["bbox"],
                "detectionMethod": "geometry_header_region",
            },
            SETTINGS.debug.header_region_color,
            "header_profile",
        )
        header_colors = dict(SETTINGS.debug.header_entity_colors)
        for entity in header_profile["entities"]:
            add(
                str(entity["type"]),
                entity,
                header_colors[str(entity["type"])],
                str(entity["type"]),
            )

    if summary is not None:
        add(
            "section_heading",
            summary["heading"],
            SETTINGS.section_colors["summary"],
            "section_heading",
        )
        for value in summary["content"]:
            item_type = str(value["type"])
            style = SUMMARY_STYLES[item_type]
            add(item_type, value, style.color, style.label)

    if experience is not None:
        add(
            "section_heading",
            experience["heading"],
            SETTINGS.section_colors["experience"],
            "section_heading",
        )
        experience_specs = EXPERIENCE_STYLES
        for entry in experience["entries"]:
            for value in entry["subheadingLines"]:
                style = experience_specs["experience_subheading"]
                add("experience_subheading", value, style.color, style.label)
            for key, item_type in (
                ("jobTitles", "job_title"),
                ("companies", "company"),
                ("dates", "date"),
                ("locations", "location"),
                ("urls", "url"),
                ("paragraphs", "paragraph"),
                ("bullets", "bullet"),
            ):
                style = experience_specs[item_type]
                for value in entry[key]:
                    # A company grounded by its link annotation rather than by
                    # NER is labelled for what actually found it.
                    item_label = (
                        "annotation: company"
                        if (
                            item_type == "company"
                            and value.get("detectionMethod") == "url_company_reconciled"
                        )
                        else style.label
                    )
                    add(item_type, value, style.color, item_label)

    if education is not None:
        add(
            "section_heading",
            education["heading"],
            SETTINGS.section_colors["education"],
            "section_heading",
        )
        education_specs = EDUCATION_STYLES
        for entry in education["entries"]:
            for value in entry["metadataRows"]:
                style = education_specs["education_metadata"]
                add("education_metadata", value, style.color, style.label)
            for key, item_type in (
                ("titles", "education_title"),
                ("institutions", "institution"),
                ("dates", "date"),
                ("locations", "location"),
                ("gpa", "gpa"),
                ("skills", "skill"),
                ("urls", "url"),
                ("paragraphs", "paragraph"),
                ("bullets", "bullet"),
            ):
                style = education_specs[item_type]
                for value in entry[key]:
                    add(item_type, value, style.color, style.label)

    if skills is not None:
        add(
            "section_heading",
            skills["heading"],
            SETTINGS.section_colors["skills"],
            "section_heading",
        )
        for value in skills["rows"]:
            style = SKILLS_STYLES["skill_row"]
            add("skill_row", value, style.color, style.label)
        for group in skills["groups"]:
            if group["subheading"] is not None:
                style = SKILLS_STYLES["skill_subheading"]
                add("skill_subheading", group["subheading"], style.color, style.label)
            for key, item_type in (
                ("paragraphs", "paragraph"),
                ("bullets", "bullet"),
                ("urls", "url"),
            ):
                style = SKILLS_STYLES[item_type]
                for value in group[key]:
                    add(item_type, value, style.color, style.label)

    grouped_sections: list[tuple[str, dict[str, Any]]] = []
    if projects is not None:
        grouped_sections.append(("projects", projects))
    for section_type, section in supplementary_sections.items():
        for section_part in (
            section["sections"] if section_type == "others" else [section]
        ):
            grouped_sections.append((section_type, section_part))

    for section_type, section in grouped_sections:
        section_color = SETTINGS.section_colors[section_type]
        add(
            "section_heading",
            section["heading"],
            section_color,
            "section_heading",
        )
        for value in section["rows"]:
            style = GROUPED_STYLES["grouped_row"]
            add("grouped_row", value, style.color, style.label)
        for entry in section["entries"]:
            for value in entry["subheadingLines"]:
                add("grouped_subheading", value, section_color, "subheading")
            for value in entry.get("attributes", []):
                attribute_type = str(value["type"])
                add(
                    "profile_attribute",
                    value,
                    profile_attribute_color(attribute_type, section_color),
                    f"attribute: {attribute_type}",
                )
            for key, item_type in (
                ("dates", "date"),
                ("urls", "url"),
                ("paragraphs", "paragraph"),
                ("bullets", "bullet"),
            ):
                style = GROUPED_STYLES[item_type]
                for value in entry[key]:
                    add(item_type, value, style.color, style.label)

    if not items:
        return
    item_types = {str(item["type"]) for item in items}
    _render_entry_debug_images(
        document,
        items,
        output_directory,
        {item_type: COMBINED_OUTLINE for item_type in item_types},
        {item_type: item_type for item_type in item_types},
        {
            "name",
            "experience_subheading",
            "education_metadata",
            "skill_row",
            "grouped_row",
        },
    )
