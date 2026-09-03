"""Pillow debug rendering.

Every overlay is drawn through one renderer so a box means the same thing in
every section, and so model-backed detections stay visually distinct from
deterministic reconstruction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageDraw

from restruct.configs import SETTINGS
from restruct.debug.colors import (
    COMBINED_OUTLINE,
    EDUCATION_STYLES,
    EXPERIENCE_STYLES,
    GROUPED_STYLES,
    LABEL_BACKGROUND,
    PROJECTS_STYLES,
    SKILLS_STYLES,
    SUMMARY_STYLES,
    colors_of,
    labels_of,
    profile_attribute_color,
)
from restruct.geometry import pixel_box


def render_debug_images(
    document: pymupdf.Document,
    header_profile: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    """Draw only the top profile region and its detected entities."""
    if header_profile is None:
        return
    output_directory.mkdir(parents=True, exist_ok=True)
    matrix = pymupdf.Matrix(SETTINGS.debug.scale, SETTINGS.debug.scale)
    page_number = header_profile["page"]
    page = document[page_number - 1]
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    draw = ImageDraw.Draw(image)

    profile_box = pixel_box(header_profile["bbox"])
    draw.rectangle(
        profile_box,
        outline=SETTINGS.debug.header_region_color,
        width=SETTINGS.debug.header_region_stroke_width,
    )
    draw.text(
        (
            profile_box[0] + SETTINGS.debug.label_x_padding,
            max(0, profile_box[1] - SETTINGS.debug.label_y_offset),
        ),
        "header_profile",
        fill=SETTINGS.debug.header_region_color,
    )

    entity_colors = dict(SETTINGS.debug.header_entity_colors)
    for entity in header_profile["entities"]:
        color = entity_colors[entity["type"]]
        entity_box = pixel_box(entity["bbox"])
        draw.rectangle(
            entity_box,
            outline=color,
            width=SETTINGS.debug.header_entity_stroke_width,
        )
        if entity["type"] == "name":
            label_position = (
                entity_box[2] + SETTINGS.debug.label_x_padding,
                entity_box[1],
            )
            label = "name"
        else:
            label_position = (
                entity_box[0] + SETTINGS.debug.label_x_padding,
                max(0, entity_box[1] - SETTINGS.debug.label_y_offset),
            )
            label = entity["type"]
        draw.text(
            label_position,
            label,
            fill=color,
        )

    image.save(output_directory / f"page-{page_number}.png")


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

    colors = colors_of(SUMMARY_STYLES, "summary")
    matrix = pymupdf.Matrix(SETTINGS.debug.scale, SETTINGS.debug.scale)
    for page_number, page_items in sorted(items_by_page.items()):
        page = document[page_number - 1]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        draw = ImageDraw.Draw(image)
        for item in page_items:
            item_type = str(item["type"])
            color = colors[item_type]
            item_box = pixel_box(item["bbox"])
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
    output_directory.mkdir(parents=True, exist_ok=True)
    matrix = pymupdf.Matrix(SETTINGS.debug.scale, SETTINGS.debug.scale)
    for page_number, page_items in sorted(items_by_page.items()):
        page = document[page_number - 1]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        draw = ImageDraw.Draw(image)
        page_item_boxes = [pixel_box(item["bbox"]) for item in page_items]
        placed_label_boxes: list[tuple[int, int, int, int]] = []
        for item_index, item in enumerate(page_items):
            item_type = item["type"]
            box = pixel_box(item["bbox"])
            color = str(item.get("_debugColor") or colors[item_type])
            detection_method = str(item.get("detectionMethod", ""))
            model_entity = detection_method.startswith(("distilbert", "minilm"))
            draw.rectangle(
                box,
                outline=color,
                width=(
                    SETTINGS.debug.header_entity_stroke_width + 2
                    if model_entity
                    else (
                        SETTINGS.debug.heading_stroke_width
                        if item_type == "section_heading"
                        else 2
                    )
                ),
            )
            label = str(item.get("_debugLabel") or labels[item_type])
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
            label_box = draw.textbbox(label_position, label)
            collides = any(
                label_box[0] < other_box[2]
                and other_box[0] < label_box[2]
                and label_box[1] < other_box[3]
                and other_box[1] < label_box[3]
                for other_index, other_box in enumerate(page_item_boxes)
                if other_index != item_index
            ) or any(
                label_box[0] < other_box[2]
                and other_box[0] < label_box[2]
                and label_box[1] < other_box[3]
                and other_box[1] < label_box[3]
                for other_box in placed_label_boxes
            )
            if collides:
                label_position = (
                    max(
                        0,
                        box[0] - label_width - SETTINGS.debug.label_x_padding,
                    ),
                    max(
                        0,
                        min(
                            image.height - label_height,
                            (box[1] + box[3] - label_height) // 2,
                        ),
                    ),
                )
                label_box = draw.textbbox(label_position, label)
            draw.rectangle(label_box, fill=LABEL_BACKGROUND)
            draw.text(label_position, label, fill=color)
            placed_label_boxes.append(label_box)
        image.save(output_directory / f"page-{page_number}.png")


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


def render_experience_debug_images(
    document: pymupdf.Document,
    experience: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if experience is None:
        return
    items: list[dict[str, Any]] = [{"type": "section_heading", **experience["heading"]}]
    for entry in experience["entries"]:
        items.extend({"type": "experience_subheading", **item} for item in entry["subheadingLines"])
        for key, item_type in (
            ("jobTitles", "job_title"),
            ("companies", "company"),
            ("dates", "date"),
            ("locations", "location"),
            ("urls", "url"),
        ):
            items.extend(
                {
                    "type": item_type,
                    **item,
                    **(
                        {"_debugLabel": "annotation: company"}
                        if item_type == "company"
                        and item.get("detectionMethod") == "url_company_reconciled"
                        else {}
                    ),
                }
                for item in entry[key]
            )
        items.extend({"type": "paragraph", **item} for item in entry["paragraphs"])
        items.extend({"type": "bullet", **item} for item in entry["bullets"])
    _render_entry_debug_images(
        document,
        items,
        output_directory,
        colors_of(EXPERIENCE_STYLES, "experience"),
        labels_of(EXPERIENCE_STYLES),
        {"experience_subheading"},
    )


def render_education_debug_images(
    document: pymupdf.Document,
    education: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if education is None:
        return
    items: list[dict[str, Any]] = [{"type": "section_heading", **education["heading"]}]
    for entry in education["entries"]:
        items.extend({"type": "education_metadata", **item} for item in entry["metadataRows"])
        for key, item_type in (
            ("titles", "education_title"),
            ("institutions", "institution"),
            ("dates", "date"),
            ("locations", "location"),
            ("gpa", "gpa"),
            ("skills", "skill"),
            ("urls", "url"),
        ):
            items.extend({"type": item_type, **item} for item in entry[key])
        items.extend({"type": "paragraph", **item} for item in entry["paragraphs"])
        items.extend({"type": "bullet", **item} for item in entry["bullets"])
    _render_entry_debug_images(
        document,
        items,
        output_directory,
        colors_of(EDUCATION_STYLES, "education"),
        labels_of(EDUCATION_STYLES),
        {"education_metadata"},
    )


def render_skills_debug_images(
    document: pymupdf.Document,
    skills: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if skills is None:
        return
    items: list[dict[str, Any]] = [{"type": "section_heading", **skills["heading"]}]
    items.extend({"type": "skill_row", **item} for item in skills["rows"])
    for group in skills["groups"]:
        if group["subheading"] is not None:
            items.append({"type": "skill_subheading", **group["subheading"]})
        items.extend({"type": "paragraph", **item} for item in group["paragraphs"])
        items.extend({"type": "bullet", **item} for item in group["bullets"])
        items.extend({"type": "url", **item} for item in group["urls"])
    _render_entry_debug_images(
        document,
        items,
        output_directory,
        colors_of(SKILLS_STYLES, "skills"),
        labels_of(SKILLS_STYLES),
        {"skill_row"},
    )


def render_projects_debug_images(
    document: pymupdf.Document,
    projects: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if projects is None:
        return
    items: list[dict[str, Any]] = [{"type": "section_heading", **projects["heading"]}]
    items.extend({"type": "project_row", **item} for item in projects["rows"])
    for entry in projects["entries"]:
        items.extend({"type": "project_subheading", **item} for item in entry["subheadingLines"])
        items.extend({"type": "date", **item} for item in entry["dates"])
        items.extend({"type": "url", **item} for item in entry["urls"])
        items.extend({"type": "paragraph", **item} for item in entry["paragraphs"])
        items.extend({"type": "bullet", **item} for item in entry["bullets"])
    _render_entry_debug_images(
        document,
        items,
        output_directory,
        colors_of(PROJECTS_STYLES, "projects"),
        labels_of(PROJECTS_STYLES),
        {"project_row"},
    )


def render_supplementary_sections_debug_images(
    document: pymupdf.Document,
    sections: dict[str, dict[str, Any]],
    debug_directory: Path,
) -> None:
    for section_type, section in sections.items():
        section_color = SETTINGS.section_colors[section_type]
        section_parts = (
            section["sections"]
            if section_type == "others"
            else [section]
        )
        items: list[dict[str, Any]] = []
        for section_part in section_parts:
            items.append({"type": "section_heading", **section_part["heading"]})
            items.extend(
                {"type": "grouped_row", **item}
                for item in section_part["rows"]
            )
            for entry in section_part["entries"]:
                items.extend(
                    {"type": "grouped_subheading", **item}
                    for item in entry["subheadingLines"]
                )
                items.extend({"type": "date", **item} for item in entry["dates"])
                items.extend(
                    {
                        **item,
                        "type": "profile_attribute",
                        "_debugColor": dict(
                            SETTINGS.debug.header_entity_colors
                        ).get(str(item["type"]), section_color),
                        "_debugLabel": f"attribute: {item['type']}",
                    }
                    for item in entry.get("attributes", [])
                )
                items.extend({"type": "url", **item} for item in entry["urls"])
                items.extend(
                    {"type": "paragraph", **item}
                    for item in entry["paragraphs"]
                )
                items.extend({"type": "bullet", **item} for item in entry["bullets"])
        _render_entry_debug_images(
            document,
            items,
            debug_directory / section_type,
            {
                **colors_of(GROUPED_STYLES, section_type),
                "grouped_subheading": section_color,
                "profile_attribute": section_color,
            },
            {
                **labels_of(GROUPED_STYLES),
                "grouped_subheading": "subheading",
                "profile_attribute": "attribute",
            },
            {"grouped_row"},
        )
