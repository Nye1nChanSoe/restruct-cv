"""The single registry of debug overlay colors and labels.

Before this module the same item styles were written out twice -- once in each
per-section renderer and again in the combined overlay -- so a color could be
changed in one view and not the other. Each style is now defined once.

Labels name the *evidence*, not just the field, so an overlay says whether a box
came from a regex, from NER, from MiniLM, or from geometry alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from restruct.configs import SETTINGS


@dataclass(frozen=True)
class ItemStyle:
    """How one kind of detection is drawn and named."""

    color: str
    label: str


# Detections that mean the same thing wherever they appear.
DATE = ItemStyle("#F9A825", "regex: date")
URL = ItemStyle("#6D4C41", "annotation: url")
LOCATION = ItemStyle("#1E88E5", "NER: location")

# Prose weights differ by section on purpose: the record-shaped sections use a
# lighter fill so their metadata boxes stay readable on top.
RECORD_PARAGRAPH = ItemStyle("#CFD8DC", "paragraph")
RECORD_BULLET = ItemStyle("#B0BEC5", "bullet")
LIST_PARAGRAPH = ItemStyle("#90A4AE", "paragraph")
LIST_BULLET = ItemStyle("#5C6BC0", "bullet")

SUMMARY_STYLES: dict[str, ItemStyle] = {
    "subheading": ItemStyle("#EF6C00", "subheading"),
    "paragraph": ItemStyle("#546E7A", "paragraph"),
    "bullet": ItemStyle("#5C6BC0", "bullet"),
}

EXPERIENCE_STYLES: dict[str, ItemStyle] = {
    "experience_subheading": ItemStyle("#B0BEC5", "metadata_line"),
    "job_title": ItemStyle("#00897B", "MiniLM: job_title"),
    "company": ItemStyle("#D81B60", "NER: company"),
    "date": DATE,
    "location": LOCATION,
    "url": URL,
    "paragraph": RECORD_PARAGRAPH,
    "bullet": RECORD_BULLET,
}

EDUCATION_STYLES: dict[str, ItemStyle] = {
    "education_metadata": ItemStyle("#D1C4E9", "geometry_row"),
    "education_title": ItemStyle("#00897B", "pattern: education_title"),
    "institution": ItemStyle("#D81B60", "entity: institution"),
    "date": DATE,
    "location": LOCATION,
    "gpa": ItemStyle("#8E24AA", "regex: GPA"),
    "skill": ItemStyle("#43A047", "skill"),
    "url": URL,
    "paragraph": RECORD_PARAGRAPH,
    "bullet": RECORD_BULLET,
}

SKILLS_STYLES: dict[str, ItemStyle] = {
    "skill_row": ItemStyle("#CFD8DC", "geometry_row"),
    "skill_subheading": ItemStyle("#00897B", "skill_group"),
    "paragraph": LIST_PARAGRAPH,
    "bullet": LIST_BULLET,
    "url": URL,
}

# The standalone projects view tints rows and subheadings so records stand out
# on their own page; inside the combined overlay they take the neutral grouped
# styling instead, so the surrounding sections stay legible.
PROJECTS_STYLES: dict[str, ItemStyle] = {
    "project_row": ItemStyle("#FFE0B2", "geometry_row"),
    "project_subheading": ItemStyle("#EF6C00", "project_subheading"),
    "date": DATE,
    "url": URL,
    "paragraph": LIST_PARAGRAPH,
    "bullet": LIST_BULLET,
}

GROUPED_STYLES: dict[str, ItemStyle] = {
    "grouped_row": ItemStyle("#ECEFF1", "geometry_row"),
    "date": DATE,
    "url": URL,
    "paragraph": LIST_PARAGRAPH,
    "bullet": LIST_BULLET,
}

# --- passes 1-3 -------------------------------------------------------------
# These render physical reconstruction rather than semantic decisions, so they
# stay deliberately cooler and thinner than the section overlays: a reader
# should be able to tell at a glance whether a box is something the document
# said, or something a model concluded.

PHYSICAL_STYLES: dict[str, ItemStyle] = {
    "span": ItemStyle("#90A4AE", "span"),
    "rule": ItemStyle("#D81B60", "drawn rule"),
    "image": ItemStyle("#8E24AA", "image region"),
    "link": ItemStyle("#6D4C41", "link annotation"),
    "furniture": ItemStyle("#E53935", "running header/footer"),
}

WORD_STYLES: dict[str, ItemStyle] = {
    "word": ItemStyle("#1E88E5", "word"),
    "ocr_word": ItemStyle("#00897B", "OCR word"),
    "linked_word": ItemStyle("#6D4C41", "word under a link"),
}

LINE_STYLES: dict[str, ItemStyle] = {
    "line": ItemStyle("#546E7A", "line"),
    "baseline": ItemStyle("#F9A825", "baseline"),
    "cell": ItemStyle("#D81B60", "cell"),
    "row": ItemStyle("#43A047", "row"),
}

# Unsupported layouts are drawn on the pass-1 overlay, because the geometry
# that produced the warning is what pass 1 shows. They share one hot colour on
# purpose: the point is that something is wrong, and the label says what.
UNSUPPORTED_STYLES: dict[str, ItemStyle] = {
    "multiple_columns": ItemStyle("#FF3D00", "unsupported: column gutter"),
    "vertical_text": ItemStyle("#FF3D00", "unsupported: vertical text"),
    "overlapping_text": ItemStyle("#FF3D00", "unsupported: overlapping text"),
    "text_in_graphics": ItemStyle("#FF3D00", "unsupported: text in a graphic"),
    "nested_table": ItemStyle("#FF3D00", "unsupported: nested table"),
}

# Drawn on top of every other overlay, so it needs no fill of its own.
LABEL_BACKGROUND = "#FFFFFF"
COMBINED_OUTLINE = "#000000"


def section_heading_style(section_type: str) -> ItemStyle:
    """Section headings take their section's own color."""
    return ItemStyle(SETTINGS.section_colors[section_type], "section_heading")


def profile_attribute_color(attribute_type: str, fallback: str) -> str:
    """Personal attributes keep the color they have in the header overlay."""
    return dict(SETTINGS.debug.header_entity_colors).get(attribute_type, fallback)


def colors_of(styles: dict[str, ItemStyle], section_type: str) -> dict[str, str]:
    """The color map ``_render_entry_debug_images`` expects."""
    return {
        "section_heading": section_heading_style(section_type).color,
        **{name: style.color for name, style in styles.items()},
    }


def labels_of(styles: dict[str, ItemStyle]) -> dict[str, str]:
    """The label map ``_render_entry_debug_images`` expects."""
    return {
        "section_heading": "section_heading",
        **{name: style.label for name, style in styles.items()},
    }
