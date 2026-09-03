"""Guards on the debug colour registry.

Debug images are not covered by the golden snapshots, so a mistake in this
registry is invisible in normal test output. These checks pin the invariants
that make an overlay readable.
"""

from __future__ import annotations

import pytest

from restruct.configs import SETTINGS
from restruct.debug import colors
from restruct.schema import V1_SECTION_ORDER

ALL_STYLE_MAPS = {
    "summary": colors.SUMMARY_STYLES,
    "experience": colors.EXPERIENCE_STYLES,
    "education": colors.EDUCATION_STYLES,
    "skills": colors.SKILLS_STYLES,
    "projects": colors.PROJECTS_STYLES,
    "grouped": colors.GROUPED_STYLES,
}


@pytest.mark.parametrize("name", sorted(ALL_STYLE_MAPS))
def test_every_style_has_a_colour_and_a_label(name: str) -> None:
    for item_type, style in ALL_STYLE_MAPS[name].items():
        assert style.color.startswith("#") and len(style.color) == 7, item_type
        assert style.label, item_type


def test_shared_evidence_looks_the_same_in_every_section() -> None:
    """A date box must not change colour depending on which section drew it."""
    for name, styles in ALL_STYLE_MAPS.items():
        if "date" in styles:
            assert styles["date"] is colors.DATE, name
        if "url" in styles:
            assert styles["url"] is colors.URL, name
        if "location" in styles:
            assert styles["location"] is colors.LOCATION, name


def test_labels_name_the_evidence_not_just_the_field() -> None:
    """An overlay should say what found a box, not only what it is."""
    assert colors.DATE.label == "regex: date"
    assert colors.URL.label == "annotation: url"
    assert colors.EXPERIENCE_STYLES["job_title"].label.startswith("MiniLM:")
    assert colors.EXPERIENCE_STYLES["company"].label.startswith("NER:")


@pytest.mark.parametrize(
    ("styles", "section_type"),
    [
        (colors.SUMMARY_STYLES, "summary"),
        (colors.EXPERIENCE_STYLES, "experience"),
        (colors.EDUCATION_STYLES, "education"),
        (colors.SKILLS_STYLES, "skills"),
        (colors.PROJECTS_STYLES, "projects"),
    ],
)
def test_colour_and_label_maps_cover_the_same_item_types(
    styles: dict[str, colors.ItemStyle], section_type: str
) -> None:
    """A renderer that looks up a missing item type raises KeyError mid-render."""
    colour_map = colors.colors_of(styles, section_type)
    label_map = colors.labels_of(styles)
    assert set(colour_map) == set(label_map)
    assert "section_heading" in colour_map


@pytest.mark.parametrize("section_type", V1_SECTION_ORDER[1:])
def test_every_routed_section_has_a_heading_colour(section_type: str) -> None:
    """Grouped sections take their heading colour from the section registry."""
    assert colors.section_heading_style(section_type).color.startswith("#")


def test_profile_attributes_keep_their_header_overlay_colour() -> None:
    """The same attribute should look the same in the header and in a section."""
    header_colors = dict(SETTINGS.debug.header_entity_colors)
    assert colors.profile_attribute_color("gender", "#000000") == header_colors["gender"]
    assert colors.profile_attribute_color("not_an_attribute", "#123456") == "#123456"
