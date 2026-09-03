"""Guards on compound heading splitting.

The corpus already carries seven compound headings that resolve five different
ways, and each of those is pinned here against the reasoning recorded in the
label notes. The one mechanism no fixture reaches -- a local subheading owning
the lines under it -- is driven directly, because the heading detector promotes
a bold label to a section heading of its own before the run logic can see it.
"""

from __future__ import annotations

import pytest

from restruct.document.types import DetectedHeading, ExtractedLine
from restruct.structure.compound import (
    LogicalSection,
    logical_sections,
    split_heading,
)


def component_types(text: str) -> list[str | None]:
    return [component.section_type for component in split_heading(text)]


def component_texts(text: str) -> list[str]:
    return [component.text for component in split_heading(text)]


# -- splitting ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("CERTIFICATIONS & LANGUAGES", ["CERTIFICATIONS", "LANGUAGES"]),
        ("SAFETY & TRAINING", ["SAFETY", "TRAINING"]),
        ("Languages, Tools; Interests", ["Languages", "Tools", "Interests"]),
        ("Brand + Marketing", ["Brand", "Marketing"]),
        ("Design as well as Research", ["Design", "Research"]),
        ("Projects along with Publications", ["Projects", "Publications"]),
        ("Awards with Honours", ["Awards", "Honours"]),
        ("Skills plus Tools", ["Skills", "Tools"]),
    ],
)
def test_a_compound_heading_splits_on_every_named_separator(
    heading: str,
    expected: list[str],
) -> None:
    assert component_texts(heading) == expected


@pytest.mark.parametrize(
    "heading",
    [
        "TECHNICAL SKILLS",
        "PROFESSIONAL EXPERIENCE",
        "EDUCATION",
        "ADDITIONAL INFORMATION",
    ],
)
def test_an_ordinary_heading_yields_one_component(heading: str) -> None:
    assert len(split_heading(heading)) == 1


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("Education and Training", "education"),
        ("Awards and Honors", "awards"),
        ("Licenses and Certifications", "certifications"),
        ("Tools and Equipment", "tools_equipment"),
        ("Certificates and Training", "certifications"),
        ("Training and Certifications", "certifications"),
    ],
)
def test_a_known_section_name_is_never_split(heading: str, expected: str) -> None:
    """Several reference names contain a separator. Splitting one turns a
    heading the document got exactly right into two guesses."""
    components = split_heading(heading)
    assert len(components) == 1
    assert components[0].section_type == expected


def test_components_are_classified_only_on_an_exact_match() -> None:
    """"TRAINING" leans towards certifications and equally towards education.
    A lean is not evidence, and treating it as one is what sent resume 1's
    safety bullets into licenses."""
    assert component_types("SAFETY & TRAINING") == [None, None]
    assert component_types("TRAINING & CERTIFICATIONS") == [None, "certifications"]


def test_offsets_locate_each_component_in_the_original() -> None:
    """A split must stay reversible, the same contract header spans carry."""
    heading = "CERTIFICATIONS & LANGUAGES"
    for component in split_heading(heading):
        assert heading[component.start : component.end].strip() == component.text


# -- ownership ---------------------------------------------------------------


def line(text: str, *, size: float = 10.0, bold: bool = False) -> ExtractedLine:
    return ExtractedLine(
        page=1,
        text=text,
        bbox=(50.0, 100.0, 300.0, 112.0),
        size=size,
        bold=bold,
        used_ocr=False,
    )


def sections_for(heading_text: str, body: list[ExtractedLine]) -> list[LogicalSection]:
    lines = [line(heading_text, size=13.0, bold=True), *body]
    heading = DetectedHeading(
        line_index=0,
        section_type="certifications",
        similarity=0.9,
        runner_up_similarity=0.5,
    )
    return logical_sections(lines, heading, list(range(1, len(lines))))


def routed(sections: list[LogicalSection]) -> list[tuple[str, int]]:
    return [(section.section_type, len(section.line_indexes)) for section in sections]


def test_an_explicit_label_gives_each_component_its_own_line() -> None:
    """Resume 6: the document states which half each line belongs to."""
    sections = sections_for(
        "CERTIFICATIONS & LANGUAGES",
        [
            line("Certifications: AWS Certified Cloud Practitioner (2024)"),
            line("Languages: English (Professional), Thai (Conversational)"),
        ],
    )
    assert routed(sections) == [("certifications", 1), ("languages", 1)]


def test_a_local_subheading_owns_the_lines_under_it() -> None:
    """The one place where following a line is evidence, because writing a
    subheading is exactly the claim that what follows belongs to it."""
    sections = sections_for(
        "AWARDS & PUBLICATIONS",
        [
            line("Awards", bold=True),
            line("- Team Excellence Award, 2023"),
            line("- Regional Innovation Prize, 2022"),
            line("Publications", bold=True),
            line("- Partitioning Strategies for Retail Fact Tables, 2022"),
        ],
    )
    assert routed(sections) == [("awards", 3), ("publications", 2)]


def test_a_key_value_label_claims_only_its_own_line() -> None:
    """5.ocr: 'Thai: Native' says nothing about the 'Driving licence' line
    below it, so the remainder is uncertain and goes to others."""
    sections = sections_for(
        "LANGUAGES & ADDITIONAL INFORMATION",
        [
            line("- Thai: Native"),
            line("- English: Working proficiency"),
            line("- Driving licence: Thai private car licence"),
            line("- Availability: 30 days' notice"),
        ],
    )
    assert routed(sections) == [("languages", 2), ("others", 2)]


def test_nothing_claiming_itself_gives_the_section_to_the_first_destination() -> None:
    """2 and 7.anomaly: an undivided section is evidence for one destination,
    whichever order the heading named it in."""
    body = [line("- Working at Height Awareness"), line("- Basic First Aid")]
    assert routed(sections_for("CERTIFICATIONS & TRAINING", body)) == [("certifications", 2)]
    assert routed(sections_for("TRAINING & CERTIFICATIONS", body)) == [("certifications", 2)]


def test_no_component_naming_a_destination_sends_everything_to_others() -> None:
    """Resume 1. Neither 'SAFETY' nor 'TRAINING' names a destination, so
    nothing here can be trusted to route content."""
    sections = sections_for(
        "SAFETY & TRAINING",
        [line("- Basic Occupational Safety Training"), line("- Lockout/Tagout Awareness")],
    )
    assert routed(sections) == [("others", 2)]
    assert sections[0].compound_heading_text == "SAFETY & TRAINING"


def test_a_split_that_owns_nothing_produces_no_section() -> None:
    """8.compound: both local labels were promoted to headings of their own,
    leaving the compound heading with no content. An empty section would
    register a destination that owns nothing, and the real section of that
    type would then be the second occurrence and go unread."""
    assert sections_for("AWARDS & PUBLICATIONS", []) == []


def test_an_ordinary_heading_keeps_its_own_destination() -> None:
    """The single-component path must stay untouched; every non-compound
    section in the corpus depends on it."""
    lines = [line("TECHNICAL SKILLS", size=13.0, bold=True), line("Python, SQL")]
    heading = DetectedHeading(
        line_index=0,
        section_type="skills",
        similarity=0.9,
        runner_up_similarity=0.4,
    )
    sections = logical_sections(lines, heading, [1])
    assert routed(sections) == [("skills", 1)]
    assert sections[0].compound_heading_text is None
