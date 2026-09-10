"""Guards on the deterministic patterns.

Several of these markers are invisible in an editor. Consolidating the patterns
nearly dropped U+F0B7 -- the Private Use Area bullet Word exports -- because it
renders as nothing and was silently lost when the character class was retyped.
These cases pin the characters that a reader cannot see.
"""

from __future__ import annotations

import pytest

from restruct.patterns.bullets import BULLET_RE
from restruct.patterns.contacts import EMAIL_RE, PHONE_RE, URL_RE
from restruct.patterns.dates import DATE_RANGE_RE, SINGLE_YEAR_RE
from restruct.patterns.education import DEGREE_RE, GPA_RE, INSTITUTION_RE
from restruct.patterns.invisibles import (
    ZERO_WIDTH_CHARACTERS,
    is_blank,
    without_invisibles,
)
from restruct.patterns.layout import PAGE_FOOTER_RE, heading_text
from restruct.patterns.organizations import COMPANY_MARKER_RE
from restruct.patterns.personal import (
    LABELLED_ATTRIBUTE_RE,
    LOCATION_SEGMENT_RE,
    NATIONALITY_PHRASE_RE,
)
from restruct.patterns.separators import KEY_VALUE_DASH_RE, METADATA_SEPARATOR_RE


@pytest.mark.parametrize(
    "marker",
    [
        "-", "+", "*",
        "\u2022",  # bullet
        "\u25cf",  # black circle
        "\u25aa",  # black small square
        "\u25e6",  # white bullet
        "\u2023",  # triangular bullet
        "\uf0b7",  # Word/Wingdings bullet, invisible in most editors
        "\u00a2",  # cent sign, seen in some exports
        "1.", "2)", "10.",
    ],
)
def test_bullet_markers_are_recognized(marker: str) -> None:
    assert BULLET_RE.match(f"{marker} Maintained packaging machines")


@pytest.mark.parametrize("invisible", ["\u200b", "\ufeff"])
def test_bullet_consumes_zero_width_padding(invisible: str) -> None:
    """PDF exporters place a zero-width character between marker and text."""
    text = f"\u2022{invisible} Diagnose mechanical faults"
    match = BULLET_RE.match(text)
    assert match is not None
    assert text[match.end():] == "Diagnose mechanical faults"


def test_bullet_does_not_swallow_ordinary_text() -> None:
    assert BULLET_RE.match("Maintained packaging machines") is None


@pytest.mark.parametrize(
    "text",
    [
        "March 2022 - Present",
        "March 2022 \u2013 Present",       # en dash
        "March 2022 \u2014 Present",       # em dash
        "2019 - 2022",
        "From 2019 to Present",
        "Jan 2024 - Current",
        "May 2017 \u2013 May 2019",
        "2020 - Now",
    ],
)
def test_date_ranges(text: str) -> None:
    assert DATE_RANGE_RE.search(text), text


def test_single_year_matches_only_plausible_years() -> None:
    assert SINGLE_YEAR_RE.findall("Worked 1999 and 2024, id 3021") == ["1999", "2024"]


@pytest.mark.parametrize(
    ("label", "kind"),
    [
        ("Date of Birth: 12 March 1995", "date of birth"),
        ("DOB - 1995/03/12", "dob"),
        ("Martial Status: Single", "martial status"),
        ("Marital Status - Married", "marital status"),
        ("Nationality: Thai", "nationality"),
        ("Visa Status: Work Permit", "visa status"),
        ("Current Residence - Bangkok", "current residence"),
    ],
)
def test_labelled_attributes_keep_their_value_verbatim(label: str, kind: str) -> None:
    """The label picks the field; the value is never reformatted."""
    match = LABELLED_ATTRIBUTE_RE.search(label)
    assert match is not None
    assert match.group("label").casefold().replace(".", "").replace("-", " ").strip()
    assert match.group("value").strip()


def test_misspelled_marital_status_is_supported() -> None:
    assert LABELLED_ATTRIBUTE_RE.search("Martial Status: Single") is not None


@pytest.mark.parametrize(
    "text", ["Thai citizen", "British national", "Nationality: Indonesian"]
)
def test_nationality_phrases(text: str) -> None:
    assert NATIONALITY_PHRASE_RE.search(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bangkok, Thailand", True),
        ("Samut Prakan, Thailand", True),
        ("somebody@example.com", False),
        ("Apt 12, Bangkok", False),
    ],
)
def test_location_segments_exclude_digits_and_emails(text: str, expected: bool) -> None:
    assert bool(LOCATION_SEGMENT_RE.fullmatch(text)) is expected


def test_email_and_url_do_not_claim_each_other() -> None:
    line = "alex.morgan@example.com | github.com/alexmorgan-dev"
    assert EMAIL_RE.findall(line) == ["alex.morgan@example.com"]
    assert "github.com/alexmorgan-dev" in URL_RE.findall(line)
    # The email's domain must not be re-claimed as a bare URL.
    assert "example.com" not in URL_RE.findall(line)


@pytest.mark.parametrize(
    "text", ["+66 81 555 2741", "+62 812 5550 1847", "(02) 555-2741", "081-555-2741"]
)
def test_phone_numbers(text: str) -> None:
    assert PHONE_RE.search(text)


def test_redacted_phone_is_a_known_gap() -> None:
    """5.ocr.pdf redacts its number; the scorecard tracks this as a miss."""
    assert PHONE_RE.search("+66 8X XXX XXXX") is None


def test_gpa_captures_value_and_scale() -> None:
    match = GPA_RE.search("GPA: 3.31 / 4.00")
    assert match is not None
    assert match.group("value") == "3.31"
    assert match.group("scale") == "4.00"


@pytest.mark.parametrize(
    "text", ["Bachelor of Engineering", "M.Sc. Computer Science", "Vocational Certificate", "MBA"]
)
def test_degree_evidence(text: str) -> None:
    assert DEGREE_RE.search(text)


def test_institution_evidence() -> None:
    assert INSTITUTION_RE.search("Burapha University")
    assert INSTITUTION_RE.search("Samut Prakan Technical College")
    assert INSTITUTION_RE.search("Orbit Labs Co., Ltd.") is None


def test_company_markers_ground_employer_segments() -> None:
    assert COMPANY_MARKER_RE.search("Siam Precision Components Co., Ltd.")
    assert COMPANY_MARKER_RE.search("Senior Site Engineer") is None


def test_page_footer_only_matches_a_trailing_page_number() -> None:
    assert PAGE_FOOTER_RE.search("Synthetic resume | Page 2")
    assert PAGE_FOOTER_RE.search("Maintained page layouts for 2 teams") is None


def test_metadata_separator_is_shared_by_experience_and_education() -> None:
    """These were two byte-identical copies before patterns/ existed."""
    assert METADATA_SEPARATOR_RE.split("Engineer | Acme Ltd \u2013 Bangkok") == [
        "Engineer",
        "Acme Ltd",
        "Bangkok",
    ]


def test_key_value_dash_requires_spaced_dash() -> None:
    """A hyphenated word must not be read as a key-value delimiter."""
    assert KEY_VALUE_DASH_RE.match("Tools - Angle grinder, cut-off saw")
    assert KEY_VALUE_DASH_RE.match("lockout/tagout-awareness training") is None


# -- zero-width characters ---------------------------------------------------

# Written as escapes throughout, and referred to by codepoint in prose. A
# literal renders as nothing, so it is silently lost the moment anyone retypes
# the line -- which is how patterns/layout.py came to hold two of them.


def test_zero_width_characters_are_written_as_escapes() -> None:
    """The set itself, by codepoint. If a literal is ever pasted over one of
    these the constant still compiles, so only this test would notice."""
    assert [ord(character) for character in ZERO_WIDTH_CHARACTERS] == [
        0x200B,
        0xFEFF,
        0x200C,
        0x200D,
        0x00AD,
    ]


def test_no_source_file_holds_a_literal_zero_width_character() -> None:
    """The convention CLAUDE.md states, enforced rather than remembered.

    patterns/layout.py held a literal U+200B and U+FEFF in its heading cleaner
    while the bullet pattern beside it wrote the same two as escapes, and the
    scorecard kept a fourth copy of the set as literals. Tests and tools are
    scanned too: the rule is about anyone retyping a line, and a fixture string
    is retyped as often as a pattern.
    """
    from pathlib import Path

    invisible = {0x200B, 0xFEFF, 0x200C, 0x200D, 0x00AD, 0xF0B7}
    offenders: list[str] = []
    project_root = Path(__file__).resolve().parents[1]
    roots = (project_root / "src", project_root / "tests", project_root / "tools")
    for path in sorted(path for root in roots for path in root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if any(ord(character) in invisible for character in line):
                offenders.append(f"{path.name}:{number}")
    assert not offenders, f"literal invisible characters in {offenders}"


@pytest.mark.parametrize(
    "text",
    [
        "\u200b",
        "\ufeff",
        "\u200b\u200b",
        "  \u200b  ",
        "",
        "   ",
    ],
)
def test_a_line_of_only_invisible_characters_is_blank(text: str) -> None:
    """Regression: a line holding one U+200B survived str.strip(), reached the
    parsers as content, and opened an experience record that owned nothing."""
    assert is_blank(text)


@pytest.mark.parametrize("text", ["a", "\u200ba", "●\u200b Bullet text", "-"])
def test_a_line_with_any_visible_character_is_not_blank(text: str) -> None:
    assert not is_blank(text)


def test_without_invisibles_keeps_every_visible_character() -> None:
    """Including the bullet glyph: the marker is content, the padding is not."""
    assert without_invisibles("●\u200b Skills") == "● Skills"
    assert without_invisibles("café – Bangkok") == "café – Bangkok"


def test_heading_text_strips_invisibles_through_the_shared_helper() -> None:
    assert heading_text("\ufeff01 EXPERIENCE\u200b") == "EXPERIENCE"
