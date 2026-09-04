"""Guards on drawing a resume back out of its JSON.

The reconstruction is read instead of the JSON, so the way it fails is by
looking fine: a value that never reaches the page is indistinguishable from a
value the resume never had. Most of these tests are therefore about content
arriving -- every populated field reaching the page, and anything the renderer
cannot place being shown rather than dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from restruct.debug.reconstruct import (
    Block,
    choose_font,
    render_resume,
    render_resume_file,
    resume_blocks,
)
from tests.helpers import PROJECT_ROOT

FULL_RESUME: dict[str, Any] = {
    "schema_version": "1.0",
    "header_profile": {
        "name": "Somchai Rattanakul",
        "job_titles": ["Maintenance Technician"],
        "location": "Samut Prakan, Thailand",
        "date_of_birth": "12/05/1995",
        "age": "29",
        "gender": "Male",
        "marital_status": "Single",
        "visa_status": "Work permit",
        "nationality": "Thai",
        "current_residence": "Bangkok",
        "current_income": "45,000 THB",
        "current_package": "Housing included",
        "emails": ["somchai@example.com"],
        "phones": ["+66 81 555 2741"],
        "urls": [{"text": "portfolio", "url": "example.com/somchai"}],
    },
    "summary": {
        "content": [
            {"type": "paragraph", "text": "Seven years maintaining machinery."},
            {"type": "bullet", "text": "Preventive maintenance"},
        ]
    },
    "experience": [
        {
            "job_titles": ["Maintenance Technician"],
            "companies": ["Siam Precision"],
            "dates": ["March 2022 - Present"],
            "locations": ["Bang Phli"],
            "urls": [{"text": "siam.example.com", "url": "siam.example.com"}],
            "paragraphs": ["Ran the shift."],
            "bullets": ["Replaced bearings"],
        }
    ],
    "education": [
        {
            "titles": ["Vocational Certificate"],
            "institutions": ["Bangkok Technical"],
            "dates": ["2014"],
            "locations": ["Bangkok"],
            "gpa": ["3.4"],
            "skills": ["Welding"],
            "urls": [],
            "paragraphs": ["Coursework in welding."],
            "bullets": ["Workshop safety"],
        }
    ],
    "skills": [
        {
            "subheading": "Welding",
            "paragraphs": ["MIG/MAG, TIG"],
            "bullets": ["Fillet joints"],
            "urls": [],
        }
    ],
    "projects": None,
    "certifications": [
        {
            "subheadings": ["Safety Passport"],
            "dates": ["2024"],
            "urls": [],
            "paragraphs": ["Issued by the institute."],
            "bullets": ["Renewed annually"],
        }
    ],
    "licenses": [],
    "tools_equipment": [],
    "languages": [],
    "volunteering": [],
    "awards": [],
    "publications": [],
    "references": [],
    "interests": [],
    "others": [
        {
            "heading": "ADDITIONAL INFORMATION",
            "entries": [
                {
                    "subheadings": ["Availability"],
                    "attributes": [{"type": "notice_period", "value": "30 days"}],
                    "dates": [],
                    "urls": [],
                    "paragraphs": ["Available immediately."],
                    "bullets": ["Willing to relocate"],
                }
            ],
        }
    ],
}


def _drawn_text(directory: Path) -> str:
    """Everything the rendered PDF actually put on a page."""
    with pymupdf.open(directory / "reconstruction.pdf") as document:
        return "\n".join(page.get_text() for page in document)


def _leaf_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _leaf_strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _leaf_strings(item)]
    return []


# -- nothing populated may go missing ----------------------------------------


def test_every_populated_value_reaches_the_page(tmp_path: Path) -> None:
    """The failure mode this tool has is looking complete while being
    incomplete: a value that never reaches the page cannot be told apart from
    a value the resume never had."""
    render_resume(FULL_RESUME, tmp_path)
    drawn = _drawn_text(tmp_path).replace("\n", " ")
    missing = [
        text
        for text in _leaf_strings(FULL_RESUME)
        if text not in {"1.0", "paragraph", "bullet", "notice_period"}
        and text not in drawn
    ]
    assert not missing, f"never drawn: {missing}"


def test_a_section_that_is_present_is_named(tmp_path: Path) -> None:
    render_resume(FULL_RESUME, tmp_path)
    drawn = _drawn_text(tmp_path)
    for title in ("SUMMARY", "EXPERIENCE", "EDUCATION", "SKILLS", "CERTIFICATIONS"):
        assert title in drawn
    # `others` keeps the heading the document wrote, which is the point of it.
    assert "ADDITIONAL INFORMATION" in drawn


# -- what is absent stays absent ---------------------------------------------


def test_absent_and_empty_sections_draw_nothing(tmp_path: Path) -> None:
    """A page of "none" rows is a page nobody proof-reads."""
    render_resume(FULL_RESUME, tmp_path)
    drawn = _drawn_text(tmp_path)
    for title in ("PROJECTS", "LICENSES", "LANGUAGES", "AWARDS", "INTERESTS"):
        assert title not in drawn
    assert "None" not in drawn and "null" not in drawn


def test_an_empty_resume_still_draws_a_page(tmp_path: Path) -> None:
    """Nothing extracted is itself a result worth looking at, and a run that
    wrote no file at all would read as a crash."""
    empty = {"schema_version": "1.0", "header_profile": None, "others": []}
    written = render_resume(empty, tmp_path)
    assert (tmp_path / "reconstruction.pdf").exists()
    assert any(path.suffix == ".png" for path in written)


# -- content with nowhere to go is shown, never dropped ----------------------


def test_a_field_the_renderer_cannot_place_is_drawn_anyway(tmp_path: Path) -> None:
    """Adding a schema field and forgetting to teach this module would
    otherwise make the field invisible -- and invisible, in a proof-reading
    tool, reads as an extraction failure."""
    resume = dict(FULL_RESUME, security_clearance="Level 2")
    blocks = resume_blocks(resume)
    assert Block("unplaced_heading", "Unplaced") in blocks
    render_resume(resume, tmp_path)
    drawn = _drawn_text(tmp_path)
    assert "UNPLACED" in drawn and "Level 2" in drawn


def test_a_known_shape_is_not_reported_as_unplaced(tmp_path: Path) -> None:
    """A false alarm here trains the reader to ignore the real one."""
    assert not [block for block in resume_blocks(FULL_RESUME) if block.style == "unplaced"]


@pytest.mark.parametrize(
    "stem", ["1", "2", "6", "7.anomaly", "8.compound", "9.ocr", "10.tight", "11"]
)
def test_no_committed_result_reports_unplaced_content(stem: str) -> None:
    """The corpus is the widest sample of real output there is; if the renderer
    has fallen behind the schema, it shows up here first."""
    path = PROJECT_ROOT / "results" / stem / "resume.json"
    if not path.exists():  # a corpus that has not been regenerated
        pytest.skip(f"{stem} has no committed result")
    resume = json.loads(path.read_text(encoding="utf-8"))
    unplaced = [block.text for block in resume_blocks(resume) if block.style == "unplaced"]
    assert not unplaced, f"{stem}: {unplaced}"


# -- the font ----------------------------------------------------------------


def test_the_font_is_chosen_against_the_text_it_must_draw() -> None:
    """Which font covers a document is a property of the document. The base-14
    faces draw an en dash, a curly quote and Thai all as a middle dot, which in
    a proof-reading tool would be read as an extraction bug."""
    chosen = choose_font(set("Rattanakul – · “ ปวส"))
    assert not chosen.missing, f"no installed font covers: {chosen.missing}"


def test_missing_glyphs_are_reported_rather_than_drawn_silently(
    tmp_path: Path,
) -> None:
    """A glyph the font cannot draw must not be mistaken for a lost value, so
    the page says which characters it could not render."""
    resume = dict(FULL_RESUME)
    chosen = choose_font({"\U0001f600"})  # no text font carries an emoji
    if not chosen.missing:
        pytest.skip("this machine has a font covering the probe character")
    resume["header_profile"] = dict(FULL_RESUME["header_profile"], name="Somchai \U0001f600")
    render_resume(resume, tmp_path)
    assert "missing from" in _drawn_text(tmp_path)


# -- the file, and only the file ---------------------------------------------


def test_it_draws_the_file_that_was_written(tmp_path: Path) -> None:
    """Reading the written file rather than the pipeline's own dictionary is
    what makes this a check on what was actually published."""
    resume_path = tmp_path / "resume.json"
    resume_path.write_text(json.dumps(FULL_RESUME), encoding="utf-8")
    written = render_resume_file(resume_path, tmp_path / "drawn")
    assert (tmp_path / "drawn" / "reconstruction.pdf") in written
    assert "Somchai Rattanakul" in _drawn_text(tmp_path / "drawn")


def test_only_the_glyphs_drawn_are_embedded(tmp_path: Path) -> None:
    """A font chosen for coverage is a large file -- Arial Unicode is 23MB --
    and embedding it whole made a two-page resume too big to open on a phone
    or send anywhere."""
    render_resume(FULL_RESUME, tmp_path)
    assert (tmp_path / "reconstruction.pdf").stat().st_size < 1_000_000


def test_a_long_resume_paginates(tmp_path: Path) -> None:
    resume = dict(FULL_RESUME)
    resume["experience"] = [
        dict(FULL_RESUME["experience"][0], bullets=[f"Did thing {n}" for n in range(40)])
        for _ in range(4)
    ]
    written = render_resume(resume, tmp_path)
    pages = [path for path in written if path.suffix == ".png"]
    assert len(pages) > 1
    with pymupdf.open(tmp_path / "reconstruction.pdf") as document:
        assert document.page_count == len(pages)
        assert "page 1 of" in document[0].get_text()
