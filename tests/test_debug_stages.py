"""Guards on the pass 1-4 overlays.

These images are the only way to check reconstruction geometry -- a count can be
right while every box sits ten points too low -- so the tests here protect that
they render at all, cover every page, and survive the degenerate boxes real
documents produce.

The bug that motivated rendering them is pinned in test_stats.py: five table
cells were being flagged as running footers, which was invisible in JSON and
obvious the moment the overlay was drawn.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from restruct.debug import canvas
from restruct.debug.colors import (
    LINE_STYLES,
    PHYSICAL_STYLES,
    SECTION_STYLES,
    WORD_STYLES,
    ItemStyle,
)
from restruct.debug.stages import render_sections, render_stage_overlays
from restruct.document.stats import measure
from restruct.ingestion.native import read_document
from restruct.layout.words import reconstruct_words
from tests.helpers import SYNTHETIC_DIRECTORY, tesseract_available

STAGE_DIRECTORIES = ("pass-1-physical", "pass-2-words", "pass-3-lines")


def rendered(stem: str, destination: Path):
    pdf = pymupdf.open(SYNTHETIC_DIRECTORY / f"{stem}.pdf")
    document = read_document(pdf)
    statistics = measure(document)
    document = reconstruct_words(document, statistics)
    render_stage_overlays(pdf, document, statistics, destination)
    return document


@pytest.mark.parametrize("stem", ["1", "6"])
def test_every_pass_renders_every_page(stem: str, tmp_path: Path) -> None:
    document = rendered(stem, tmp_path)
    for directory in STAGE_DIRECTORIES:
        for page in document.pages:
            image_path = tmp_path / directory / f"page-{page.number}.png"
            assert image_path.exists(), image_path
            with Image.open(image_path) as image:
                assert image.width > 100 and image.height > 100


def test_overlays_draw_onto_the_page_rather_than_a_blank_canvas(tmp_path: Path) -> None:
    """A blank overlay would pass a file-exists check while showing nothing."""
    rendered("6", tmp_path)
    with Image.open(tmp_path / "pass-3-lines" / "page-1.png") as image:
        colours = image.convert("RGB").getcolors(maxcolors=1_000_000) or []
    assert len(colours) > 50, "overlay looks blank"


def test_selecting_stages_renders_only_those(tmp_path: Path) -> None:
    pdf = pymupdf.open(SYNTHETIC_DIRECTORY / "6.pdf")
    document = read_document(pdf)
    statistics = measure(document)
    document = reconstruct_words(document, statistics)
    render_stage_overlays(pdf, document, statistics, tmp_path, stages=(2,))
    assert (tmp_path / "pass-2-words").exists()
    assert not (tmp_path / "pass-1-physical").exists()
    assert not (tmp_path / "pass-3-lines").exists()


@pytest.mark.skipif(not tesseract_available(), reason="needs tesseract")
def test_a_scanned_page_renders_all_three_passes(tmp_path: Path) -> None:
    """OCR pages have no glyph origins and no character boxes; the overlays
    must degrade rather than raise."""
    rendered("9.ocr", tmp_path)
    for directory in STAGE_DIRECTORIES:
        assert (tmp_path / directory / "page-1.png").exists()


# -- canvas primitives ------------------------------------------------------


def test_outline_ignores_a_degenerate_box(tmp_path: Path) -> None:
    """Zero-width boxes occur in real documents and would otherwise raise."""
    pdf = pymupdf.open(SYNTHETIC_DIRECTORY / "6.pdf")
    image, draw = canvas.page_canvas(pdf, 1)
    style = ItemStyle("#FF0000", "test")
    canvas.outline(draw, (10.0, 10.0, 10.0, 10.0), style)      # zero area
    canvas.outline(draw, (50.0, 50.0, 10.0, 10.0), style)      # inverted
    canvas.save(image, tmp_path, 1)
    assert (tmp_path / "page-1.png").exists()


def test_every_stage_style_has_a_distinct_colour() -> None:
    """Two layers sharing a colour makes an overlay unreadable."""
    for styles in (PHYSICAL_STYLES, WORD_STYLES, LINE_STYLES, SECTION_STYLES):
        colours = [style.color for style in styles.values()]
        assert len(colours) == len(set(colours)), styles


def test_stage_styles_are_labelled_for_the_legend() -> None:
    """The legend is what makes an overlay self-describing."""
    for styles in (PHYSICAL_STYLES, WORD_STYLES, LINE_STYLES, SECTION_STYLES):
        for style in styles.values():
            assert style.label.strip()
            assert style.color.startswith("#") and len(style.color) == 7


# -- pass 4 -----------------------------------------------------------------


def _section(section_type: str, *, compound: str | None = None) -> dict:
    heading = {"text": "HEADING", "page": 1, "bbox": [50.0, 100.0, 300.0, 115.0]}
    if compound is not None:
        heading["compoundHeadingText"] = compound
    return {
        "sectionType": section_type,
        "heading": heading,
        "content": [
            {"type": "paragraph", "text": "body", "page": 1, "bbox": [50.0, 120.0, 300.0, 132.0]},
            {"type": "bullet", "text": "item", "page": 1, "bbox": [50.0, 134.0, 300.0, 146.0]},
        ],
    }


def test_pass_four_renders_every_page(tmp_path: Path) -> None:
    pdf = pymupdf.open(SYNTHETIC_DIRECTORY / "6.pdf")
    render_sections(pdf, [_section("skills")], tmp_path)
    for page_number in range(1, pdf.page_count + 1):
        assert (tmp_path / f"page-{page_number}.png").exists()


def test_both_halves_of_a_split_heading_stay_visible(tmp_path: Path) -> None:
    """Two sections from one heading line draw at the same coordinates. Drawn
    plainly, the second label covers the first and the overlay silently claims
    the split produced one destination."""
    pdf = pymupdf.open(SYNTHETIC_DIRECTORY / "6.pdf")
    compound = "CERTIFICATIONS & LANGUAGES"
    sections = [
        _section("certifications", compound=compound),
        _section("languages", compound=compound),
    ]
    render_sections(pdf, sections, tmp_path / "split")
    render_sections(pdf, sections[:1], tmp_path / "single")

    with Image.open(tmp_path / "split" / "page-1.png") as split_image:
        split_pixels = list(split_image.convert("RGB").getdata())
    with Image.open(tmp_path / "single" / "page-1.png") as single_image:
        single_pixels = list(single_image.convert("RGB").getdata())
    assert split_pixels != single_pixels, "second half of the split drew nothing new"


# -- one renderer -----------------------------------------------------------


def test_a_model_backed_box_is_drawn_more_heavily_than_a_deterministic_one() -> None:
    """A reader has to be able to tell at a glance whether a box is something
    the document said or something a model concluded."""
    from restruct.debug.canvas import stroke_width

    assert stroke_width("job_title", "semantic_similarity") > stroke_width(
        "date", "date_regex"
    )
    assert stroke_width("company", "distilbert_ner") > stroke_width(
        "bullet", "bullet_marker"
    )


def test_a_section_heading_keeps_its_own_weight() -> None:
    """Headings are already distinguished by colour and by being headings, and
    the model that confirmed one says nothing about how thick to draw it."""
    from restruct.debug.canvas import stroke_width

    assert stroke_width("section_heading", "geometry_semantic") == stroke_width(
        "section_heading", "geometry_typography"
    )


def test_a_label_moves_aside_rather_than_covering_what_it_annotates(
    tmp_path: Path,
) -> None:
    pdf = pymupdf.open(SYNTHETIC_DIRECTORY / "6.pdf")
    _, draw = canvas.page_canvas(pdf, 1)
    blocked = draw.textbbox((100, 100), "location")
    placed = canvas.place_label(
        draw,
        position=(100, 100),
        text="location",
        color="#EF6C00",
        avoid=[blocked],
        fallback=(100, 400),
    )
    assert placed[1] >= 400


def test_a_label_stays_put_when_nothing_is_in_the_way(tmp_path: Path) -> None:
    pdf = pymupdf.open(SYNTHETIC_DIRECTORY / "6.pdf")
    _, draw = canvas.page_canvas(pdf, 1)
    placed = canvas.place_label(
        draw,
        position=(100, 100),
        text="location",
        color="#EF6C00",
        avoid=[(900, 900, 1000, 1000)],
        fallback=(100, 400),
    )
    assert placed[1] < 400
