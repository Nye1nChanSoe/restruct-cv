"""Guards on the pass 1-3 overlays.

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
from restruct.debug.colors import LINE_STYLES, PHYSICAL_STYLES, WORD_STYLES, ItemStyle
from restruct.debug.stages import render_stage_overlays
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
    for styles in (PHYSICAL_STYLES, WORD_STYLES, LINE_STYLES):
        colours = [style.color for style in styles.values()]
        assert len(colours) == len(set(colours)), styles


def test_stage_styles_are_labelled_for_the_legend() -> None:
    """The legend is what makes an overlay self-describing."""
    for styles in (PHYSICAL_STYLES, WORD_STYLES, LINE_STYLES):
        for style in styles.values():
            assert style.label.strip()
            assert style.color.startswith("#") and len(style.color) == 7
