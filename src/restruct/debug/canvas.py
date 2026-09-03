"""Shared Pillow primitives for every debug overlay.

One drawing surface and one box-drawing routine, so a box means the same thing
whichever pass drew it, and so the stage overlays and the section overlays stay
visually consistent as they converge.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

from restruct.configs import SETTINGS
from restruct.debug.colors import LABEL_BACKGROUND, ItemStyle
from restruct.geometry import pixel_box
from restruct.structure.resolver import is_model_backed

Box = tuple[float, float, float, float] | list[float]


def page_canvas(document: pymupdf.Document, page_number: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Render one page as the background for an overlay."""
    matrix = pymupdf.Matrix(SETTINGS.debug.scale, SETTINGS.debug.scale)
    pixmap = document[page_number - 1].get_pixmap(matrix=matrix, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return image, ImageDraw.Draw(image)


def outline(
    draw: ImageDraw.ImageDraw,
    box: Box,
    style: ItemStyle,
    *,
    width: int = 2,
    label: str | None = None,
) -> None:
    """Draw one box, optionally labelled above its top-left corner."""
    pixels = pixel_box(box)
    # A zero-width or inverted box would raise rather than draw.
    if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
        return
    draw.rectangle(pixels, outline=style.color, width=width)
    if label is not None:
        text_at(draw, (pixels[0] + SETTINGS.debug.label_x_padding,
                       max(0, pixels[1] - SETTINGS.debug.label_y_offset)), label, style.color)


def rule_line(draw: ImageDraw.ImageDraw, box: Box, style: ItemStyle, *, width: int = 3) -> None:
    """Draw a horizontal marker across a box, for baselines and drawn rules."""
    pixels = pixel_box(box)
    draw.line((pixels[0], pixels[3], pixels[2], pixels[3]), fill=style.color, width=width)


def text_at(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    color: str,
) -> None:
    """Draw a label on an opaque plate so it stays readable over dark content."""
    box = draw.textbbox(position, text)
    draw.rectangle((box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1), fill=LABEL_BACKGROUND)
    draw.text(position, text, fill=color)


def place_label(
    draw: ImageDraw.ImageDraw,
    *,
    position: tuple[int, int],
    text: str,
    color: str,
    avoid: list[tuple[int, int, int, int]],
    fallback: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Draw one label, moving it aside when it would land on something.

    ``avoid`` is every box the label must not cover -- the other items on the
    page and the labels already placed. Callers choose where a label prefers to
    sit, because that is their own policy; collision handling is shared, so an
    overlay never silently hides the thing it is annotating.

    Returns where the label ended up, to be added to the caller's ``avoid``.
    """
    box = draw.textbbox(position, text)
    if fallback is not None and any(
        box[0] < other[2] and other[0] < box[2] and box[1] < other[3] and other[1] < box[3]
        for other in avoid
    ):
        position = fallback
        box = draw.textbbox(position, text)
    draw.rectangle((box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1), fill=LABEL_BACKGROUND)
    draw.text(position, text, fill=color)
    return box


def legend(
    draw: ImageDraw.ImageDraw,
    styles: list[tuple[ItemStyle, int]],
    *,
    title: str,
) -> None:
    """Draw a key in the top-left corner naming what each colour means.

    Without this an overlay is a wall of coloured boxes; with it the image is
    self-describing, which is the point of rendering one at all.
    """
    x, y = 8, 8
    text_at(draw, (x, y), title, "#000000")
    y += 16
    for style, count in styles:
        draw.rectangle((x, y + 3, x + 10, y + 11), outline=style.color, width=3)
        text_at(draw, (x + 16, y), f"{style.label} ({count})", style.color)
        y += 16


def save(image: Image.Image, output_directory: Path, page_number: int) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    image.save(output_directory / f"page-{page_number}.png")


def stroke_width(item_type: str, detection_method: str) -> int:
    """How heavily to draw one item.

    Model-backed boxes must stay visually distinct from deterministic ones --
    a reader has to be able to tell at a glance whether a box is something the
    document said or something a model concluded. That question is asked of
    ``is_model_backed`` rather than of the method name's spelling, which is how
    ``ner_minilm_reconciliation`` and ``semantic_similarity`` were being drawn
    as though they were deterministic.

    Section headings keep their own weight: they are already distinguished by
    colour and by being headings, and the model that confirmed one says
    nothing useful about how thick its box should be.
    """
    if item_type == "section_heading":
        return SETTINGS.debug.heading_stroke_width
    if is_model_backed(detection_method):
        return SETTINGS.debug.header_entity_stroke_width + 2
    return 2
