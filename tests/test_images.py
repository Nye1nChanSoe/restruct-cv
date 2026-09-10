"""Guards on reading a resume that arrives as a photograph or a screenshot.

An image is a scanned page that never reached a PDF, so almost all of this is
about it *becoming* one: the page it is given must be measured in points, must
keep the image's proportions, and must be the right way up. Those are the three
things that decide whether every geometric rule downstream is reading a number
that means what it says, and none of them needs an OCR engine to check.

The fixtures are rendered from a committed resume rather than stored as images,
so this file adds no binaries to the repository and the text it expects to read
back is the text of a resume that is already ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf
import pytest

from restruct import cli
from restruct.configs import SETTINGS
from restruct.errors import InvalidDocument, TesseractMissing
from restruct.ingestion import ocr as ocr_module
from restruct.ingestion.image import read_image
from tests.helpers import SYNTHETIC_DIRECTORY, tesseract_available


def render_page(destination: Path, *, dpi: int = 150, keep_dpi: bool = False) -> Path:
    """Page one of a committed resume, written out as an image file.

    ``keep_dpi`` off is the ordinary case rather than the awkward one: a phone
    photo and most screenshots carry no resolution at all, which is exactly the
    case where a page box taken from the file would be a guess.
    """
    with pymupdf.open(SYNTHETIC_DIRECTORY / "1.pdf") as document:
        pixmap = document[0].get_pixmap(dpi=dpi, alpha=False)
    if not keep_dpi:
        pixmap.set_dpi(0, 0)
    pixmap.save(destination)
    return destination


# -- the page an image is given ---------------------------------------------


def test_the_page_is_measured_in_points_rather_than_in_pixels(tmp_path: Path) -> None:
    """The whole reason this reader exists. MuPDF opens an image as a page whose
    box is its pixel count, and every rule downstream that reads a box as points
    would then be reading a number that does not mean what it says."""
    source = render_page(tmp_path / "page.png", dpi=150)
    with pymupdf.open(source) as opened_directly:
        assert opened_directly[0].rect.width > 1000  # pixels, called points

    with read_image(source) as document:
        page = document[0]
        assert page.rect.height == pytest.approx(
            SETTINGS.image.page_long_side_points
        )
        assert page.rect.width < page.rect.height


def test_the_proportions_of_the_image_are_kept_exactly(tmp_path: Path) -> None:
    """Nothing about the image is invented; only the units are chosen. A page
    of a different shape would stretch the text and mislead every measurement
    taken from it."""
    source = render_page(tmp_path / "page.png", dpi=150)
    with pymupdf.open(source) as opened_directly:
        pixel_aspect = (
            opened_directly[0].rect.width / opened_directly[0].rect.height
        )
    with read_image(source) as document:
        page_aspect = document[0].rect.width / document[0].rect.height
    assert page_aspect == pytest.approx(pixel_aspect, rel=1e-3)


@pytest.mark.parametrize("dpi", [72, 150, 300, 400])
def test_the_render_does_not_grow_with_the_source(tmp_path: Path, dpi: int) -> None:
    """A page four times the pixels must not cost four times the render.

    This is the failure the reader was written for: a metadata-free photograph
    opened directly is a page of several thousand 'points', and rendering that
    at the OCR DPI produced an 85-megapixel raster of pixels that were never
    in the file.
    """
    source = render_page(tmp_path / f"page-{dpi}.png", dpi=dpi)
    with read_image(source) as document:
        pixmap = document[0].get_pixmap(dpi=SETTINGS.ocr.dpi, alpha=False)
    assert max(pixmap.width, pixmap.height) == pytest.approx(3508, abs=8)


# -- which way up ------------------------------------------------------------


def store_sideways(
    source: Path,
    destination: Path,
    *,
    orientation: int,
    turn: int,
) -> Path:
    """The same pixels a camera writes: turned, plus a tag saying so."""
    from PIL import Image

    with Image.open(source) as image:
        rotated = image.transpose(turn)
        exif = rotated.getexif()
        exif[274] = orientation
        rotated.save(destination, exif=exif, quality=90)
    return destination


def page_pixels(source: Path) -> Any:
    """The page this reader produces, as greyscale values to compare."""
    import numpy
    from PIL import Image

    with read_image(source) as document:
        pixmap = document[0].get_pixmap(dpi=72, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return numpy.asarray(image.convert("L"), dtype=float)


@pytest.mark.parametrize(
    ("orientation", "turn"),
    [(6, 2), (8, 4), (3, 3)],  # ROTATE_90, ROTATE_270, ROTATE_180
)
def test_a_photo_taken_sideways_is_turned_upright(
    tmp_path: Path,
    orientation: int,
    turn: int,
) -> None:
    """A phone stores the sensor's pixels and a tag saying which way it was
    held, so a portrait photo is stored in landscape. MuPDF reads that tag when
    it opens an image as a document and ignores it when it places one on a
    page, which is the path this reader takes.

    Compared against the upright page rather than by shape, because a page can
    be the right shape and still be upside down. The scale is set by the
    control below it: turning the upright page costs 23 grey levels, so single
    figures is the same page and nothing else is.
    """
    import numpy

    upright = render_page(tmp_path / "upright.png", dpi=150)
    sideways = store_sideways(
        upright,
        tmp_path / f"sideways-{orientation}.jpg",
        orientation=orientation,
        turn=turn,
    )
    expected = page_pixels(upright)
    turned = page_pixels(sideways)

    assert turned.shape == expected.shape
    assert numpy.abs(turned - expected).mean() < 8.0
    # The same page rotated is what a mis-read tag would produce.
    assert numpy.abs(turned - numpy.rot90(expected, 2)).mean() > 15.0


def test_a_mirrored_orientation_is_left_as_it_was(tmp_path: Path) -> None:
    """Orientation 2 is a flip, not a turn. Rare enough that guessing at it is
    worse than leaving it: turning a page the wrong way round loses it."""
    upright = render_page(tmp_path / "upright.png", dpi=150)
    mirrored = store_sideways(
        upright,
        tmp_path / "mirrored.jpg",
        orientation=2,
        turn=0,  # Image.FLIP_LEFT_RIGHT keeps the shape; 0 is a no-op transpose
    )
    with read_image(mirrored) as document:
        assert document[0].rect.width < document[0].rect.height


# -- what cannot be read -----------------------------------------------------


def test_an_image_needs_the_ocr_engine_before_the_models_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every other format degrades without Tesseract -- a PDF still has its
    native text, a DOCX never needed one. An image has nothing else, so the
    check belongs with the other things that can be answered before several
    hundred megabytes of weights are read."""
    monkeypatch.setattr(ocr_module.shutil, "which", lambda command: None)
    monkeypatch.setattr(ocr_module, "_known_install_locations", tuple)
    source = render_page(tmp_path / "page.png")

    with pytest.raises(TesseractMissing):
        cli._validate(source)


def test_a_pdf_still_reads_without_the_ocr_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the rule above, and the one that would hurt: making
    OCR mandatory for every format is exactly what the check must not do."""
    monkeypatch.setattr(ocr_module.shutil, "which", lambda command: None)
    monkeypatch.setattr(ocr_module, "_known_install_locations", tuple)
    cli._validate(SYNTHETIC_DIRECTORY / "1.pdf")


def test_a_file_that_is_not_an_image_reports_itself(tmp_path: Path) -> None:
    """Right extension, wrong contents -- the same distinction the corrupt PDF
    test draws: a conversion problem rather than a format one."""
    source = tmp_path / "broken.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n and then nothing that decodes")
    with pytest.raises(InvalidDocument):
        read_image(source)
    assert (
        cli.main([str(source), "-o", str(tmp_path / "out.json")])
        == cli.EXIT_INVALID_DOCUMENT
    )


def test_an_image_format_this_version_does_not_read_is_refused(
    tmp_path: Path,
) -> None:
    """TIFF, GIF, HEIC and the rest are formats a resume rarely arrives as, and
    each one claimed is a fixture and a test owed."""
    source = tmp_path / "resume.gif"
    source.write_bytes(b"GIF89a")
    assert (
        cli.main([str(source), "-o", str(tmp_path / "out.json")])
        == cli.EXIT_UNSUPPORTED_FORMAT
    )


# -- reading one, end to end -------------------------------------------------


@pytest.mark.skipif(not tesseract_available(), reason="needs tesseract")
@pytest.mark.parametrize("extension", [".png", ".jpg"])
def test_a_photographed_page_extracts_what_the_page_says(
    tmp_path: Path,
    models,
    extension: str,
) -> None:
    """The point of the whole feature, checked against the contact details of a
    resume whose text is already ground truth."""
    from tests.helpers import run_pipeline

    source = render_page(tmp_path / f"page{extension}", dpi=200)
    resume = run_pipeline(source, tmp_path / "workspace", models)
    profile = resume["header_profile"]

    assert profile["name"] == "SOMCHAI RATTANAKUL"
    assert profile["emails"] == ["somchai.rattanakul.test@example.com"]
    assert profile["phones"] == ["+66 81 555 2741"]
    assert "Industrial Maintenance Technician" in profile["job_titles"]


@pytest.mark.skipif(not tesseract_available(), reason="needs tesseract")
def test_the_page_a_sideways_photo_yields_is_the_page_it_shows(
    tmp_path: Path,
    models,
) -> None:
    """The control for the rotation: the same pixels with the tag stripped read
    back as nonsense, so the tag is doing the work rather than Tesseract."""
    from tests.helpers import run_pipeline

    upright = render_page(tmp_path / "upright.png", dpi=200)
    tagged = store_sideways(
        upright,
        tmp_path / "tagged.jpg",
        orientation=6,
        turn=2,  # Image.ROTATE_90
    )
    from PIL import Image

    with Image.open(tagged) as image:
        image.save(tmp_path / "untagged.jpg", quality=90)

    turned = run_pipeline(tagged, tmp_path / "workspace", models)
    assert turned["header_profile"]["emails"] == [
        "somchai.rattanakul.test@example.com"
    ]

    left_sideways = run_pipeline(
        tmp_path / "untagged.jpg",
        tmp_path / "workspace",
        models,
    )
    assert left_sideways["header_profile"]["emails"] == []
