"""Image ingestion: a PNG or JPEG presented as the page it photographs.

A photograph or screenshot of a resume is a scanned page that never reached a
PDF. It has exactly the same shape as a scanned page -- no native text, no
styles, nothing but pixels -- so the honest thing is to give it a page and let
the OCR path that already exists read it, rather than to add a third ingestion
track that would need its own copy of every rule.

MuPDF will open an image directly, and that is the tempting one-liner. It is
also wrong in a way that is easy to miss: the page it gives back has a box
measured in the image's *pixels*, because an image carries no statement of what
its pixels measure and most carry none at all. Every rule downstream that reads
a box as points is then reading a number that does not mean what it says --
rendering a metadata-free phone photo "at 300 dpi" upscales it fourfold into an
85-megapixel raster of pixels that were never there, and the debug canvas
multiplies the same wrong number again.

So the image is placed on a page of a stated size instead, keeping its aspect
ratio exactly. Nothing is invented: the pixels and their proportions are the
document's own, and only the units they are read in are chosen. What that size
was measured against is in ``SETTINGS.image``.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from restruct.configs import SETTINGS
from restruct.errors import InvalidDocument

# The formats this reader accepts. Deliberately the two that resumes actually
# arrive as, rather than everything MuPDF can decode: each one costs a fixture
# and a test to claim honestly.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

# EXIF tag 274, and the orientations that are a plain rotation. A camera writes
# the sensor's pixels and a tag saying which way up it was held, so a photo
# taken in portrait is stored in landscape and every viewer turns it. Nothing
# in MuPDF reads that tag, and OCR of a sideways page returns nothing.
#
# The four mirrored orientations (2, 4, 5, 7) are left alone: they come from a
# front camera or an editing accident, they are rare, and turning a page the
# wrong way round is worse than leaving one alone.
_EXIF_ORIENTATION_TAG = 274
_EXIF_ROTATIONS = {3: 180, 6: 270, 8: 90}


def is_image(path: Path) -> bool:
    return path.suffix.casefold() in IMAGE_SUFFIXES


def _image_shape(path: Path) -> tuple[int, int, int]:
    """The image's pixel size and the rotation needed to display it.

    Read from the header rather than by decoding: a 108-megapixel photograph
    costs 300 MB to decode and this needs two integers and a tag.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            width, height = image.size
            orientation = image.getexif().get(_EXIF_ORIENTATION_TAG, 1)
    except UnidentifiedImageError as error:
        raise InvalidDocument(path, "not an image this version can read") from error
    except OSError as error:
        raise InvalidDocument(path, str(error)) from error

    rotation = _EXIF_ROTATIONS.get(int(orientation or 1), 0)
    if width <= 0 or height <= 0:
        raise InvalidDocument(path, "the image has no pixels")
    return width, height, rotation


def read_image(path: Path) -> pymupdf.Document:
    """Open one image as a single-page document at a stated page size.

    The caller closes it, as it would any other MuPDF document; this one is
    built in memory rather than read from disk, which nothing downstream can
    tell or needs to.
    """
    width, height, rotation = _image_shape(path)
    if rotation in (90, 270):
        # A photo held sideways is stored sideways: the page is the shape the
        # image will be *after* it is turned, not the shape it was stored in.
        width, height = height, width

    scale = SETTINGS.image.page_long_side_points / max(width, height)
    document = pymupdf.open()
    try:
        page = document.new_page(width=width * scale, height=height * scale)
        page.insert_image(page.rect, filename=str(path), rotate=rotation)
    except Exception as error:  # MuPDF raises several unrelated types
        document.close()
        raise InvalidDocument(path, str(error)) from error
    return document
