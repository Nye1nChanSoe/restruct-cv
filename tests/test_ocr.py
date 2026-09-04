"""Guards on when OCR is required and how a missing engine is reported.

Tesseract is an optional system dependency, and the risk is that it quietly
stops being optional: a check in the wrong place makes every native PDF and
every DOCX refuse to run on a machine that will never render a page. So these
tests are mostly about what must *not* happen -- no detection up front, no
render before the check, no PATH lookup standing in for the real question.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from restruct.errors import TesseractMissing
from restruct.ingestion import ocr as ocr_module
from tests.helpers import SYNTHETIC_DIRECTORY


@pytest.fixture
def no_tesseract_anywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine with no OCR engine on PATH and none where installers put it."""
    monkeypatch.setattr(ocr_module.shutil, "which", lambda command: None)
    monkeypatch.setattr(ocr_module, "_known_install_locations", tuple)


def test_the_binary_is_found_on_the_path_when_it_is_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ocr_module.shutil, "which", lambda command: f"/somewhere/{command}"
    )
    assert ocr_module.find_tesseract() == "/somewhere/tesseract"


def test_a_binary_off_the_path_is_still_found_where_installers_put_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The Windows installer named in the error message installs into Program
    Files and leaves PATH alone, so PATH alone would report it absent."""
    installed = tmp_path / "tesseract"
    installed.write_text("#!/bin/sh\n")
    installed.chmod(0o755)
    monkeypatch.setattr(ocr_module.shutil, "which", lambda command: None)
    monkeypatch.setattr(ocr_module, "_known_install_locations", lambda: (str(installed),))
    assert ocr_module.find_tesseract() == str(installed)


def test_nothing_is_found_when_nothing_is_installed(no_tesseract_anywhere) -> None:
    assert ocr_module.find_tesseract() is None


def test_a_native_pdf_never_asks_for_an_ocr_engine(no_tesseract_anywhere) -> None:
    """The whole reason detection is not done up front. A document with its own
    text layer must read on a machine that has no OCR engine at all."""
    import pymupdf

    from restruct.ingestion.native import read_document

    with pymupdf.open(SYNTHETIC_DIRECTORY / "1.pdf") as document:
        physical = read_document(document)
    assert physical.lines and not physical.used_ocr


def test_a_docx_never_asks_for_an_ocr_engine(no_tesseract_anywhere) -> None:
    from restruct.ingestion.docx import read_docx

    assert read_docx(SYNTHETIC_DIRECTORY / "11.docx").lines


def test_a_missing_engine_is_reported_before_the_page_is_rendered(
    no_tesseract_anywhere, tmp_path: Path
) -> None:
    """Rasterising several hundred milliseconds of page nothing can read is
    work done purely to arrive at a failure already knowable."""

    class RefusesToRender:
        def get_pixmap(self, **keywords: object) -> object:
            raise AssertionError("rendered a page for an OCR engine that is absent")

    with pytest.raises(TesseractMissing):
        ocr_module.ocr_page(RefusesToRender(), 0, tmp_path)


def test_the_failure_names_a_command_for_each_platform() -> None:
    """A message that only says "not installed" leaves the reader to search for
    the package name, which differs on all three platforms."""
    message = str(TesseractMissing("tesseract"))
    assert "brew install tesseract" in message
    assert "apt-get install tesseract-ocr" in message
    assert "Windows" in message
