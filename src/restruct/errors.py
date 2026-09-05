"""What can go wrong, named.

The engine raises these; it never exits, never prints, and never chooses a
status code. ``cli.py`` is the only module that maps them to exit codes, which
is what keeps the Python API usable as a library -- a caller embedding restruct
gets an exception it can catch by type, not a process that has already died.

Each class is a distinct failure a caller might reasonably handle differently.
"Something went wrong" is not one of them, which is why ``ExtractionFailed``
carries the original exception rather than replacing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


class RestructError(Exception):
    """Base for every failure this package raises deliberately."""


class InputNotFound(RestructError):
    """The path given does not exist."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"no such file: {path}")
        self.path = path


class UnsupportedFormat(RestructError):
    """The file exists but this version cannot read that format."""

    def __init__(self, path: Path, supported: tuple[str, ...]) -> None:
        super().__init__(
            f"unsupported format '{path.suffix or path.name}': "
            f"restruct v1 reads {', '.join(supported)}"
        )
        self.path = path
        self.supported = supported


class InvalidDocument(RestructError):
    """The file is the right format but cannot be opened or read.

    Covers a corrupt file and a password-protected one alike: from the
    caller's side both mean the same thing, and saying which is a detail of
    the message rather than of the type.
    """

    def __init__(self, path: Path, detail: str) -> None:
        super().__init__(f"cannot read {path}: {detail}")
        self.path = path
        self.detail = detail


class ModelAssetsMissing(RestructError):
    """Local model weights are absent.

    An extraction never downloads them itself, so this is a setup problem with
    a specific fix rather than something to retry. The fix is one command --
    ``restruct --install-models`` -- and the message names it, because the
    reader's next question is always what to do about it.
    """

    def __init__(
        self,
        directory: Path,
        searched: Sequence[Path] = (),
    ) -> None:
        # Naming every place that was looked in, because the reader's next
        # question is always "where does it want them?" -- and for an
        # installed copy, unlike a checkout, the answer is not obvious.
        locations = "".join(f"\n  {candidate}" for candidate in searched)
        super().__init__(
            f"model weights are missing: {directory}. "
            "Run 'restruct --install-models' to download them, or see the "
            "README for the expected layout."
            + (f" Looked in:{locations}" if locations else "")
        )
        self.directory = directory
        self.searched = tuple(searched)


class ModelDownloadFailed(RestructError):
    """An explicit model install reached the network and did not finish.

    Distinct from ``ModelAssetsMissing`` because the two have different
    remedies and only one of them is worth retrying: weights that were never
    fetched are a setup step not yet taken, while a download that broke part
    way through is a connection, a disk, or a file that did not arrive as the
    bytes it was pinned to.
    """

    def __init__(self, url: str, detail: str) -> None:
        super().__init__(f"could not download {url}: {detail}")
        self.url = url
        self.detail = detail


class TesseractMissing(RestructError):
    """OCR is needed for this document and the binary is not installed.

    Raised only when a page actually has too little text to parse. A native
    PDF must never require Tesseract, so this cannot be checked up front.
    """

    def __init__(self, command: str) -> None:
        super().__init__(
            f"this document needs OCR and '{command}' is not installed. "
            "Install it with: brew install tesseract (macOS), "
            "apt-get install tesseract-ocr (Debian/Ubuntu), "
            "or the UB-Mannheim installer (Windows)."
        )
        self.command = command


class OcrFailed(RestructError):
    """Tesseract ran and returned an error."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"OCR failed: {detail}")
        self.detail = detail


class ExtractionFailed(RestructError):
    """The document opened but parsing did not finish.

    Wraps the original exception rather than replacing it, because this is the
    one failure with no specific remedy and the traceback is the only useful
    thing left to hand back.
    """

    def __init__(self, path: Path, cause: BaseException) -> None:
        super().__init__(f"could not extract {path}: {cause}")
        self.path = path
        self.cause = cause


class OutputWriteFailed(RestructError):
    """Extraction succeeded and the result could not be written."""

    def __init__(self, path: Path, detail: str) -> None:
        super().__init__(f"cannot write {path}: {detail}")
        self.path = path
        self.detail = detail
