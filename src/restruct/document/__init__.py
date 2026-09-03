"""Shared in-memory representation of one parsed document."""

from restruct.document.types import (
    DetectedHeading,
    ExtractedLine,
    HeaderEntityMatch,
    overlaps_existing,
)

__all__ = [
    "DetectedHeading",
    "ExtractedLine",
    "HeaderEntityMatch",
    "overlaps_existing",
]
