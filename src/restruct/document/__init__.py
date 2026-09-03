"""Shared in-memory representation of one parsed document."""

from restruct.document.types import (
    DetectedHeading,
    ExtractedLine,
    HeaderEntityMatch,
    append_regex_matches,
    overlaps_existing,
)

__all__ = [
    "DetectedHeading",
    "ExtractedLine",
    "HeaderEntityMatch",
    "append_regex_matches",
    "overlaps_existing",
]
