"""Patterns describing page furniture rather than resume content."""

from __future__ import annotations

import re

# Matched only against lines already near the bottom of a page, so an ordinary
# sentence ending in a number is not mistaken for a running footer.
PAGE_FOOTER_RE = re.compile(r"\bpage\s+\d+\s*$", re.IGNORECASE)


# A section heading numbered by its author: "01 EXPERIENCE", "2. Education",
# "IV — Skills". The number orders the sections and says nothing about which
# destination the heading names, so it is removed before the heading is
# classified. Without this the whole prefix goes into the embedding and a
# heading that reads perfectly to a person resolves to nothing.
#
# One or two digits only: a four-digit run is a year, and "2024 Highlights" is
# a heading about a year rather than the twenty-fourth section.
HEADING_ORDINAL_RE = re.compile(
    r"^\s*(?:\d{1,2}|[IVXivx]{1,4})\s*[.)\]:–—-]?\s+(?=\S)",
)


def heading_text(text: str) -> str:
    """A heading with its ordinal prefix and invisible characters removed."""
    cleaned = text.replace("​", "").replace("﻿", "").strip()
    stripped = HEADING_ORDINAL_RE.sub("", cleaned)
    # Never strip everything: a heading that is only a number is not one, and
    # returning an empty string would make it match nothing at all.
    return stripped or cleaned
