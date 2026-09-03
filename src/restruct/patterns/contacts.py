"""Deterministic contact patterns: the first stage of extraction precedence."""

from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")

# Deliberately loose on internal separators so '+66 81 555 2741' and
# '(02) 555-2741' both match, while refusing to start or end mid-word.
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d ()\-.]{6,}\d(?!\w)")

URL_RE = re.compile(
    r"(?:"
    r"https?://[^\s|,;)]+"
    r"|www\.[^\s|,;)]+"
    r"|(?<![@\w.-])(?:linkedin|github)\.com/[^\s|,;)]+"
    r"|(?<![@\w.-])[a-z0-9](?:[a-z0-9-]{0,59}[a-z0-9])"
    r"(?:\.[a-z]{2,})+(?:/[^\s|,;)]*)?"
    r"(?![@\w-]|\.[a-z0-9])"
    r")",
    re.IGNORECASE,
)
