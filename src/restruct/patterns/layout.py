"""Patterns describing page furniture rather than resume content."""

from __future__ import annotations

import re

# Matched only against lines already near the bottom of a page, so an ordinary
# sentence ending in a number is not mistaken for a running footer.
PAGE_FOOTER_RE = re.compile(r"\bpage\s+\d+\s*$", re.IGNORECASE)
