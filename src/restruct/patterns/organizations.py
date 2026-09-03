"""Deterministic company evidence, used to ground NER organization guesses."""

from __future__ import annotations

import re

# A legal-form suffix is strong evidence a segment names an employer rather than
# a job title, independent of what NER thinks.
COMPANY_MARKER_RE = re.compile(
    r"\b(?:co\.?|company|ltd\.?|limited|inc\.?|corp\.?|corporation|llc|plc)\b",
    re.IGNORECASE,
)
