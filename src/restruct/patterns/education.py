"""Deterministic education evidence: degrees, institutions, grades, coursework."""

from __future__ import annotations

import re

DEGREE_RE = re.compile(
    r"\b(?:associate(?:'s)?|bachelor(?:'s)?|master(?:'s)?|doctor(?:ate|al)?|"
    r"ph\.?\s*d\.?|m\.?\s*(?:sc|s|a|eng|ba)\.?|b\.?\s*(?:sc|s|a|eng|ba)\.?|"
    r"a\.?\s*(?:a|s)\.?|mba|degree|diploma|certificate|certification|"
    r"undergraduate|graduate|postgraduate|vocational)\b",
    re.IGNORECASE,
)

INSTITUTION_RE = re.compile(
    r"\b(?:university|college|institute|institution|school|academy|polytechnic|"
    r"conservatory|seminary|faculty)\b",
    re.IGNORECASE,
)

GPA_RE = re.compile(
    r"\b(?P<label>c?gpa|grade(?:\s+point\s+average)?)\s*:?[\s-]*"
    r"(?P<value>\d+(?:\.\d+)?)"
    r"(?:\s*(?:/|out\s+of)\s*(?P<scale>\d+(?:\.\d+)?))?\b",
    re.IGNORECASE,
)

# Marks prose that belongs to an education record rather than starting a new one.
COURSEWORK_RE = re.compile(
    r"^\s*(?:relevant\s+)?(?:coursework|courses?|subjects?|modules?)\b",
    re.IGNORECASE,
)
