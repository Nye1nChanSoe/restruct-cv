"""Personal-attribute patterns for the header profile.

The label establishes the field, and the value is preserved verbatim, so a date
of birth is never forced into one date format and an unfamiliar spelling such as
'martial status' still resolves.
"""

from __future__ import annotations

import re

ATTRIBUTE_LABEL_PATTERN = (
    r"date\s+of\s+birth|birth\s+date|d\.?\s*o\.?\s*b\.?|dob|"
    r"current\s+age|age|"
    r"gender|sex|marital\s+status|martial\s+status|civil\s+status|marital|"
    r"visa\s+status|visa\s+type|work\s+visa|immigration\s+status|residency\s+visa|"
    r"work\s+authorization|right\s+to\s+work|visa|"
    r"nationality|citizenship|"
    r"current\s+residen(?:ce|t)|current\s+location|place\s+of\s+residence|"
    r"residen(?:ce|t)|"
    # Compensation. Longest alternatives first, or "current ctc" is consumed by
    # a bare "ctc" and the label recorded is the wrong half of the phrase.
    r"current\s+package|annual\s+package|total\s+package|"
    r"compensation\s+package|total\s+compensation|current\s+ctc|ctc|package|"
    r"current\s+salary|present\s+salary|current\s+income|current\s+pay|"
    r"monthly\s+salary|monthly\s+income|last\s+drawn\s+salary|salary|income"
)

# A known label followed by its value, stopping at the next delimiter or label.
LABELLED_ATTRIBUTE_RE = re.compile(
    rf"(?P<label>\b(?:{ATTRIBUTE_LABEL_PATTERN})\b)"
    rf"(?:\s*(?::|[-–—])\s*|\t+|\s+)"
    rf"(?P<value>.+?)"
    rf"(?=\s*(?:[|•·]|\b(?:{ATTRIBUTE_LABEL_PATTERN})\b)|$)",
    re.IGNORECASE,
)

# Any short label followed by a value. Whether it names a supported attribute is
# decided semantically, not by this pattern.
GENERIC_ATTRIBUTE_RE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z .'/]{1,40}?)"
    r"\s*(?::|[-–—])\s*"
    r"(?P<value>.+?)(?=\s*[|•·]|$)",
    re.IGNORECASE,
)

# The same shape anchored to a whole line, for labelled rows inside a section.
ATTRIBUTE_INLINE_RE = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z .'/]{1,40}?)"
    r"\s*(?::|[-–—])\s*(?P<body>.+)$"
)

# 'City, Country' shaped, with no digits or '@' to exclude addresses and emails.
LOCATION_SEGMENT_RE = re.compile(r"^[^\d@]{2,48},\s*[^\d@]{2,48}$", re.UNICODE)

NATIONALITY_CONTEXT_RE = re.compile(
    r"\b(?:citizen|citizenship|national|nationality)\b",
    re.IGNORECASE,
)

NATIONALITY_PHRASE_RE = re.compile(
    r"(?:"
    r"\b[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*)?\s+"
    r"(?:citizen|national)\b"
    r"|\b(?:citizenship|nationality)\s*:\s*"
    r"[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*)?\b"
    r")",
    re.IGNORECASE,
)
