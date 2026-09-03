"""Separator patterns.

A separator is evidence, not an instruction: the same colon or dash means
different things in a skills row and in an experience metadata line. These
patterns only locate candidates — the section parsers decide what a split means,
and always keep the original unsplit text alongside it.
"""

from __future__ import annotations

import re

# Splits a header line into its visually delimited segments.
SEGMENT_RE = re.compile(r"[^|•·]+")

JOB_TITLE_SEPARATOR_RE = re.compile(r"[|•·]|\s+/\s+")

# Experience and education metadata split identically; they were two byte-equal
# copies before this module existed.
METADATA_SEPARATOR_RE = re.compile(r"\s*(?:[|•·]|–|—|\s-\s)\s*")

# Key-value delimiters, tried in this order. The label side is length-capped so
# a full sentence containing a colon is not read as a key-value pair.
KEY_VALUE_COLON_RE = re.compile(r"^(?P<label>[^:\t]{1,60}):\s*(?P<body>.+)$")
KEY_VALUE_TAB_RE = re.compile(r"^(?P<label>[^\t]{1,60})\t+\s*(?P<body>.+)$")
KEY_VALUE_DASH_RE = re.compile(
    r"^(?P<label>.{1,60}?)\s+(?:-|–|—)\s+(?P<body>.+)$"
)
