"""Deterministic company evidence, used to ground NER organization guesses."""

from __future__ import annotations

import re

# A legal-form suffix is strong evidence a segment names an employer rather than
# a job title, independent of what NER thinks, and is the strongest marker, but plenty of employers carry none.
# The second group is the collective nouns a company name ends in and a job
# title does not: nobody is hired as a "Group" or a "Holdings". They are
# anchored to the end of the segment for exactly that reason -- "Group Product
# Manager" is a title, "Nara Commerce Group" is an employer, and only the
# position of the word tells them apart.
COMPANY_MARKER_RE = re.compile(
    r"\b(?:co\.?|company|ltd\.?|limited|inc\.?|corp\.?|corporation|llc|plc)\b"
    r"|\b(?:group|holdings?|partners|ventures|industries|enterprises|"
    r"solutions|technologies|systems|services|laboratories|labs)\s*$",
    re.IGNORECASE,
)
