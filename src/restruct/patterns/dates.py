"""Date and date-range patterns, including open-ended 'Present' ranges."""

from __future__ import annotations

import re

MONTH_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)"
)
DATED_YEAR_PATTERN = rf"(?:{MONTH_PATTERN}(?:\s+|-))?(?:19|20)\d{{2}}"

# Year-to-year ranges are matched separately from month-year ranges so a bare
# hyphen between two years still reads as a range without loosening the rule
# for arbitrary text spans.
DATE_RANGE_RE = re.compile(
    rf"\b(?:"
    rf"(?:From\s+)?(?:19|20)\d{{2}}\s*-\s*(?:(?:19|20)\d{{2}}|Present|Current|Now)"
    rf"|(?:From\s+)?{DATED_YEAR_PATTERN}"
    rf"(?:\s+-\s+|\s*[–—]\s*|\s+to\s+)"
    rf"(?:{DATED_YEAR_PATTERN}|Present|Current|Now)"
    rf")\b",
    re.IGNORECASE,
)

SINGLE_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
