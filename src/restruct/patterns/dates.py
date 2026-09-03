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

# One endpoint of a range rather than a whole range: "2019", "Jan 2020",
# "Present". Used to ask whether a dash joins two dates or separates two
# fields, which needs each side tested on its own.
SINGLE_DATE_RE = re.compile(
    rf"(?:{DATED_YEAR_PATTERN}|Present|Current|Now)",
    re.IGNORECASE,
)


def date_matches(text: str) -> list[re.Match[str]]:
    """Every date in a line: ranges, plus bare years no range already covers.

    A range is claimed first and a year inside one is not claimed again, which
    is the same precedence the rest of the package uses -- the stronger, more
    specific reading takes the characters and the weaker one skips them.

    Shared because the grouped sections and education were answering the same
    question differently: education matched only ranges, so a graduation year
    written on its own -- which is how most resumes write one -- was not a date
    at all.
    """
    ranges = list(DATE_RANGE_RE.finditer(text))
    singles = [
        match
        for match in SINGLE_YEAR_RE.finditer(text)
        if not any(
            date_range.start() < match.end() and match.start() < date_range.end()
            for date_range in ranges
        )
    ]
    return sorted([*ranges, *singles], key=lambda match: match.start())
