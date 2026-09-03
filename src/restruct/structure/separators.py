"""What a separator means, given what surrounds it.

The same character does different jobs in different places, and a pattern alone
cannot tell them apart:

    Languages: Python, SQL          a colon separating a label from its value
    Available: 09:30 - 17:00        a colon inside a time, twice
    2019 - 2022                     a dash joining two dates into one range
    Senior Analyst - Logistics      a dash separating two fields
    Senior Engineer @ Acme Corp     an at-sign separating a title from a company

So each question here takes the surroundings as an argument. Nothing decides
what a label *means* -- that stays with the extraction stage. These only answer
whether a split is there at all.

Every answer carries the character offsets of the parts, and callers keep the
original unsplit text alongside, so a later stage can always see and reverse a
split this module got wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from restruct.patterns.dates import SINGLE_DATE_RE

# A label this short is a label whatever else is true of it.
_ALWAYS_SHORT_LABEL_WORDS = 7
_ALWAYS_SHORT_LABEL_CHARACTERS = 50

# A longer label is still a label when the rows around it are labelled the same
# way, because a labelled block is a layout the document is committing to.
_REPEATED_LABEL_CHARACTERS = 80
_REPEATED_LABEL_ROWS = 2

# Both sides numeric and unspaced is a time or a ratio, never a field label:
# "09:30", "1:4". A real label ends in a word.
_NUMERIC_COLON_RE = re.compile(r"^\s*\d{1,2}\s*$")

# An at-sign in a metadata line reads as "at": a role at an employer. Requires
# space around it so an email address is never split here -- an address is
# claimed deterministically long before anything asks this question.
AT_SEPARATOR_RE = re.compile(r"\s+@\s+")

# A dash with space either side. Whether it separates or joins is decided by
# what sits on the two sides, not by the character.
SPACED_DASH_RE = re.compile(r"\s+[-–—]\s+")


@dataclass(frozen=True)
class Split:
    """One separator split, with the original always recoverable."""

    original: str
    left: str
    right: str
    left_start: int
    left_end: int
    right_start: int
    right_end: int
    method: str


def _is_date_like(text: str) -> bool:
    """Whether a fragment reads as a date on its own.

    Used to tell a range from a boundary, so it has to be the whole fragment:
    "2019" is a date, "Analyst since 2019" is a phrase that mentions one.
    """
    stripped = text.strip()
    return bool(stripped) and SINGLE_DATE_RE.fullmatch(stripped) is not None


def colon_is_key_value(label: str, *, repeated_label_rows: int = 0) -> bool:
    """Whether a colon separates a label from a value.

    ``repeated_label_rows`` is how many nearby rows are labelled the same way.
    A short label needs no corroboration; a longer one is accepted only when
    the surrounding rows show the document is using a labelled layout, which is
    the difference between a field and a sentence that happens to contain a
    colon.
    """
    stripped = label.strip()
    if not stripped or _NUMERIC_COLON_RE.match(stripped):
        return False
    if stripped.casefold().startswith(("http", "www.")):
        return False
    if (
        len(stripped.split()) <= _ALWAYS_SHORT_LABEL_WORDS
        and len(stripped) <= _ALWAYS_SHORT_LABEL_CHARACTERS
    ):
        return True
    return (
        repeated_label_rows >= _REPEATED_LABEL_ROWS
        and len(stripped) <= _REPEATED_LABEL_CHARACTERS
    )


def dash_is_range(left: str, right: str) -> bool:
    """Whether a dash joins two dates rather than separating two fields.

    "2019 - 2022" is one value and splitting it produces two half-dates;
    "Senior Analyst - Logistics" is two values and not splitting it produces
    one field that is neither.
    """
    return _is_date_like(left) and _is_date_like(right)


def _split_on(text: str, pattern: re.Pattern[str], method: str) -> Split | None:
    match = pattern.search(text)
    if match is None:
        return None
    raw_left, raw_right = text[: match.start()], text[match.end() :]
    left, right = raw_left.strip(), raw_right.strip()
    if not left or not right:
        return None
    left_start = raw_left.find(left)
    right_start = match.end() + raw_right.find(right)
    return Split(
        original=text,
        left=left,
        right=right,
        left_start=left_start,
        left_end=left_start + len(left),
        right_start=right_start,
        right_end=right_start + len(right),
        method=method,
    )


def dash_field_boundary(text: str) -> Split | None:
    """Split on a dash only where it separates fields rather than dates."""
    split = _split_on(text, SPACED_DASH_RE, "delimiter_dash_boundary")
    if split is None or dash_is_range(split.left, split.right):
        return None
    return split


def at_sign_split(text: str) -> Split | None:
    """Split "Senior Engineer @ Acme Corp" into its two halves.

    Left is the role and right the employer, which is the only way round the
    construction is ever written.
    """
    return _split_on(text, AT_SEPARATOR_RE, "delimiter_at")


def repeated_label_rows(labels: list[str], label: str) -> int:
    """How many of ``labels`` share a shape with ``label``.

    Shape, not spelling: a labelled block repeats the *form* -- a short
    capitalised phrase before a colon -- while every label in it differs.
    """
    target_words = len(label.split())
    return sum(
        1
        for other in labels
        if other.strip()
        and other.strip() != label.strip()
        and abs(len(other.split()) - target_words) <= 2
    )
