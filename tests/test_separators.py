"""Guards on context-sensitive separator handling.

Each of these characters does more than one job, and the tests come in pairs
for that reason: the same character splitting in one context and not in the
other is the whole behaviour.
"""

from __future__ import annotations

import re

import pytest

from restruct.structure.separators import (
    at_sign_split,
    colon_is_key_value,
    dash_field_boundary,
    dash_is_range,
    repeated_label_rows,
)


# -- colon -------------------------------------------------------------------


@pytest.mark.parametrize(
    "label",
    ["Languages", "Technical Skills", "Driving licence", "Availability"],
)
def test_a_short_label_needs_no_corroboration(label: str) -> None:
    assert colon_is_key_value(label)


def test_a_time_is_not_a_labelled_field() -> None:
    """"09:30" matches a label/value pattern perfectly and means nothing of
    the sort. A label ends in a word, never in a bare number."""
    assert not colon_is_key_value("09")
    assert not colon_is_key_value("1")


def test_a_url_is_not_a_labelled_field() -> None:
    assert not colon_is_key_value("https")
    assert not colon_is_key_value("http")


def test_a_long_label_alone_is_a_sentence() -> None:
    sentence = (
        "Throughout the role I was responsible for the following areas of work"
    )
    assert not colon_is_key_value(sentence)


def test_a_long_label_is_accepted_when_the_rows_around_it_repeat_the_shape() -> None:
    """A labelled block is a layout the document is committing to, and that
    is what makes a longer label a label rather than prose."""
    label = (
        "Warehouse and distribution systems experience across the region"
    )
    neighbours = [
        "Supplier scorecard reporting and procurement analysis coverage",
        "Inventory reconciliation against ERP extracts and stock records",
    ]
    assert not colon_is_key_value(label)
    assert colon_is_key_value(
        label,
        repeated_label_rows=repeated_label_rows(neighbours, label),
    )


def test_a_label_does_not_corroborate_itself() -> None:
    label = "Warehouse and distribution systems experience across the region"
    assert repeated_label_rows([label, label], label) == 0


# -- dash --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [("2019", "2022"), ("Jan 2020", "Jun 2020"), ("May 2023", "Present")],
)
def test_a_dash_between_dates_is_a_range(left: str, right: str) -> None:
    """Splitting there yields two half-dates and loses the only thing the row
    was saying."""
    assert dash_is_range(left, right)
    assert dash_field_boundary(f"{left} - {right}") is None


def test_a_dash_between_text_is_a_field_boundary() -> None:
    split = dash_field_boundary("Senior Analyst - Logistics")
    assert split is not None
    assert (split.left, split.right) == ("Senior Analyst", "Logistics")


def test_a_phrase_mentioning_a_year_is_not_a_date() -> None:
    """"Analyst since 2019" contains a date; it is not one. The distinction is
    what stops a real field boundary being read as a range."""
    assert not dash_is_range("Analyst since 2019", "2022")


# -- at sign -----------------------------------------------------------------


def test_an_at_sign_separates_a_role_from_an_employer() -> None:
    split = at_sign_split("Senior Engineer @ Acme Corp")
    assert split is not None
    assert (split.left, split.right) == ("Senior Engineer", "Acme Corp")


def test_an_email_address_is_never_split_at_its_at_sign() -> None:
    """Addresses are claimed deterministically long before anything asks this,
    but the pattern must not depend on that having happened."""
    assert at_sign_split("priya.nair@example.com") is None


# -- offsets -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["Senior Engineer @ Acme Corp", "Data Analyst  @  Nusantara Retail"],
)
def test_every_split_stays_reversible(text: str) -> None:
    """A later parser must be able to see, and undo, a split this module made."""
    split = at_sign_split(text)
    assert split is not None
    assert split.original == text
    assert text[split.left_start : split.left_end] == split.left
    assert text[split.right_start : split.right_end] == split.right


def test_blanking_a_date_span_preserves_every_other_offset() -> None:
    """Experience blanks its claimed dates before looking for a separator, so
    a split found afterwards still points into the source line."""
    from restruct.parsers.experience import _text_outside
    from restruct.patterns.dates import DATE_RANGE_RE

    text = "Senior Engineer @ Acme Corp  2019 - 2022"
    blanked = _text_outside(text, list(DATE_RANGE_RE.finditer(text)))
    assert len(blanked) == len(text)
    assert blanked.index("Senior") == text.index("Senior")
    assert "2019" not in blanked
    split = at_sign_split(blanked)
    assert split is not None
    assert text[split.right_start : split.right_end] == "Acme Corp"
