"""Guards on the extraction precedence.

The precedence used to live in comments, and it had already drifted: the MiniLM
attribute stage ran before both the contact regexes and NER, which no test
could see. These pin the ordering as an invariant instead.
"""

from __future__ import annotations

import re

import pytest

from restruct.document.types import HeaderEntityMatch
from restruct.structure.resolver import PrecedenceError, SpanResolver, Tier


def match(kind: str, start: int, end: int, line_index: int = 0) -> HeaderEntityMatch:
    return HeaderEntityMatch(
        kind=kind,
        text="x" * (end - start),
        line_index=line_index,
        start=start,
        end=end,
        detection_method="test",
    )


def test_tiers_are_ordered_strongest_first() -> None:
    assert Tier.DETERMINISTIC < Tier.CONTEXT < Tier.NER
    assert Tier.NER < Tier.SEMANTIC < Tier.GEOMETRY < Tier.UNRESOLVED


def test_a_stage_running_out_of_order_raises() -> None:
    """The whole point. A drifted stage is a programming error caught here,
    not a subtle output change found three commits later."""
    resolver = SpanResolver()
    resolver.open(Tier.NER)
    with pytest.raises(PrecedenceError, match="deterministic ran after ner"):
        resolver.open(Tier.DETERMINISTIC)


def test_one_tier_may_open_several_times() -> None:
    """The header claims labels, emails, phones and link annotations all
    deterministically; those are one tier with several stages."""
    resolver = SpanResolver()
    resolver.open(Tier.DETERMINISTIC)
    resolver.open(Tier.DETERMINISTIC)
    assert resolver.current_tier is Tier.DETERMINISTIC


def test_claiming_before_any_tier_opens_raises() -> None:
    with pytest.raises(PrecedenceError):
        SpanResolver().claim(match("name", 0, 4))


def test_an_earlier_claim_blocks_a_later_overlapping_one() -> None:
    """The mechanism the precedence rests on: a weaker stage cannot take
    characters a stronger one already read."""
    resolver = SpanResolver()
    resolver.open(Tier.DETERMINISTIC)
    assert resolver.claim(match("email", 10, 30))
    resolver.open(Tier.NER)
    assert not resolver.claim(match("name", 20, 40))
    assert resolver.claim(match("name", 30, 40))
    assert [item.kind for item in resolver.matches] == ["email", "name"]


def test_claims_on_different_lines_do_not_collide() -> None:
    resolver = SpanResolver()
    resolver.open(Tier.DETERMINISTIC)
    assert resolver.claim(match("email", 0, 10, line_index=0))
    assert resolver.claim(match("email", 0, 10, line_index=1))


def test_claim_pattern_skips_what_is_already_claimed() -> None:
    resolver = SpanResolver()
    resolver.open(Tier.DETERMINISTIC)
    resolver.claim(match("phone", 0, 5))
    taken = resolver.claim_pattern(
        re.compile(r"\d+"),
        line_index=0,
        text="12345 6789",
        kind="number",
    )
    assert [item.text for item in taken] == ["6789"]


def test_has_kind_asks_about_fields_not_characters() -> None:
    """A fallback stage runs when a field is still missing, which is a
    different question from whether some characters are spoken for."""
    resolver = SpanResolver()
    resolver.open(Tier.DETERMINISTIC)
    resolver.claim(match("email", 0, 10))
    assert resolver.has_kind("email")
    assert not resolver.has_kind("name")


def test_the_tier_that_claimed_a_match_is_recorded() -> None:
    resolver = SpanResolver()
    resolver.open(Tier.DETERMINISTIC)
    deterministic = match("email", 0, 10)
    resolver.claim(deterministic)
    resolver.open(Tier.GEOMETRY)
    guessed = match("name", 20, 30)
    resolver.claim(guessed)
    assert resolver.tier_of(deterministic) is Tier.DETERMINISTIC
    assert resolver.tier_of(guessed) is Tier.GEOMETRY
