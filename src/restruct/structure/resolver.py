"""The one place the extraction precedence is written down.

Every parser resolves the same way: several stages compete to claim character
spans of a line, and the strongest evidence wins. Before this module that order
lived in prose. ``build_header_profile`` carried a comment explaining why
labelled attributes must precede the contact regexes, ``_experience_line_entities``
worked out its own order again from scratch, and nothing checked either -- the
MiniLM attribute stage had drifted to before both the regexes and NER, and no
test could see it.

The order, strongest first, and what each tier means:

    DETERMINISTIC   the document says so outright -- an exact label from a
                    known list, a URL annotation, a regex over a shape that
                    means only one thing
    CONTEXT         deterministic, but only valid given its surroundings: a
                    colon is a key-value separator here and a time here, a
                    dash is a range between dates and a boundary between words
    NER             a model naming a span it recognises
    SEMANTIC        MiniLM, ranking a span against reference phrasings
    GEOMETRY        position and typography alone, with nothing in the text
                    to go on
    UNRESOLVED      claimed by nobody, kept verbatim rather than guessed at

A tier is opened once and cannot be reopened. That is the whole point: a stage
running out of order is a programming error the resolver raises on, not a
subtle output change discovered three commits later.

Spans are never overwritten. A stage offers a claim and the resolver refuses it
if an earlier one already covers those characters, which is why every match
keeps ``start``/``end`` into the source line -- a later parser must always be
able to see, and reverse, a split.
"""

from __future__ import annotations

import re
from enum import IntEnum
from typing import Iterable

from restruct.document.types import HeaderEntityMatch


class Tier(IntEnum):
    """Extraction precedence, strongest first."""

    DETERMINISTIC = 1
    CONTEXT = 2
    NER = 3
    SEMANTIC = 4
    GEOMETRY = 5
    UNRESOLVED = 6

    @property
    def label(self) -> str:
        return self.name.lower()


class PrecedenceError(RuntimeError):
    """A stage ran out of precedence order."""


class SpanResolver:
    """Collects claims over one document's lines, in precedence order."""

    def __init__(self) -> None:
        self._matches: list[HeaderEntityMatch] = []
        self._tier_of: dict[int, Tier] = {}
        self._open_tier: Tier | None = None

    # -- ordering ---------------------------------------------------------

    def open(self, tier: Tier) -> None:
        """Begin a stage. Raises if precedence has already moved past it.

        Reopening the tier that is already current is allowed, because one tier
        legitimately has several stages -- the header claims labels, emails,
        phones and link annotations all deterministically.
        """
        if self._open_tier is not None and tier < self._open_tier:
            raise PrecedenceError(
                f"{tier.label} ran after {self._open_tier.label}; "
                "extraction precedence is strongest-first and each tier opens once"
            )
        self._open_tier = tier

    @property
    def current_tier(self) -> Tier | None:
        return self._open_tier

    # -- claiming ---------------------------------------------------------

    def is_claimed(self, *, line_index: int, start: int, end: int) -> bool:
        """Whether an earlier stage already covers any of these characters."""
        return any(
            match.line_index == line_index and match.start < end and start < match.end
            for match in self._matches
        )

    def claim(self, match: HeaderEntityMatch) -> bool:
        """Offer one match. Returns whether it was taken.

        Refused when an earlier stage already covers the span, which is the
        mechanism the whole precedence rests on.
        """
        if self._open_tier is None:
            raise PrecedenceError("claim() before any tier was opened")
        if self.is_claimed(
            line_index=match.line_index,
            start=match.start,
            end=match.end,
        ):
            return False
        self._matches.append(match)
        self._tier_of[id(match)] = self._open_tier
        return True

    def claim_all(self, matches: Iterable[HeaderEntityMatch]) -> list[HeaderEntityMatch]:
        """Offer several matches in order, returning those taken."""
        return [match for match in matches if self.claim(match)]

    def claim_pattern(
        self,
        pattern: re.Pattern[str],
        *,
        line_index: int,
        text: str,
        kind: str,
        detection_method: str = "regex",
    ) -> list[HeaderEntityMatch]:
        """Claim every match of ``pattern`` no earlier stage already covers."""
        return self.claim_all(
            HeaderEntityMatch(
                kind=kind,
                text=found.group(0),
                line_index=line_index,
                start=found.start(),
                end=found.end(),
                detection_method=detection_method,
            )
            for found in pattern.finditer(text)
        )

    # -- reading back -----------------------------------------------------

    @property
    def matches(self) -> list[HeaderEntityMatch]:
        """The claims so far, in the order they were taken."""
        return self._matches

    def tier_of(self, match: HeaderEntityMatch) -> Tier | None:
        """Which tier claimed this match."""
        return self._tier_of.get(id(match))

    def has_kind(self, kind: str) -> bool:
        """Whether anything has claimed this field yet.

        Distinct from ``is_claimed``: that asks about characters, this asks
        about fields. A stage that only runs when a field is still missing --
        the geometry name guess, the nationality fallback -- asks this one.
        """
        return any(match.kind == kind for match in self._matches)
