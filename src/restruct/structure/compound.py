"""Compound headings, and who owns the blocks beneath one.

``CERTIFICATIONS & LANGUAGES`` names two destinations. Embedded whole it
resolves to at most one of them, and everything under it is filed there --
which is how resume 6's languages line ended up inside certifications.

Splitting the heading is the easy half. The hard half is that a component
naming a destination does not yet own anything: content has to be assigned to
it on evidence. This module keeps the two questions apart.

    split_heading()      what destinations does this heading name?
    logical_sections()   which blocks, if any, does each of them own?

Assignment runs deterministically and stops early. A block is claimed by an
explicit local label, by a local subheading, or by deterministic evidence for a
specific destination. Anything left over is uncertain, and uncertain content is
never assigned -- it is preserved under ``others`` with the original compound
heading, exactly as it was written.

Components are classified by exact match against the section reference lists
and by nothing else. A near match is not evidence: "TRAINING" leans towards
certifications and equally towards education, and the whole point of the
conservative fallback is that a lean is not enough to move content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from restruct.configs import SETTINGS
from restruct.document.types import DetectedHeading, ExtractedLine
from restruct.structure.headings import (
    _looks_like_subheading,
    _routed_section_headings,
    _section_body_style,
)
from restruct.patterns.bullets import BULLET_RE
from restruct.patterns.languages import LANGUAGE_NAME_RE
from restruct.patterns.separators import (
    COMPOUND_HEADING_SEPARATOR_RE,
    KEY_VALUE_COLON_RE,
)

# A component shorter than this is punctuation or an initial, not a section
# name; splitting on it would only manufacture noise.
_MINIMUM_COMPONENT_CHARACTERS = 2


@dataclass(frozen=True)
class HeadingComponent:
    """One destination-naming part of a heading.

    ``start``/``end`` index into the original heading text, so a later stage
    can always recover what was split and reverse the decision -- the same
    contract ``HeaderEntityMatch`` carries for header lines.
    """

    text: str
    start: int
    end: int
    section_type: str | None

    @property
    def resolved(self) -> bool:
        return self.section_type is not None


@dataclass(frozen=True)
class LogicalSection:
    """A destination and the line indexes assigned to it.

    One physical heading yields one of these when it names a single
    destination, and several when it names more and the content divides. The
    line indexes are explicit rather than a range, because the blocks of a
    split section interleave.
    """

    section_type: str
    heading: DetectedHeading
    line_indexes: tuple[int, ...]
    # The heading as written, kept whenever a split sent content to ``others``
    # so the output can still show which heading the content came from.
    compound_heading_text: str | None = None


def _exact_section_type(text: str) -> str | None:
    """The destination whose reference list contains this text verbatim."""
    folded = " ".join(text.split()).casefold()
    for section_type, references in SETTINGS.section_references.items():
        if any(folded == reference.casefold().strip() for reference in references):
            return section_type
    return None


def split_heading(text: str) -> tuple[HeadingComponent, ...]:
    """The destinations a heading names, in the order it names them.

    Returns a single component for an ordinary heading, so callers can treat
    every heading the same way and ask ``len(components) > 1`` when they care.

    A heading that is itself a known section name is never split. Several
    reference names contain a separator -- "Education and Training", "Awards
    and Honors", "Licenses and Certifications" -- and cutting those in half
    would turn a heading the document got exactly right into two guesses.
    """
    whole = _exact_section_type(text)
    if whole is not None:
        return (HeadingComponent(text=text, start=0, end=len(text), section_type=whole),)

    components: list[HeadingComponent] = []
    cursor = 0
    for piece in COMPOUND_HEADING_SEPARATOR_RE.split(text):
        start = text.find(piece, cursor) if piece else cursor
        if start < 0:
            start = cursor
        cursor = start + len(piece)
        stripped = piece.strip()
        if len(stripped) < _MINIMUM_COMPONENT_CHARACTERS or not any(
            character.isalpha() for character in stripped
        ):
            continue
        components.append(
            HeadingComponent(
                text=stripped,
                start=start,
                end=cursor,
                section_type=_exact_section_type(stripped),
            )
        )

    if not components:
        return (HeadingComponent(text=text, start=0, end=len(text), section_type=None),)
    return tuple(components)


# -- deterministic evidence that a block belongs to a specific destination ----


def _without_marker(text: str) -> str:
    """A block's text with any bullet marker removed.

    The marker says the line is an item, never which section it belongs to, and
    leaving it on would hide the key-value pair behind it: ``- Thai: Native``
    reads as the label ``- Thai``, which names no language.
    """
    stripped = text.strip()
    match = BULLET_RE.match(stripped)
    return stripped[match.end():].strip() if match else stripped


def _claims_languages(text: str) -> bool:
    """A ``Thai: Native`` style line, whose label names a language.

    The label side is what has to be a language: "English" as the key of a
    key-value pair is a language entry, while "English" inside a sentence about
    technical emails is not.
    """
    match = KEY_VALUE_COLON_RE.match(_without_marker(text))
    if match is None:
        return False
    return LANGUAGE_NAME_RE.fullmatch(match.group("label").strip()) is not None


# One entry per destination that has deterministic evidence of its own. Most
# have none, and a destination with no entry can still be claimed by an
# explicit label or subheading -- it simply cannot claim content by content.
_DETERMINISTIC_CLAIMS: dict[str, Callable[[str], bool]] = {
    "languages": _claims_languages,
}


def _explicit_label_owner(
    text: str,
    components: tuple[HeadingComponent, ...],
) -> str | None:
    """The component named outright at the start of a block.

    ``Certifications: AWS ...`` under ``CERTIFICATIONS & LANGUAGES`` says which
    half of the heading it belongs to. This is the strongest evidence there is,
    because the document states it rather than implying it.
    """
    match = KEY_VALUE_COLON_RE.match(_without_marker(text))
    if match is None:
        return None
    label = " ".join(match.group("label").split()).casefold()
    for component in components:
        if not component.resolved:
            continue
        if label == component.text.casefold() or _exact_section_type(label) == component.section_type:
            return component.section_type
    return None


def _block_owner(
    text: str,
    components: tuple[HeadingComponent, ...],
) -> str | None:
    """Which destination this block's own content assigns it to, if any."""
    owner = _explicit_label_owner(text, components)
    if owner is not None:
        return owner
    for component in components:
        if not component.resolved:
            continue
        claims = _DETERMINISTIC_CLAIMS.get(component.section_type or "")
        if claims is not None and claims(text):
            return component.section_type
    return None


# -- assignment --------------------------------------------------------------


def _assign_line_indexes(
    lines: list[ExtractedLine],
    line_indexes: list[int],
    components: tuple[HeadingComponent, ...],
) -> tuple[dict[str, list[int]], list[int]]:
    """Split a section's lines between its heading's components.

    Returns what each destination claimed and what nothing claimed. A line is
    only ever claimed on its own evidence; following a claimed line is not
    evidence, which is the same rule that stops content being classified merely
    because it follows a heading.
    """
    body_size, body_bold = _section_body_style([lines[index] for index in line_indexes])
    claimed: dict[str, list[int]] = {}
    unclaimed: list[int] = []
    # A local subheading owns the lines under it until the next one. This is
    # the one place following a line is evidence, because that is precisely
    # what writing a subheading means. A key-value label claims only its own
    # line: "Thai: Native" says nothing about the "Driving licence" line below.
    run_owner: str | None = None

    for line_index in line_indexes:
        line = lines[line_index]
        text = _without_marker(line.text)
        if _looks_like_subheading(line, body_size=body_size, body_bold=body_bold):
            subheading_owner = _exact_section_type(text)
            if any(
                subheading_owner == component.section_type
                for component in components
                if component.resolved
            ):
                run_owner = subheading_owner
                claimed.setdefault(run_owner or "", []).append(line_index)
                continue
            # A subheading naming something else ends the previous run rather
            # than joining it; what it introduces is not what came before.
            run_owner = None

        owner = _block_owner(line.text, components) or run_owner
        if owner is None:
            unclaimed.append(line_index)
        else:
            claimed.setdefault(owner, []).append(line_index)
    return claimed, unclaimed


def logical_sections(
    lines: list[ExtractedLine],
    heading: DetectedHeading,
    line_indexes: list[int],
) -> list[LogicalSection]:
    """Divide one physical section into the logical sections it contains.

    An ordinary heading yields exactly one, carrying the destination the
    heading already resolved to, so callers need no special case.

    A compound heading yields one per destination that owns content. The rules,
    and why each is the conservative reading:

    * No component names a destination -- ``SAFETY & TRAINING`` -- so nothing
      here can be trusted to route content. Everything goes to ``others``,
      under the heading as written.
    * Nothing in the section claimed itself. The section is then undivided
      evidence for one destination, and the first component that names one
      takes all of it: ``TRAINING & CERTIFICATIONS`` is a certifications
      section whichever order it was written in.
    * Some blocks claimed themselves and some did not. The section really is
      divided, so the claims stand and the remainder -- which is genuinely
      unowned, not merely unlabelled -- is preserved under ``others``.
    """
    components = split_heading(lines[heading.line_index].text)
    if len(components) < 2:
        return [
            LogicalSection(
                section_type=heading.section_type,
                heading=heading,
                line_indexes=tuple(line_indexes),
            )
        ]

    # A compound heading whose content all became sections of its own -- local
    # labels promoted to headings in their own right -- has nothing left to
    # divide. Emitting the empty remainder would register a destination that
    # owns no content, and the real section of that type would then be the
    # second occurrence and go unread.
    if not line_indexes:
        return []

    heading_text = lines[heading.line_index].text
    resolved = [component for component in components if component.resolved]
    if not resolved:
        return [
            LogicalSection(
                section_type="others",
                heading=heading,
                line_indexes=tuple(line_indexes),
                compound_heading_text=heading_text,
            )
        ]

    claimed, unclaimed = _assign_line_indexes(lines, line_indexes, components)
    if not claimed:
        return [
            LogicalSection(
                section_type=resolved[0].section_type or "others",
                heading=heading,
                line_indexes=tuple(line_indexes),
                compound_heading_text=heading_text,
            )
        ]

    sections = [
        LogicalSection(
            section_type=component.section_type or "others",
            heading=heading,
            line_indexes=tuple(claimed[component.section_type or ""]),
            compound_heading_text=heading_text,
        )
        for component in resolved
        if claimed.get(component.section_type or "")
    ]
    if unclaimed:
        sections.append(
            LogicalSection(
                section_type="others",
                heading=heading,
                line_indexes=tuple(unclaimed),
                compound_heading_text=heading_text,
            )
        )
    return sections


def routed_logical_sections(
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
    *,
    minimum_line_index: int = 0,
) -> list[LogicalSection]:
    """Every logical section of a document, in reading order.

    The single view both ``build_sections`` and the section parsers route from,
    so a compound heading cannot be split one way for the clean output and
    another way for the debug artifacts.
    """
    routed = _routed_section_headings(
        lines,
        headings,
        minimum_line_index=minimum_line_index,
    )
    sections: list[LogicalSection] = []
    for position, heading in enumerate(routed):
        end = (
            routed[position + 1].line_index
            if position + 1 < len(routed)
            else len(lines)
        )
        sections.extend(
            logical_sections(lines, heading, list(range(heading.line_index + 1, end)))
        )
    return sections
