"""Heading detection and section-boundary routing.

A heading is accepted from typography and confirmed semantically, never from
position alone. Routing then separates true section boundaries from local
subheadings that merely look like them.
"""
from __future__ import annotations

import statistics

import pymupdf

from restruct.configs import SETTINGS
from restruct.document.types import DetectedHeading, ExtractedLine
from restruct.geometry import vertical_overlap
from restruct.patterns.bullets import BULLET_RE


def first_header_boundary(
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
) -> DetectedHeading | None:
    """Return the first high-confidence section heading on page one."""
    for heading in headings:
        line = lines[heading.line_index]
        if line.page != 1 or heading.line_index == 0:
            continue
        if _is_reliable_section_heading(line, heading):
            return heading
    return None

def _is_reliable_section_heading(
    line: ExtractedLine,
    heading: DetectedHeading,
) -> bool:
    text = line.text.replace("\u200b", "").replace("\ufeff", "").strip()
    exact_references = {
        reference.casefold().strip()
        for references in SETTINGS.section_references.values()
        for reference in references
    }
    if text.casefold() in exact_references:
        return True
    letters = [character for character in text if character.isalpha()]
    uppercase = bool(letters) and sum(character.isupper() for character in letters) / len(letters) >= SETTINGS.heading.uppercase_ratio
    return (
        uppercase
        and heading.similarity >= SETTINGS.header_profile.boundary_similarity_threshold
        and heading.similarity - heading.runner_up_similarity
        >= SETTINGS.header_profile.boundary_winner_margin
    )

def _shares_visual_row_with_right_content(
    lines: list[ExtractedLine],
    line_index: int,
) -> bool:
    candidate = lines[line_index]
    candidate_box = pymupdf.Rect(candidate.bbox)
    for other_index, other in enumerate(lines):
        if other_index == line_index or other.page != candidate.page:
            continue
        other_box = pymupdf.Rect(other.bbox)
        if other_box.x0 <= candidate_box.x1 + max(2.0, candidate.size * 0.5):
            continue
        if vertical_overlap(candidate_box, other_box) >= min(
            candidate_box.height, other_box.height
        ) * 0.45:
            return True
    return False

def _is_local_subheading_candidate(
    lines: list[ExtractedLine],
    heading: DetectedHeading,
    parent_heading: DetectedHeading,
) -> bool:
    candidate = lines[heading.line_index]
    parent = lines[parent_heading.line_index]
    if candidate.page < parent.page:
        return False
    smaller = candidate.size < parent.size * 0.98
    indented = candidate.bbox[0] > parent.bbox[0] + max(4.0, parent.size * 0.5)
    return (
        smaller
        and indented
        and _shares_visual_row_with_right_content(lines, heading.line_index)
    )

def _peer_sized_section_heading(
    lines: list[ExtractedLine],
    line_index: int,
    parent_heading: DetectedHeading,
    semantic_heading: DetectedHeading | None,
) -> DetectedHeading | None:
    line = lines[line_index]
    parent = lines[parent_heading.line_index]
    text = line.text.replace("\u200b", "").replace("\ufeff", "").strip()
    if (
        not text
        or BULLET_RE.match(text)
        or len(text) > SETTINGS.heading.maximum_characters
        or len(text.split()) > SETTINGS.heading.maximum_words
        or line.size < parent.size * 0.98
    ):
        return None
    letters = [character for character in text if character.isalpha()]
    uppercase = bool(letters) and sum(
        character.isupper() for character in letters
    ) / len(letters) >= SETTINGS.heading.uppercase_ratio
    if not uppercase and not line.bold:
        return None
    section_type = semantic_heading.section_type if semantic_heading else "others"
    if section_type == parent_heading.section_type:
        return None
    return DetectedHeading(
        line_index=line_index,
        section_type=section_type,
        similarity=semantic_heading.similarity if semantic_heading else 0.0,
        runner_up_similarity=(
            semantic_heading.runner_up_similarity if semantic_heading else 0.0
        ),
    )

def _routed_section_headings(
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
    *,
    minimum_line_index: int = 0,
) -> list[DetectedHeading]:
    """Keep major section boundaries while demoting geometric child labels."""
    reliable_by_index = {
        heading.line_index: heading
        for heading in headings
        if heading.line_index >= minimum_line_index
        and _is_reliable_section_heading(lines[heading.line_index], heading)
    }
    semantic_by_index = {
        heading.line_index: heading
        for heading in headings
        if heading.line_index >= minimum_line_index
    }
    routed: list[DetectedHeading] = []
    for line_index in range(minimum_line_index, len(lines)):
        heading = reliable_by_index.get(line_index)
        if heading is None and routed:
            heading = _peer_sized_section_heading(
                lines,
                line_index,
                routed[-1],
                semantic_by_index.get(line_index),
            )
        if heading is None:
            continue
        if routed and _is_local_subheading_candidate(lines, heading, routed[-1]):
            continue
        routed.append(heading)
    return routed

def _section_body_style(
    content_lines: list[ExtractedLine],
) -> tuple[float, bool]:
    prose_lines = [
        line
        for line in content_lines
        if len(line.text.split()) > SETTINGS.section_router.maximum_subheading_words
        or line.text.rstrip().endswith((".", ",", ";", ":"))
    ]
    reference_lines = prose_lines or content_lines
    sizes = [line.size for line in reference_lines if line.size > 0]
    body_size = statistics.median(sizes) if sizes else 0.0
    body_bold = bool(reference_lines) and sum(
        line.bold for line in reference_lines
    ) > len(reference_lines) / 2
    return body_size, body_bold

def _looks_like_subheading(
    line: ExtractedLine,
    *,
    body_size: float,
    body_bold: bool,
) -> bool:
    text = line.text.strip()
    if not text or text.endswith((".", ",", ";", ":")):
        return False
    if len(text) > SETTINGS.section_router.maximum_subheading_characters:
        return False
    if len(text.split()) > SETTINGS.section_router.maximum_subheading_words:
        return False
    size_contrast = (
        body_size > 0
        and line.size
        >= body_size * SETTINGS.section_router.subheading_font_size_multiplier
    )
    bold_contrast = line.bold and not body_bold
    return size_contrast or bold_contrast
