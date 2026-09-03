"""The header profile: identity and contact details above the first section.

Extraction runs in precedence order so a later, weaker stage never overwrites a
stronger one: labelled personal attributes, then contact regexes, then link
annotations, then NER, then semantic classification, then geometry.
"""

from __future__ import annotations

import re
from typing import Any

import pymupdf

from restruct.configs import SETTINGS
from restruct.document.types import (
    DetectedHeading,
    ExtractedLine,
    HeaderEntityMatch,
    append_regex_matches,
    overlaps_existing as _overlaps_existing,
)
from restruct.geometry import resolve_span_box, rounded, union_by_page
from restruct.model import (
    DistilBertNerPredictor,
    EmbeddingModel,
    classify_profile_attribute_labels,
    ner_matches_for_profile,
    semantic_job_title_matches,
)
from restruct.parsers.urls import _url_matches_for_lines
from restruct.patterns.contacts import EMAIL_RE, PHONE_RE
from restruct.patterns.personal import (
    GENERIC_ATTRIBUTE_RE,
    LABELLED_ATTRIBUTE_RE,
    LOCATION_SEGMENT_RE,
    NATIONALITY_PHRASE_RE,
)
from restruct.patterns.separators import SEGMENT_RE
from restruct.structure.headings import first_header_boundary


def _header_attribute_kind(label: str) -> str:
    normalized = re.sub(r"[^a-z]+", " ", label.casefold()).strip()
    if normalized in {"date of birth", "birth date", "d o b", "dob"}:
        return "date_of_birth"
    if normalized in {"age", "current age"}:
        return "age"
    if normalized in {"gender", "sex"}:
        return "gender"
    if normalized in {"marital status", "martial status", "civil status", "marital"}:
        return "marital_status"
    if normalized in {
        "visa",
        "visa status",
        "visa type",
        "work visa",
        "immigration status",
        "residency visa",
        "work authorization",
        "right to work",
    }:
        return "visa_status"
    if normalized in {"nationality", "citizenship"}:
        return "nationality"
    return "current_residence"

def _labelled_header_attribute_matches(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    profile_indexes: list[int],
) -> list[HeaderEntityMatch]:
    matches: list[HeaderEntityMatch] = []
    for line_index in profile_indexes:
        line = lines[line_index]
        for match in LABELLED_ATTRIBUTE_RE.finditer(line.text):
            # The label establishes the field. Preserve its same-line value
            # verbatim instead of requiring a particular DOB/date format.
            raw_value = match.group("value")
            value = raw_value.strip(" \t|\u2022\u00b7,;:-\u2013\u2014\u200b\ufeff")
            if not value or not any(character.isalnum() for character in value):
                continue
            value_start = match.start("value") + raw_value.find(value)
            value_end = value_start + len(value)
            matches.append(
                HeaderEntityMatch(
                    kind=_header_attribute_kind(match.group("label")),
                    text=value,
                    line_index=line_index,
                    start=match.start(),
                    end=match.end(),
                    detection_method="label_pattern",
                    confidence=1.0,
                    bbox=tuple(
                        resolve_span_box(
                            document,
                            line,
                            value,
                            value_start,
                            value_end,
                        )
                    ),
                )
            )
    return matches

def _semantic_header_attribute_matches(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    profile_indexes: list[int],
    semantic_model: EmbeddingModel,
    existing_matches: list[HeaderEntityMatch],
) -> list[HeaderEntityMatch]:
    candidates: list[tuple[int, re.Match[str], str, str, int, int]] = []
    for line_index in profile_indexes:
        line = lines[line_index]
        for match in GENERIC_ATTRIBUTE_RE.finditer(line.text):
            if _overlaps_existing(
                existing_matches,
                line_index=line_index,
                start=match.start(),
                end=match.end(),
            ):
                continue
            raw_value = match.group("value")
            value = raw_value.strip(" \t|\u2022\u00b7,;:-\u2013\u2014\u200b\ufeff")
            label = match.group("label").strip()
            if not label or not value:
                continue
            value_start = match.start("value") + raw_value.find(value)
            candidates.append(
                (
                    line_index,
                    match,
                    label,
                    value,
                    value_start,
                    value_start + len(value),
                )
            )

    classifications = classify_profile_attribute_labels(
        semantic_model,
        [candidate[2] for candidate in candidates],
    )
    matches: list[HeaderEntityMatch] = []
    for candidate, (attribute_type, confidence) in zip(
        candidates,
        classifications,
        strict=True,
    ):
        if attribute_type is None:
            continue
        line_index, match, _, value, value_start, value_end = candidate
        matches.append(
            HeaderEntityMatch(
                kind=attribute_type,
                text=value,
                line_index=line_index,
                start=match.start(),
                end=match.end(),
                detection_method="minilm_label",
                confidence=confidence,
                bbox=tuple(
                    resolve_span_box(
                        document,
                        lines[line_index],
                        value,
                        value_start,
                        value_end,
                    )
                ),
            )
        )
    return matches

def _other_header_matches(
    lines: list[ExtractedLine],
    profile_indexes: list[int],
    existing_matches: list[HeaderEntityMatch],
) -> list[HeaderEntityMatch]:
    """Preserve every still-unclaimed visible header span as metadata."""
    other_matches: list[HeaderEntityMatch] = []
    for line_index in profile_indexes:
        text = lines[line_index].text
        occupied = [False] * len(text)
        for match in existing_matches:
            if match.line_index != line_index:
                continue
            for position in range(max(0, match.start), min(len(text), match.end)):
                occupied[position] = True

        cursor = 0
        while cursor < len(text):
            while cursor < len(text) and occupied[cursor]:
                cursor += 1
            raw_start = cursor
            while cursor < len(text) and not occupied[cursor]:
                cursor += 1
            raw_end = cursor
            raw_text = text[raw_start:raw_end]
            stripped = raw_text.strip(" \t|\u2022\u00b7/,;:-\u200b")
            if not stripped or not any(character.isalnum() for character in stripped):
                continue
            start = raw_start + raw_text.find(stripped)
            other_matches.append(
                HeaderEntityMatch(
                    kind="other",
                    text=stripped,
                    line_index=line_index,
                    start=start,
                    end=start + len(stripped),
                    detection_method="unclassified",
                )
            )
    return other_matches

def build_header_profile(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
    semantic_model: EmbeddingModel,
    ner_model: DistilBertNerPredictor,
) -> dict[str, Any] | None:
    """Detect the identity/contact region before the first likely section."""
    boundary = first_header_boundary(lines, headings)
    page_one_indexes = [index for index, line in enumerate(lines) if line.page == 1]
    if boundary is not None:
        boundary_top = lines[boundary.line_index].bbox[1]
        profile_indexes = [
            index
            for index in page_one_indexes
            if lines[index].bbox[1] < boundary_top
        ]
    else:
        profile_indexes = page_one_indexes[
            : SETTINGS.header_profile.maximum_lines_without_boundary
        ]
    if not profile_indexes:
        return None

    # Claim labelled personal attributes before broad contact regexes so a
    # numeric DOB cannot be consumed as a phone number. Then claim contacts so
    # semantic models never see usernames, domains, or phone fragments.
    matches = _labelled_header_attribute_matches(
        document,
        lines,
        profile_indexes,
    )
    matches.extend(
        _semantic_header_attribute_matches(
            document,
            lines,
            profile_indexes,
            semantic_model,
            matches,
        )
    )
    for line_index in profile_indexes:
        text = lines[line_index].text
        append_regex_matches(
            matches,
            line_index=line_index,
            text=text,
            kind="email",
            pattern=EMAIL_RE,
        )
        append_regex_matches(
            matches,
            line_index=line_index,
            text=text,
            kind="phone",
            pattern=PHONE_RE,
        )

    for url_match in _url_matches_for_lines(document, lines, profile_indexes):
        if _overlaps_existing(
            matches,
            line_index=url_match.line_index,
            start=url_match.start,
            end=url_match.end,
        ):
            continue
        matches.append(url_match)

    # DistilBERT receives contact-masked lines independently.
    ner_matches = ner_matches_for_profile(
        ner_model,
        lines,
        profile_indexes,
        matches,
    )
    ner_location_matches = [
        match for match in ner_matches if match.kind == "location"
    ]
    ner_name_matches = [
        match
        for match in ner_matches
        if match.kind == "name"
        and not _overlaps_existing(
            [*matches, *ner_location_matches],
            line_index=match.line_index,
            start=match.start,
            end=match.end,
        )
    ]
    matches.extend(
        match
        for match in ner_matches
        if match.kind not in {"name", "location"}
    )
    if ner_name_matches:
        # NER can see place-name fragments as people.  Keep the strongest visual
        # header name among model-backed candidates rather than emitting several.
        matches.append(
            max(
                ner_name_matches,
                key=lambda match: (
                    lines[match.line_index].size,
                    match.confidence or 0.0,
                    -match.line_index,
                ),
            )
        )
    if ner_location_matches:
        matches.append(
            max(
                ner_location_matches,
                key=lambda match: (
                    bool(LOCATION_SEGMENT_RE.fullmatch(match.text)),
                    -match.line_index,
                    match.confidence or 0.0,
                ),
            )
        )

    # Resolve titles before geometry guesses a missing name. This prevents a
    # title-only OCR header from being mislabeled as a person.
    matches.extend(
        semantic_job_title_matches(
            semantic_model,
            lines,
            profile_indexes,
            matches,
        )
    )

    if not any(match.kind == "name" for match in matches):
        name_candidates = [
            index
            for index in profile_indexes[:5]
            if 1 <= len(lines[index].text.split()) <= 6
            and any(character.isalpha() for character in lines[index].text)
            and not _overlaps_existing(
                matches,
                line_index=index,
                start=0,
                end=len(lines[index].text),
            )
        ]
        if name_candidates:
            name_index = max(
                name_candidates,
                key=lambda index: (
                    lines[index].size,
                    lines[index].bold,
                    -index,
                ),
            )
            matches.append(
                HeaderEntityMatch(
                    kind="name",
                    text=lines[name_index].text,
                    line_index=name_index,
                    start=0,
                    end=len(lines[name_index].text),
                    detection_method="geometry_fallback",
                )
            )

    if not any(match.kind == "location" for match in matches):
        for line_index in profile_indexes:
            text = lines[line_index].text
            location_match = next(
                (
                    segment_match
                    for segment_match in SEGMENT_RE.finditer(text)
                    if LOCATION_SEGMENT_RE.fullmatch(
                        segment_match.group(0).strip(" \t,;:-\u200b")
                    )
                ),
                None,
            )
            if location_match is None:
                continue
            stripped = location_match.group(0).strip(" \t,;:-\u200b")
            relative_start = location_match.group(0).find(stripped)
            start = location_match.start() + relative_start
            end = start + len(stripped)
            if _overlaps_existing(
                matches,
                line_index=line_index,
                start=start,
                end=end,
            ):
                continue
            matches.append(
                HeaderEntityMatch(
                    kind="location",
                    text=stripped,
                    line_index=line_index,
                    start=start,
                    end=end,
                    detection_method="regex_fallback",
                )
            )
            break

    if not any(match.kind == "nationality" for match in matches):
        for line_index in profile_indexes:
            text = lines[line_index].text
            nationality_match = next(NATIONALITY_PHRASE_RE.finditer(text), None)
            if nationality_match is None or _overlaps_existing(
                matches,
                line_index=line_index,
                start=nationality_match.start(),
                end=nationality_match.end(),
            ):
                continue
            matches.append(
                HeaderEntityMatch(
                    kind="nationality",
                    text=nationality_match.group(0),
                    line_index=line_index,
                    start=nationality_match.start(),
                    end=nationality_match.end(),
                    detection_method="regex_context_fallback",
                )
            )
            break

    matches.extend(_other_header_matches(lines, profile_indexes, matches))

    unique_matches: list[HeaderEntityMatch] = []
    seen: set[tuple[str, int, int, int]] = set()
    for match in sorted(matches, key=lambda item: (item.line_index, item.start, item.kind)):
        key = (match.kind, match.line_index, match.start, match.end)
        if key not in seen:
            seen.add(key)
            unique_matches.append(match)

    profile_lines = [lines[index] for index in profile_indexes]
    region_box = union_by_page(profile_lines)[0]["bbox"]
    entities = []
    for match in unique_matches:
        line = lines[match.line_index]
        entity: dict[str, Any] = {
            "type": match.kind,
            "text": match.text,
            "page": line.page,
            "bbox": (
                rounded(match.bbox)
                if match.bbox is not None
                else resolve_span_box(
                    document,
                    line,
                    match.text,
                    match.start,
                    match.end,
                )
            ),
            "detectionMethod": match.detection_method,
        }
        if match.kind == "url":
            entity["url"] = match.url or match.text
        if match.kind == "other":
            entity["inside"] = "headerProfile"
        if match.confidence is not None:
            entity["confidence"] = round(match.confidence, 4)
        entities.append(entity)

    boundary_value = None
    if boundary is not None:
        boundary_line = lines[boundary.line_index]
        boundary_value = {
            "text": boundary_line.text,
            "sectionType": boundary.section_type,
            "page": boundary_line.page,
            "bbox": rounded(boundary_line.bbox),
            "similarity": round(boundary.similarity, 4),
        }

    return {
        "page": 1,
        "bbox": region_box,
        "text": "\n".join(line.text for line in profile_lines),
        "stoppedAtSection": boundary_value,
        "entities": entities,
    }
