"""Geometry-aware routing and reconstruction for resume sections."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageDraw

from extractor_v1.configs import SETTINGS
from extractor_v1.model import (
    DetectedHeading,
    DistilBertNerPredictor,
    EmbeddingModel,
    ExtractedLine,
    classify_job_title_candidates,
    classify_profile_attribute_labels,
)


_BULLET_RE = re.compile(
    r"^\s*(?:[-+*•●▪◦‣¢]|\d+[.)])"
    r"[\s\u200b\ufeff]*"
)
_EXPERIENCE_SEPARATOR_RE = re.compile(r"\s*(?:[|•·]|–|—|\s-\s)\s*")
_EDUCATION_SEPARATOR_RE = re.compile(r"\s*(?:[|•·]|–|—|\s-\s)\s*")
_MONTH_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)"
)
_DATED_YEAR_PATTERN = rf"(?:{_MONTH_PATTERN}(?:\s+|-))?(?:19|20)\d{{2}}"
_DATE_RANGE_RE = re.compile(
    rf"\b(?:"
    rf"(?:From\s+)?(?:19|20)\d{{2}}\s*-\s*(?:(?:19|20)\d{{2}}|Present|Current|Now)"
    rf"|(?:From\s+)?{_DATED_YEAR_PATTERN}"
    rf"(?:\s+-\s+|\s*[\u2013\u2014]\s*|\s+to\s+)"
    rf"(?:{_DATED_YEAR_PATTERN}|Present|Current|Now)"
    rf")\b",
    re.IGNORECASE,
)
_SINGLE_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_EDUCATION_TITLE_RE = re.compile(
    r"\b(?:associate(?:'s)?|bachelor(?:'s)?|master(?:'s)?|doctor(?:ate|al)?|"
    r"ph\.?\s*d\.?|m\.?\s*(?:sc|s|a|eng|ba)\.?|b\.?\s*(?:sc|s|a|eng|ba)\.?|"
    r"a\.?\s*(?:a|s)\.?|mba|degree|diploma|certificate|certification|"
    r"undergraduate|graduate|postgraduate|vocational)\b",
    re.IGNORECASE,
)
_EDUCATION_INSTITUTION_RE = re.compile(
    r"\b(?:university|college|institute|institution|school|academy|polytechnic|"
    r"conservatory|seminary|faculty)\b",
    re.IGNORECASE,
)
_GPA_RE = re.compile(
    r"\b(?P<label>c?gpa|grade(?:\s+point\s+average)?)\s*:?[\s-]*"
    r"(?P<value>\d+(?:\.\d+)?)"
    r"(?:\s*(?:/|out\s+of)\s*(?P<scale>\d+(?:\.\d+)?))?\b",
    re.IGNORECASE,
)
_EDUCATION_PARAGRAPH_RE = re.compile(
    r"^\s*(?:relevant\s+)?(?:coursework|courses?|subjects?|modules?)\b",
    re.IGNORECASE,
)
_PAGE_FOOTER_RE = re.compile(r"\bpage\s+\d+\s*$", re.IGNORECASE)
_SKILL_INLINE_COLON_RE = re.compile(r"^(?P<label>[^:\t]{1,60}):\s*(?P<body>.+)$")
_SKILL_INLINE_TAB_RE = re.compile(r"^(?P<label>[^\t]{1,60})\t+\s*(?P<body>.+)$")
_SKILL_INLINE_DASH_RE = re.compile(
    r"^(?P<label>.{1,60}?)\s+(?:-|\u2013|\u2014)\s+(?P<body>.+)$"
)
_PROFILE_ATTRIBUTE_INLINE_RE = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z .'/]{1,40}?)"
    r"\s*(?::|[-\u2013\u2014])\s*(?P<body>.+)$"
)
_SUPPLEMENTARY_SECTION_TYPES = (
    "certifications",
    "licenses",
    "tools_equipment",
    "languages",
    "volunteering",
    "awards",
    "publications",
    "references",
    "interests",
    "others",
)


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
        vertical_overlap = min(candidate_box.y1, other_box.y1) - max(
            candidate_box.y0,
            other_box.y0,
        )
        if vertical_overlap >= min(candidate_box.height, other_box.height) * 0.45:
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
        or _BULLET_RE.match(text)
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


def _content_blocks(
    lines: list[ExtractedLine],
    line_indexes: list[int],
    url_entities_by_line: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Route lines into subheadings, paragraphs, or reconstructed bullets."""
    content_lines = [lines[index] for index in line_indexes]
    body_size, body_bold = _section_body_style(content_lines)
    blocks: list[dict[str, Any]] = []

    for line_index in line_indexes:
        line = lines[line_index]
        line_url_entities = url_entities_by_line.get(line_index, [])
        bullet_match = _BULLET_RE.match(line.text)
        if bullet_match is not None:
            bullet_text = line.text[bullet_match.end():].strip()
            if not bullet_text:
                continue
            block = {
                "type": "bullet",
                "text": bullet_text,
                "page": line.page,
                "bbox": [round(value, 2) for value in line.bbox],
                "detectionMethod": "bullet_marker",
                "_lastLineBbox": [round(value, 2) for value in line.bbox],
            }
            if line_url_entities:
                block["entities"] = line_url_entities
            blocks.append(block)
            continue

        role = (
            "subheading"
            if _looks_like_subheading(
                line,
                body_size=body_size,
                body_bold=body_bold,
            )
            else "paragraph"
        )
        rounded_bbox = [round(value, 2) for value in line.bbox]
        if role == "paragraph" and blocks and blocks[-1]["type"] == "bullet":
            previous = blocks[-1]
            previous_box = pymupdf.Rect(previous["_lastLineBbox"])
            current_box = pymupdf.Rect(line.bbox)
            vertical_gap = current_box.y0 - previous_box.y1
            horizontal_overlap = max(
                0.0,
                min(previous_box.x1, current_box.x1)
                - max(previous_box.x0, current_box.x0),
            )
            maximum_gap = max(
                previous_box.height,
                current_box.height,
            ) * SETTINGS.section_router.paragraph_gap_multiplier
            if (
                previous["page"] == line.page
                and -2.0 <= vertical_gap <= maximum_gap
                and horizontal_overlap > 0
            ):
                previous["text"] += "\n" + line.text
                previous["bbox"] = [
                    round(value, 2)
                    for value in (pymupdf.Rect(previous["bbox"]) | current_box)
                ]
                previous["_lastLineBbox"] = rounded_bbox
                if line_url_entities:
                    previous.setdefault("entities", []).extend(line_url_entities)
                continue

        if role == "paragraph" and blocks and blocks[-1]["type"] == "paragraph":
            previous = blocks[-1]
            previous_box = pymupdf.Rect(previous["_lastLineBbox"])
            current_box = pymupdf.Rect(line.bbox)
            vertical_gap = current_box.y0 - previous_box.y1
            horizontal_overlap = max(
                0.0,
                min(previous_box.x1, current_box.x1)
                - max(previous_box.x0, current_box.x0),
            )
            maximum_gap = max(
                previous_box.height,
                current_box.height,
            ) * SETTINGS.section_router.paragraph_gap_multiplier
            if (
                previous["page"] == line.page
                and -2.0 <= vertical_gap <= maximum_gap
                and horizontal_overlap > 0
            ):
                previous["text"] += "\n" + line.text
                previous["bbox"] = [
                    round(value, 2)
                    for value in (pymupdf.Rect(previous["bbox"]) | current_box)
                ]
                previous["_lastLineBbox"] = rounded_bbox
                if line_url_entities:
                    previous.setdefault("entities", []).extend(line_url_entities)
                continue

        block: dict[str, Any] = {
            "type": role,
            "text": line.text,
            "page": line.page,
            "bbox": rounded_bbox,
            "detectionMethod": (
                "geometry_typography"
                if role == "subheading"
                else "geometry_default"
            ),
            "_lastLineBbox": rounded_bbox,
        }
        if line_url_entities:
            block["entities"] = line_url_entities
        blocks.append(block)

    for block in blocks:
        block.pop("_lastLineBbox", None)
    return blocks


def build_sections(
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
    url_entities_by_line: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Use MiniLM-confirmed headings as boundaries and geometry within sections."""
    first_boundary = first_header_boundary(lines, headings)
    if first_boundary is None:
        return []
    routed_headings = _routed_section_headings(
        lines,
        headings,
        minimum_line_index=first_boundary.line_index,
    )
    semantic_heading_indexes = {heading.line_index for heading in headings}

    sections: list[dict[str, Any]] = []
    for position, heading in enumerate(routed_headings):
        next_heading_index = (
            routed_headings[position + 1].line_index
            if position + 1 < len(routed_headings)
            else len(lines)
        )
        heading_line = lines[heading.line_index]
        content_indexes = list(range(heading.line_index + 1, next_heading_index))
        heading_value: dict[str, Any] = {
            "text": heading_line.text,
            "page": heading_line.page,
            "bbox": [round(value, 2) for value in heading_line.bbox],
            "similarity": round(heading.similarity, 4),
            "detectionMethod": (
                "geometry_semantic"
                if heading.line_index in semantic_heading_indexes
                else "geometry_unknown_boundary"
            ),
        }
        heading_url_entities = url_entities_by_line.get(heading.line_index, [])
        if heading_url_entities:
            heading_value["entities"] = heading_url_entities
        sections.append(
            {
                "sectionType": heading.section_type,
                "heading": heading_value,
                "content": _content_blocks(
                    lines,
                    content_indexes,
                    url_entities_by_line,
                ),
            }
        )
    return sections


def _span_bbox(
    document: pymupdf.Document,
    line: ExtractedLine,
    text: str,
    start: int,
    end: int,
) -> list[float]:
    found = document[line.page - 1].search_for(text, clip=pymupdf.Rect(line.bbox))
    if found:
        return [round(value, 2) for value in found[0]]
    line_box = pymupdf.Rect(line.bbox)
    character_width = line_box.width / max(1, len(line.text))
    return [
        round(line_box.x0 + character_width * start, 2),
        round(line_box.y0, 2),
        round(line_box.x0 + character_width * end, 2),
        round(line_box.y1, 2),
    ]


def _metadata_candidates(
    text: str,
    date_spans: list[re.Match[str]],
    separator_re: re.Pattern[str],
    *,
    preserve_education_hyphens: bool = False,
) -> list[tuple[str, int, int]]:
    """Split metadata while retaining exact character offsets into its source line."""
    segment_ranges: list[tuple[int, int]] = []
    cursor = 0
    for separator in separator_re.finditer(text):
        if any(
            match.start() < separator.end() and separator.start() < match.end()
            for match in date_spans
        ):
            continue
        separator_text = separator.group(0).strip()
        if preserve_education_hyphens and separator_text == "-":
            left = text[cursor:separator.start()].strip()
            right = text[separator.end():].strip()
            should_split = bool(
                (_EDUCATION_TITLE_RE.search(left) and _EDUCATION_INSTITUTION_RE.search(right))
                or _EDUCATION_INSTITUTION_RE.search(left)
            )
            if not should_split:
                continue
        segment_ranges.append((cursor, separator.start()))
        cursor = separator.end()
    segment_ranges.append((cursor, len(text)))

    candidates: list[tuple[str, int, int]] = []
    for base_start, base_end in segment_ranges:
        residual_ranges = [(base_start, base_end)]
        for date_match in date_spans:
            occupied_start, occupied_end = date_match.start(), date_match.end()
            next_ranges: list[tuple[int, int]] = []
            for start, end in residual_ranges:
                if occupied_end <= start or end <= occupied_start:
                    next_ranges.append((start, end))
                    continue
                if start < occupied_start:
                    next_ranges.append((start, occupied_start))
                if occupied_end < end:
                    next_ranges.append((occupied_end, end))
            residual_ranges = next_ranges
        for raw_start, raw_end in residual_ranges:
            raw = text[raw_start:raw_end]
            value = raw.strip(" \t,;:-–—\u200b\ufeff")
            if not value:
                continue
            start = raw_start + raw.find(value)
            candidates.append((value, start, start + len(value)))
    return candidates


def _experience_line_entities(
    document: pymupdf.Document,
    line: ExtractedLine,
    model: EmbeddingModel,
    ner_model: DistilBertNerPredictor,
    urls: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {
        "jobTitles": [], "companies": [], "dates": [], "locations": [], "urls": urls,
    }
    date_spans = list(_DATE_RANGE_RE.finditer(line.text))
    for match in date_spans:
        result["dates"].append({
            "text": match.group(0), "page": line.page,
            "bbox": _span_bbox(document, line, match.group(0), match.start(), match.end()),
            "detectionMethod": "date_regex",
        })

    candidates = _metadata_candidates(
        line.text,
        date_spans,
        _EXPERIENCE_SEPARATOR_RE,
    )
    classifications = classify_job_title_candidates(model, [item[0] for item in candidates])
    company_markers = re.compile(
        r"\b(?:co\.?|company|ltd\.?|limited|inc\.?|corp\.?|corporation|llc|plc)\b",
        re.IGNORECASE,
    )

    def candidate_urls(
        segment: str,
        start: int,
        end: int,
    ) -> list[dict[str, Any]]:
        normalized_segment = " ".join(
            segment.replace("\u200b", "").replace("\ufeff", "").casefold().split()
        ).strip(" |•·-–—")
        segment_box = pymupdf.Rect(_span_bbox(document, line, segment, start, end))
        matches: list[dict[str, Any]] = []
        for url in urls:
            normalized_text = " ".join(
                str(url.get("text", ""))
                .replace("\u200b", "")
                .replace("\ufeff", "")
                .casefold()
                .split()
            ).strip(" |•·-–—")
            text_matches = bool(
                normalized_segment
                and normalized_text
                and (
                    normalized_segment == normalized_text
                    or normalized_segment in normalized_text
                    or normalized_text in normalized_segment
                )
            )
            url_bbox = url.get("bbox")
            geometry_matches = False
            if url_bbox is not None:
                url_box = pymupdf.Rect(url_bbox)
                overlap = segment_box & url_box
                geometry_matches = bool(
                    overlap.width > 0
                    and overlap.height > 0
                    and overlap.get_area()
                    >= min(segment_box.get_area(), url_box.get_area()) * 0.5
                )
            if text_matches or geometry_matches:
                matches.append(url)
        return matches

    evidence: list[dict[str, Any]] = []
    for (segment, start, end), (accepted, confidence) in zip(
        candidates,
        classifications,
        strict=True,
    ):
        predictions = ner_model.predict_entities(
            segment,
            ["organization", "location"],
            SETTINGS.ner.minimum_confidence,
        )
        organizations = [item for item in predictions if item["label"] == "organization"]
        locations = [item for item in predictions if item["label"] == "location"]
        matching_urls = candidate_urls(segment, start, end)
        external_urls = [
            url
            for url in matching_urls
            if not any(
                excluded in str(url.get("url", "")).casefold()
                for excluded in ("linkedin.com", "github.com", "mailto:", "tel:")
            )
        ]
        evidence.append(
            {
                "segment": segment,
                "start": start,
                "end": end,
                "titleAccepted": accepted,
                "titleConfidence": confidence,
                "organizations": organizations,
                "locations": locations,
                "matchingUrls": matching_urls,
                "externalUrls": external_urls,
                "companyMarker": company_markers.search(segment) is not None,
            }
        )

    accepted_indexes = [
        index for index, item in enumerate(evidence) if item["titleAccepted"]
    ]
    company_evidence_indexes = {
        index
        for index, item in enumerate(evidence)
        if item["companyMarker"]
        or (
            len(evidence) > 1
            and bool(item["externalUrls"])
        )
        or (
            len(evidence) > 1
            and bool(item["organizations"])
            and any(other != index for other in accepted_indexes)
        )
    }
    title_indexes = [
        index for index in accepted_indexes if index not in company_evidence_indexes
    ]
    primary_title_index = (
        max(title_indexes, key=lambda index: float(evidence[index]["titleConfidence"]))
        if title_indexes
        else None
    )

    for index, item in enumerate(evidence):
        segment = str(item["segment"])
        start = int(item["start"])
        end = int(item["end"])
        organizations = item["organizations"]
        locations = item["locations"]

        if index == primary_title_index:
            result["jobTitles"].append({
                "text": segment,
                "page": line.page,
                "bbox": _span_bbox(document, line, segment, start, end),
                "confidence": round(float(item["titleConfidence"]), 4),
                "detectionMethod": "minilm_reconciled",
            })
            continue

        # A compact metadata segment may contain both the employer and its
        # comma-delimited location. Split only when NER grounds both roles and
        # there is a comma between the organization and location evidence.
        if organizations and locations and "," in segment:
            organization_end = max(
                int(prediction["end"]) for prediction in organizations
            )
            location_start = min(
                int(prediction["start"]) for prediction in locations
            )
            split_candidates = [
                match.start()
                for match in re.finditer(",", segment)
                if organization_end <= match.start() < location_start
            ]
            if split_candidates:
                split_at = split_candidates[-1]
                raw_company = segment[:split_at]
                raw_location = segment[split_at + 1:]
                company_text = raw_company.strip()
                location_text = raw_location.strip()
                if company_text and location_text:
                    company_start = start + raw_company.find(company_text)
                    location_relative_start = split_at + 1 + raw_location.find(
                        location_text
                    )
                    location_source_start = start + location_relative_start
                    company_value: dict[str, Any] = {
                        "text": company_text,
                        "page": line.page,
                        "bbox": _span_bbox(
                            document,
                            line,
                            company_text,
                            company_start,
                            company_start + len(company_text),
                        ),
                        "confidence": round(
                            max(
                                float(prediction["score"])
                                for prediction in organizations
                            ),
                            4,
                        ),
                        "detectionMethod": "distilbert_ner_reconciled",
                    }
                    if item["matchingUrls"]:
                        company_value["urls"] = item["matchingUrls"]
                    result["companies"].append(company_value)
                    result["locations"].append({
                        "text": location_text,
                        "page": line.page,
                        "bbox": _span_bbox(
                            document,
                            line,
                            location_text,
                            location_source_start,
                            location_source_start + len(location_text),
                        ),
                        "confidence": round(
                            max(
                                float(prediction["score"])
                                for prediction in locations
                            ),
                            4,
                        ),
                        "detectionMethod": "distilbert_ner_reconciled",
                    })
                    continue

        full_segment_company = bool(
            index in company_evidence_indexes
            or (
                organizations
                and (
                    not item["titleAccepted"]
                    or primary_title_index is not None
                )
            )
        )
        if full_segment_company:
            value: dict[str, Any] = {
                "text": segment,
                "page": line.page,
                "bbox": _span_bbox(document, line, segment, start, end),
                "detectionMethod": (
                    "url_company_reconciled"
                    if item["externalUrls"]
                    else "distilbert_ner_reconciled"
                ),
            }
            if organizations:
                value["confidence"] = round(
                    max(float(prediction["score"]) for prediction in organizations),
                    4,
                )
            if item["matchingUrls"]:
                value["urls"] = item["matchingUrls"]
            result["companies"].append(value)
            continue

        if locations and ("," in segment or (len(locations) > 1 and not organizations)):
            result["locations"].append({
                "text": segment,
                "page": line.page,
                "bbox": _span_bbox(document, line, segment, start, end),
                "confidence": round(max(float(item["score"]) for item in locations), 4),
                "detectionMethod": "distilbert_ner_reconciled",
            })
            continue

        if item["titleAccepted"]:
            result["jobTitles"].append({
                "text": segment,
                "page": line.page,
                "bbox": _span_bbox(document, line, segment, start, end),
                "confidence": round(float(item["titleConfidence"]), 4),
                "detectionMethod": "minilm_reconciled",
            })
            continue

        for prediction in [*organizations, *locations]:
            prediction_start = start + int(prediction["start"])
            prediction_end = start + int(prediction["end"])
            text = line.text[prediction_start:prediction_end].strip()
            if not text:
                continue
            key = "companies" if prediction["label"] == "organization" else "locations"
            value = {
                "text": text,
                "page": line.page,
                "bbox": _span_bbox(document, line, text, prediction_start, prediction_end),
                "confidence": round(float(prediction["score"]), 4),
                "detectionMethod": "distilbert_ner",
            }
            if key == "companies" and item["matchingUrls"]:
                value["urls"] = item["matchingUrls"]
            result[key].append(value)
    return result


def build_experience_debug(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
    model: EmbeddingModel,
    ner_model: DistilBertNerPredictor,
    url_entities_by_line: dict[int, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    routed = _routed_section_headings(lines, headings)
    position = next((i for i, item in enumerate(routed) if item.section_type == "experience"), None)
    if position is None:
        return None
    heading = routed[position]
    end = routed[position + 1].line_index if position + 1 < len(routed) else len(lines)
    heading_line = lines[heading.line_index]
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def new_entry() -> dict[str, Any]:
        entry: dict[str, Any] = {
            "detectionMethod": "ner_minilm_reconciliation",
            "subheadingLines": [], "jobTitles": [], "companies": [], "dates": [],
            "locations": [], "urls": [], "paragraphs": [], "bullets": [],
            "_bodyStarted": False, "_lastBodyType": None, "_lastLineBbox": None,
        }
        entries.append(entry)
        return entry

    for line_index in range(heading.line_index + 1, end):
        line = lines[line_index]
        bullet = _BULLET_RE.match(line.text)
        if bullet:
            current = current or new_entry()
            current["_bodyStarted"] = True
            current["bullets"].append({
                "text": line.text[bullet.end():].strip(), "page": line.page,
                "bbox": [round(value, 2) for value in line.bbox],
                "detectionMethod": "bullet_marker",
            })
            current["_lastBodyType"] = "bullet"
            current["_lastLineBbox"] = [round(value, 2) for value in line.bbox]
            continue

        can_be_metadata = (
            current is None
            or not current["_bodyStarted"]
            or line.bold
            or _DATE_RANGE_RE.search(line.text) is not None
        )
        entities = (
            _experience_line_entities(
                document, line, model, ner_model, url_entities_by_line.get(line_index, [])
            )
            if can_be_metadata
            else {"jobTitles": [], "companies": [], "dates": [], "locations": [], "urls": []}
        )
        primary = bool(entities["jobTitles"] or entities["companies"] or entities["dates"])
        metadata = primary or (bool(entities["locations"]) and current is not None and not current["_bodyStarted"])
        if metadata:
            if current is None or (primary and current["_bodyStarted"]):
                current = new_entry()
            current["subheadingLines"].append({
                "text": line.text, "page": line.page,
                "bbox": [round(value, 2) for value in line.bbox],
                "detectionMethod": "experience_metadata",
            })
            for key in ("jobTitles", "companies", "dates", "locations", "urls"):
                current[key].extend(entities[key])
            continue

        current = current or new_entry()
        current["_bodyStarted"] = True
        current_box = pymupdf.Rect(line.bbox)
        if current["_lastBodyType"] == "bullet" and current["bullets"]:
            previous_box = pymupdf.Rect(current["_lastLineBbox"])
            gap = current_box.y0 - previous_box.y1
            if -2.0 <= gap <= max(previous_box.height, current_box.height) * SETTINGS.section_router.paragraph_gap_multiplier:
                previous = current["bullets"][-1]
                previous["text"] += "\n" + line.text
                previous["bbox"] = [round(value, 2) for value in (pymupdf.Rect(previous["bbox"]) | current_box)]
                current["_lastLineBbox"] = [round(value, 2) for value in line.bbox]
                continue
        paragraph = {
            "text": line.text, "page": line.page,
            "bbox": [round(value, 2) for value in line.bbox],
            "detectionMethod": "geometry_default",
        }
        if current["paragraphs"] and current["paragraphs"][-1]["page"] == line.page:
            previous = current["paragraphs"][-1]
            previous_box, current_box = pymupdf.Rect(previous["bbox"]), pymupdf.Rect(line.bbox)
            gap = current_box.y0 - previous_box.y1
            if -2.0 <= gap <= max(previous_box.height, current_box.height) * SETTINGS.section_router.paragraph_gap_multiplier:
                previous["text"] += "\n" + line.text
                previous["bbox"] = [round(value, 2) for value in (previous_box | current_box)]
                current["_lastBodyType"] = "paragraph"
                current["_lastLineBbox"] = [round(value, 2) for value in line.bbox]
                continue
        current["paragraphs"].append(paragraph)
        current["_lastBodyType"] = "paragraph"
        current["_lastLineBbox"] = [round(value, 2) for value in line.bbox]

    for entry in entries:
        entry.pop("_bodyStarted", None)
        entry.pop("_lastBodyType", None)
        entry.pop("_lastLineBbox", None)
    next_heading = routed[position + 1] if position + 1 < len(routed) else None
    return {
        "sectionType": "experience",
        "heading": {
            "text": heading_line.text, "page": heading_line.page,
            "bbox": [round(value, 2) for value in heading_line.bbox],
            "similarity": round(heading.similarity, 4),
            "detectionMethod": "geometry_semantic",
        },
        "entries": entries,
        "stoppedAtSection": (
            {"sectionType": next_heading.section_type, "text": lines[next_heading.line_index].text}
            if next_heading else None
        ),
    }


def _visual_rows(
    lines: list[ExtractedLine],
    line_indexes: range,
) -> list[list[tuple[int, ExtractedLine]]]:
    """Cluster separately extracted left/right cells into visual rows."""
    ordered = sorted(
        ((index, lines[index]) for index in line_indexes),
        key=lambda item: (item[1].page, item[1].bbox[1], item[1].bbox[0]),
    )
    rows: list[list[tuple[int, ExtractedLine]]] = []
    for line_index, line in ordered:
        line_box = pymupdf.Rect(line.bbox)
        if rows and rows[-1][0][1].page == line.page:
            row_boxes = [pymupdf.Rect(item.bbox) for _, item in rows[-1]]
            row_y0 = min(box.y0 for box in row_boxes)
            row_y1 = max(box.y1 for box in row_boxes)
            overlap = min(row_y1, line_box.y1) - max(row_y0, line_box.y0)
            minimum_height = min(row_y1 - row_y0, line_box.height)
            center_difference = abs((row_y0 + row_y1) / 2 - (line_box.y0 + line_box.y1) / 2)
            if overlap >= minimum_height * 0.45 or center_difference <= minimum_height * 0.35:
                rows[-1].append((line_index, line))
                rows[-1].sort(key=lambda item: item[1].bbox[0])
                continue
        rows.append([(line_index, line)])
    return rows


def _row_value(row: list[tuple[int, ExtractedLine]]) -> dict[str, Any]:
    boxes = [pymupdf.Rect(line.bbox) for _, line in row]
    row_box = boxes[0]
    for box in boxes[1:]:
        row_box |= box
    return {
        "page": row[0][1].page,
        "bbox": [round(value, 2) for value in row_box],
        "detectionMethod": "geometry_row",
        "cells": [
            {
                "text": line.text,
                "page": line.page,
                "bbox": [round(value, 2) for value in line.bbox],
                "lineIndex": line_index,
            }
            for line_index, line in row
        ],
    }


def _education_row_entities(
    document: pymupdf.Document,
    row: list[tuple[int, ExtractedLine]],
    ner_model: DistilBertNerPredictor,
    url_entities_by_line: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "titles": [], "institutions": [], "dates": [], "locations": [],
        "gpa": [], "skills": [], "urls": [], "handledLineIndexes": set(),
    }
    for line_index, line in row:
        line_urls = url_entities_by_line.get(line_index, [])
        result["urls"].extend(line_urls)
        if _BULLET_RE.match(line.text):
            continue
        date_spans = list(_DATE_RANGE_RE.finditer(line.text))
        for match in date_spans:
            result["dates"].append({
                "text": match.group(0),
                "page": line.page,
                "bbox": _span_bbox(document, line, match.group(0), match.start(), match.end()),
                "detectionMethod": "date_regex",
            })

        gpa_matches = list(_GPA_RE.finditer(line.text))
        for match in gpa_matches:
            result["gpa"].append({
                "text": match.group(0),
                "value": float(match.group("value")),
                "scale": float(match.group("scale")) if match.group("scale") else None,
                "page": line.page,
                "bbox": _span_bbox(document, line, match.group(0), match.start(), match.end()),
                "detectionMethod": "gpa_regex",
            })
        if gpa_matches:
            result["handledLineIndexes"].add(line_index)

        candidates = _metadata_candidates(
            line.text,
            date_spans,
            _EDUCATION_SEPARATOR_RE,
            preserve_education_hyphens=True,
        )
        for segment, start, end in candidates:
            if any(match.start() < end and start < match.end() for match in gpa_matches):
                continue
            has_title = _EDUCATION_TITLE_RE.search(segment) is not None
            has_institution = _EDUCATION_INSTITUTION_RE.search(segment) is not None
            if has_title and not has_institution:
                result["titles"].append({
                    "text": segment,
                    "page": line.page,
                    "bbox": _span_bbox(document, line, segment, start, end),
                    "detectionMethod": "education_title_pattern",
                })
                continue
            if has_institution:
                value: dict[str, Any] = {
                    "text": segment,
                    "page": line.page,
                    "bbox": _span_bbox(document, line, segment, start, end),
                    "detectionMethod": "institution_pattern",
                }
                if line_urls:
                    value["urls"] = line_urls
                result["institutions"].append(value)
                if has_title:
                    result["titles"].append({
                        "text": segment,
                        "page": line.page,
                        "bbox": _span_bbox(document, line, segment, start, end),
                        "detectionMethod": "education_title_pattern",
                    })
                continue

            predictions = ner_model.predict_entities(
                segment,
                ["organization", "location"],
                SETTINGS.ner.minimum_confidence,
            )
            organizations = [
                prediction
                for prediction in predictions
                if prediction["label"] == "organization"
            ]
            locations = [
                prediction
                for prediction in predictions
                if prediction["label"] == "location"
            ]
            if locations and not organizations and "," in segment:
                result["locations"].append({
                    "text": segment,
                    "page": line.page,
                    "bbox": _span_bbox(document, line, segment, start, end),
                    "confidence": round(
                        max(float(prediction["score"]) for prediction in locations),
                        4,
                    ),
                    "detectionMethod": "distilbert_ner_reconciled",
                })
                continue
            for prediction in predictions:
                prediction_start = start + int(prediction["start"])
                prediction_end = start + int(prediction["end"])
                text = line.text[prediction_start:prediction_end].strip()
                if not text:
                    continue
                key = "institutions" if prediction["label"] == "organization" else "locations"
                value = {
                    "text": text,
                    "page": line.page,
                    "bbox": _span_bbox(document, line, text, prediction_start, prediction_end),
                    "confidence": round(float(prediction["score"]), 4),
                    "detectionMethod": "distilbert_ner",
                }
                if key == "institutions" and line_urls:
                    value["urls"] = line_urls
                result[key].append(value)
    return result


def build_education_debug(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
    ner_model: DistilBertNerPredictor,
    url_entities_by_line: dict[int, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Reconstruct education entries from visual rows and grounded entities."""
    routed = _routed_section_headings(lines, headings)
    position = next((i for i, item in enumerate(routed) if item.section_type == "education"), None)
    if position is None:
        return None
    heading = routed[position]
    end = routed[position + 1].line_index if position + 1 < len(routed) else len(lines)
    heading_line = lines[heading.line_index]
    rows = _visual_rows(lines, range(heading.line_index + 1, end))
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def new_entry() -> dict[str, Any]:
        entry: dict[str, Any] = {
            "detectionMethod": "geometry_ner_reconstruction",
            "metadataRows": [], "titles": [], "institutions": [], "dates": [],
            "locations": [], "gpa": [], "skills": [], "urls": [],
            "paragraphs": [], "bullets": [], "_bodyStarted": False,
            "_lastBodyType": None, "_lastLineBbox": None,
        }
        entries.append(entry)
        return entry

    for row in rows:
        if all(
            _PAGE_FOOTER_RE.search(line.text) is not None
            and line.bbox[1] >= document[line.page - 1].rect.height * 0.88
            for _, line in row
        ):
            continue
        explicit_metadata = any(
            _DATE_RANGE_RE.search(line.text)
            or _GPA_RE.search(line.text)
            or _EDUCATION_TITLE_RE.search(line.text)
            or _EDUCATION_INSTITUTION_RE.search(line.text)
            for _, line in row
        )
        metadata_shape = explicit_metadata and any(
            line.bold
            or _EDUCATION_SEPARATOR_RE.search(line.text)
            or len(line.text.split()) <= SETTINGS.section_router.maximum_subheading_words
            for _, line in row
        )
        prose_row = any(
            _EDUCATION_PARAGRAPH_RE.search(line.text)
            or (
                not metadata_shape
                and (
                    len(line.text.split()) > SETTINGS.section_router.maximum_subheading_words
                    or line.text.rstrip().endswith((".", ";", ":"))
                )
            )
            for _, line in row
        )
        continuing_body = bool(
            current is not None
            and current["_bodyStarted"]
            and not explicit_metadata
        )
        if prose_row or continuing_body:
            entities: dict[str, Any] = {
                "titles": [], "institutions": [], "dates": [], "locations": [],
                "gpa": [], "skills": [], "urls": [], "handledLineIndexes": set(),
            }
        else:
            entities = _education_row_entities(
                document,
                row,
                ner_model,
                url_entities_by_line,
            )
        primary = bool(entities["titles"] or entities["institutions"] or entities["dates"])
        metadata = primary or bool(entities["locations"] or entities["gpa"])

        if metadata:
            repeated_anchor = bool(
                current is not None
                and (
                    (entities["institutions"] and current["institutions"])
                    or (entities["dates"] and current["dates"])
                    or (
                        entities["titles"]
                        and current["titles"]
                        and current["institutions"]
                    )
                )
            )
            if current is None or (primary and current["_bodyStarted"]) or repeated_anchor:
                current = new_entry()
            current["metadataRows"].append(_row_value(row))
            for key in ("titles", "institutions", "dates", "locations", "gpa", "urls"):
                current[key].extend(entities[key])
            continue

        if (
            current is not None
            and not current["_bodyStarted"]
            and current["titles"]
            and not current["institutions"]
            and len(row) == 1
            and len(row[0][1].text.split()) <= SETTINGS.section_router.maximum_subheading_words
        ):
            line = row[0][1]
            previous = current["titles"][-1]
            previous["text"] += "\n" + line.text
            previous["bbox"] = [
                round(value, 2)
                for value in (pymupdf.Rect(previous["bbox"]) | pymupdf.Rect(line.bbox))
            ]
            previous["detectionMethod"] = "education_title_geometry_continuation"
            current["metadataRows"].append(_row_value(row))
            continue

        current = current or new_entry()
        current["_bodyStarted"] = True
        for line_index, line in row:
            if line_index in entities["handledLineIndexes"]:
                continue
            bullet = _BULLET_RE.match(line.text)
            line_box = pymupdf.Rect(line.bbox)
            rounded_bbox = [round(value, 2) for value in line.bbox]
            if bullet:
                current["bullets"].append({
                    "text": line.text[bullet.end():].strip(),
                    "page": line.page,
                    "bbox": rounded_bbox,
                    "detectionMethod": "bullet_marker",
                })
                current["_lastBodyType"] = "bullet"
                current["_lastLineBbox"] = rounded_bbox
                continue
            if current["_lastBodyType"] == "bullet" and current["bullets"]:
                previous_box = pymupdf.Rect(current["_lastLineBbox"])
                gap = line_box.y0 - previous_box.y1
                if -2.0 <= gap <= max(previous_box.height, line_box.height) * SETTINGS.section_router.paragraph_gap_multiplier:
                    previous = current["bullets"][-1]
                    previous["text"] += "\n" + line.text
                    previous["bbox"] = [round(value, 2) for value in (pymupdf.Rect(previous["bbox"]) | line_box)]
                    current["_lastLineBbox"] = rounded_bbox
                    continue
            if current["paragraphs"] and current["paragraphs"][-1]["page"] == line.page:
                previous = current["paragraphs"][-1]
                previous_box = pymupdf.Rect(previous["bbox"])
                gap = line_box.y0 - previous_box.y1
                horizontal_overlap = max(
                    0.0,
                    min(previous_box.x1, line_box.x1) - max(previous_box.x0, line_box.x0),
                )
                if (
                    -2.0 <= gap <= max(previous_box.height, line_box.height) * SETTINGS.section_router.paragraph_gap_multiplier
                    and horizontal_overlap > 0
                ):
                    previous["text"] += "\n" + line.text
                    previous["bbox"] = [round(value, 2) for value in (previous_box | line_box)]
                    current["_lastBodyType"] = "paragraph"
                    current["_lastLineBbox"] = rounded_bbox
                    continue
            current["paragraphs"].append({
                "text": line.text,
                "page": line.page,
                "bbox": rounded_bbox,
                "detectionMethod": "geometry_default",
            })
            current["_lastBodyType"] = "paragraph"
            current["_lastLineBbox"] = rounded_bbox

    for entry in entries:
        entry.pop("_bodyStarted", None)
        entry.pop("_lastBodyType", None)
        entry.pop("_lastLineBbox", None)
    next_heading = routed[position + 1] if position + 1 < len(routed) else None
    return {
        "sectionType": "education",
        "heading": {
            "text": heading_line.text,
            "page": heading_line.page,
            "bbox": [round(value, 2) for value in heading_line.bbox],
            "similarity": round(heading.similarity, 4),
            "detectionMethod": "geometry_semantic",
        },
        "entries": entries,
        "stoppedAtSection": (
            {"sectionType": next_heading.section_type, "text": lines[next_heading.line_index].text}
            if next_heading else None
        ),
    }


def _skill_inline_parts(
    text: str,
    source_start: int = 0,
    *,
    allow_single_dash_body: bool = True,
) -> tuple[str, int, int, str, int, int, str] | None:
    """Return a short group label and untouched body from a delimiter row."""
    for pattern, method in (
        (_SKILL_INLINE_COLON_RE, "delimiter_colon"),
        (_SKILL_INLINE_TAB_RE, "delimiter_tab"),
        (_SKILL_INLINE_DASH_RE, "delimiter_dash"),
    ):
        match = pattern.match(text)
        if match is None:
            continue
        label = match.group("label").strip()
        body = match.group("body").strip()
        if (
            not label
            or not body
            or len(label.split()) > 7
            or len(label) > 50
            or label.casefold().startswith(("http", "www."))
        ):
            continue
        if (
            method == "delimiter_dash"
            and not allow_single_dash_body
            and "," not in body
            and ";" not in body
        ):
            continue
        label_offset = match.start("label") + len(match.group("label")) - len(match.group("label").lstrip())
        body_offset = match.start("body") + len(match.group("body")) - len(match.group("body").lstrip())
        label_start = source_start + label_offset
        body_start = source_start + body_offset
        return (
            label,
            label_start,
            label_start + len(label),
            body,
            body_start,
            body_start + len(body),
            method,
        )
    return None


def _skill_subheading_value(
    document: pymupdf.Document,
    line: ExtractedLine,
    text: str,
    start: int,
    end: int,
    detection_method: str,
) -> dict[str, Any]:
    return {
        "text": text,
        "page": line.page,
        "bbox": _span_bbox(document, line, text, start, end),
        "detectionMethod": detection_method,
    }


def _new_skill_group(
    groups: list[dict[str, Any]],
    subheading: dict[str, Any] | None,
) -> dict[str, Any]:
    group: dict[str, Any] = {
        "subheading": subheading,
        "paragraphs": [],
        "bullets": [],
        "urls": [],
        "_lastType": None,
        "_lastLineBbox": None,
    }
    groups.append(group)
    return group


def _append_skill_paragraph(
    group: dict[str, Any],
    *,
    text: str,
    page: int,
    bbox: list[float],
    entities: list[dict[str, Any]],
) -> None:
    current_box = pymupdf.Rect(bbox)
    if group["paragraphs"] and group["paragraphs"][-1]["page"] == page:
        previous = group["paragraphs"][-1]
        previous_box = pymupdf.Rect(previous["bbox"])
        gap = current_box.y0 - previous_box.y1
        horizontal_overlap = max(
            0.0,
            min(previous_box.x1, current_box.x1) - max(previous_box.x0, current_box.x0),
        )
        if (
            -2.0 <= gap
            <= max(previous_box.height, current_box.height)
            * SETTINGS.section_router.paragraph_gap_multiplier
            and horizontal_overlap > 0
        ):
            previous["text"] += "\n" + text
            previous["bbox"] = [round(value, 2) for value in (previous_box | current_box)]
            if entities:
                previous.setdefault("entities", []).extend(entities)
            group["_lastType"] = "paragraph"
            group["_lastLineBbox"] = bbox
            return
    value: dict[str, Any] = {
        "text": text,
        "page": page,
        "bbox": bbox,
        "detectionMethod": "geometry_default",
    }
    if entities:
        value["entities"] = entities
    group["paragraphs"].append(value)
    group["_lastType"] = "paragraph"
    group["_lastLineBbox"] = bbox


def build_skills_debug(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
    url_entities_by_line: dict[int, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Preserve skill prose and attach vertical bullets to geometric groups."""
    routed = _routed_section_headings(lines, headings)
    position = next((i for i, item in enumerate(routed) if item.section_type == "skills"), None)
    if position is None:
        return None
    heading = routed[position]
    end = routed[position + 1].line_index if position + 1 < len(routed) else len(lines)
    heading_line = lines[heading.line_index]
    line_range = range(heading.line_index + 1, end)
    rows = _visual_rows(lines, line_range)
    body_size, body_bold = _section_body_style([lines[index] for index in line_range])
    groups: list[dict[str, Any]] = []
    routed_rows: list[list[tuple[int, ExtractedLine]]] = []
    current: dict[str, Any] | None = None

    for row in rows:
        if all(
            _PAGE_FOOTER_RE.search(line.text) is not None
            and line.bbox[1] >= document[line.page - 1].rect.height * 0.88
            for _, line in row
        ):
            continue
        routed_rows.append(row)

        # A short left cell paired with right-side content is a table-like group row.
        if len(row) >= 2:
            left_index, left = row[0]
            right_cells = row[1:]
            left_is_label = (
                len(left.text.split()) <= 7
                and len(left.text) <= 50
                and (
                    left.bold
                    or _looks_like_subheading(left, body_size=body_size, body_bold=body_bold)
                    or pymupdf.Rect(right_cells[0][1].bbox).x0
                    > pymupdf.Rect(left.bbox).x1 + left.size
                )
            )
            if left_is_label:
                stripped = left.text.strip().rstrip(":-–—").strip()
                start = left.text.find(stripped)
                current = _new_skill_group(
                    groups,
                    _skill_subheading_value(
                        document,
                        left,
                        stripped,
                        start,
                        start + len(stripped),
                        "geometry_table_cell",
                    ),
                )
                current["urls"].extend(url_entities_by_line.get(left_index, []))
                for line_index, line in right_cells:
                    line_urls = url_entities_by_line.get(line_index, [])
                    _append_skill_paragraph(
                        current,
                        text=line.text,
                        page=line.page,
                        bbox=[round(value, 2) for value in line.bbox],
                        entities=line_urls,
                    )
                    current["urls"].extend(line_urls)
                continue

        handled_row = False
        for line_index, line in row:
            line_urls = url_entities_by_line.get(line_index, [])
            bullet_match = _BULLET_RE.match(line.text)
            content_start = bullet_match.end() if bullet_match else 0
            content = line.text[content_start:].strip()
            leading_space = len(line.text[content_start:]) - len(line.text[content_start:].lstrip())
            content_start += leading_space
            inline = _skill_inline_parts(
                content,
                content_start,
                allow_single_dash_body=bullet_match is None,
            )
            if inline is not None:
                label, label_start, label_end, body, body_start, body_end, method = inline
                current = _new_skill_group(
                    groups,
                    _skill_subheading_value(
                        document,
                        line,
                        label,
                        label_start,
                        label_end,
                        method,
                    ),
                )
                body_value: dict[str, Any] = {
                    "text": body,
                    "page": line.page,
                    "bbox": _span_bbox(document, line, body, body_start, body_end),
                    "detectionMethod": "bullet_marker" if bullet_match else "geometry_default",
                }
                if line_urls:
                    body_value["entities"] = line_urls
                target = "bullets" if bullet_match else "paragraphs"
                current[target].append(body_value)
                current["urls"].extend(line_urls)
                current["_lastType"] = "bullet" if bullet_match else "paragraph"
                current["_lastLineBbox"] = [round(value, 2) for value in line.bbox]
                handled_row = True
                continue

            standalone_subheading = (
                not bullet_match
                and len(row) == 1
                and (
                    (
                        line.text.strip().endswith((":", "-", "–", "—"))
                        and len(line.text.split()) <= 7
                    )
                    or _looks_like_subheading(
                        line,
                        body_size=body_size,
                        body_bold=body_bold,
                    )
                )
            )
            if standalone_subheading:
                stripped = line.text.strip().rstrip(":-–—").strip()
                start = line.text.find(stripped)
                current = _new_skill_group(
                    groups,
                    _skill_subheading_value(
                        document,
                        line,
                        stripped,
                        start,
                        start + len(stripped),
                        "geometry_typography",
                    ),
                )
                current["urls"].extend(line_urls)
                handled_row = True
                continue

            current = current or _new_skill_group(groups, None)
            rounded_bbox = [round(value, 2) for value in line.bbox]
            if bullet_match:
                value: dict[str, Any] = {
                    "text": content,
                    "page": line.page,
                    "bbox": rounded_bbox,
                    "detectionMethod": "bullet_marker",
                }
                if line_urls:
                    value["entities"] = line_urls
                current["bullets"].append(value)
                current["urls"].extend(line_urls)
                current["_lastType"] = "bullet"
                current["_lastLineBbox"] = rounded_bbox
            elif current["_lastType"] == "bullet" and current["_lastLineBbox"] is not None:
                previous_box = pymupdf.Rect(current["_lastLineBbox"])
                line_box = pymupdf.Rect(line.bbox)
                gap = line_box.y0 - previous_box.y1
                if (
                    -2.0 <= gap
                    <= max(previous_box.height, line_box.height)
                    * SETTINGS.section_router.paragraph_gap_multiplier
                ):
                    previous = current["bullets"][-1]
                    previous["text"] += "\n" + line.text
                    previous["bbox"] = [
                        round(value, 2)
                        for value in (pymupdf.Rect(previous["bbox"]) | line_box)
                    ]
                    if line_urls:
                        previous.setdefault("entities", []).extend(line_urls)
                    current["urls"].extend(line_urls)
                    current["_lastLineBbox"] = rounded_bbox
                else:
                    _append_skill_paragraph(
                        current,
                        text=line.text,
                        page=line.page,
                        bbox=rounded_bbox,
                        entities=line_urls,
                    )
                    current["urls"].extend(line_urls)
            else:
                _append_skill_paragraph(
                    current,
                    text=line.text,
                    page=line.page,
                    bbox=rounded_bbox,
                    entities=line_urls,
                )
                current["urls"].extend(line_urls)
            handled_row = True
        if not handled_row:
            continue

    for group in groups:
        group.pop("_lastType", None)
        group.pop("_lastLineBbox", None)
    next_heading = routed[position + 1] if position + 1 < len(routed) else None
    return {
        "sectionType": "skills",
        "heading": {
            "text": heading_line.text,
            "page": heading_line.page,
            "bbox": [round(value, 2) for value in heading_line.bbox],
            "similarity": round(heading.similarity, 4),
            "detectionMethod": "geometry_semantic",
        },
        "rows": [_row_value(row) for row in routed_rows],
        "groups": groups,
        "stoppedAtSection": (
            {"sectionType": next_heading.section_type, "text": lines[next_heading.line_index].text}
            if next_heading else None
        ),
    }


def _grouped_section_date_matches(text: str) -> list[re.Match[str]]:
    ranges = list(_DATE_RANGE_RE.finditer(text))
    singles = [
        match
        for match in _SINGLE_YEAR_RE.finditer(text)
        if not any(
            date_range.start() < match.end() and match.start() < date_range.end()
            for date_range in ranges
        )
    ]
    return sorted([*ranges, *singles], key=lambda match: match.start())


def _grouped_section_entry() -> dict[str, Any]:
    return {
        "detectionMethod": "geometry_reconstruction",
        "subheadingLines": [],
        "metadataRows": [],
        "dates": [],
        "urls": [],
        "attributes": [],
        "paragraphs": [],
        "bullets": [],
        "_lastType": None,
        "_lastLineBbox": None,
    }


def _build_grouped_section_debug(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
    url_entities_by_line: dict[int, list[dict[str, Any]]],
    section_type: str,
    occurrence: int = 0,
    semantic_model: EmbeddingModel | None = None,
) -> dict[str, Any] | None:
    """Group titles, dates, paragraphs, bullets, and URLs for a minor section."""
    routed = _routed_section_headings(lines, headings)
    positions = [
        index
        for index, item in enumerate(routed)
        if item.section_type == section_type
    ]
    if occurrence >= len(positions):
        return None
    position = positions[occurrence]
    heading = routed[position]
    end = routed[position + 1].line_index if position + 1 < len(routed) else len(lines)
    heading_line = lines[heading.line_index]
    line_range = range(heading.line_index + 1, end)
    rows = _visual_rows(lines, line_range)
    body_size, body_bold = _section_body_style([lines[index] for index in line_range])
    titled_section = section_type in {
        "projects",
        "certifications",
        "licenses",
        "volunteering",
        "awards",
        "publications",
        "references",
    }
    entries: list[dict[str, Any]] = []
    visible_rows: list[list[tuple[int, ExtractedLine]]] = []
    current: dict[str, Any] | None = None

    for row in rows:
        if all(
            _PAGE_FOOTER_RE.search(line.text) is not None
            and line.bbox[1] >= document[line.page - 1].rect.height * 0.88
            for _, line in row
        ):
            continue
        visible_rows.append(row)

        # Unknown sections frequently contain labelled groups such as
        # ``Availability: ...`` or a short left label paired with a value on
        # the right. Detect those groups before the generic paragraph fallback,
        # using the same precedence as the skills parser.
        if section_type == "others" and len(row) >= 2:
            left_index, left = row[0]
            right_cells = row[1:]
            left_is_label = (
                len(left.text.split()) <= 7
                and len(left.text) <= 50
                and (
                    left.bold
                    or _looks_like_subheading(
                        left,
                        body_size=body_size,
                        body_bold=body_bold,
                    )
                    or pymupdf.Rect(right_cells[0][1].bbox).x0
                    > pymupdf.Rect(left.bbox).x1 + left.size
                )
            )
            if left_is_label:
                stripped = left.text.strip().rstrip(":-–—").strip()
                attribute_type: str | None = None
                confidence = 0.0
                if semantic_model is not None:
                    attribute_type, confidence = classify_profile_attribute_labels(
                        semantic_model,
                        [stripped],
                    )[0]
                start = left.text.find(stripped)
                current = _grouped_section_entry()
                entries.append(current)
                current["subheadingLines"].append(
                    _skill_subheading_value(
                        document,
                        left,
                        stripped,
                        start,
                        start + len(stripped),
                        "geometry_table_cell",
                    )
                )
                left_urls = url_entities_by_line.get(left_index, [])
                current["urls"].extend(left_urls)
                current["metadataRows"].append(_row_value(row))
                for line_index, line in right_cells:
                    line_urls = url_entities_by_line.get(line_index, [])
                    current["urls"].extend(line_urls)
                    if attribute_type is not None:
                        current["attributes"].append({
                            "type": attribute_type,
                            "text": line.text,
                            "page": line.page,
                            "bbox": [round(number, 2) for number in line.bbox],
                            "confidence": round(confidence, 4),
                            "detectionMethod": (
                                "label_pattern"
                                if confidence >= 0.9999
                                else "minilm_label"
                            ),
                        })
                    else:
                        _append_skill_paragraph(
                            current,
                            text=line.text,
                            page=line.page,
                            bbox=[round(number, 2) for number in line.bbox],
                            entities=line_urls,
                        )
                continue

        if section_type == "others":
            inline_parts: list[
                tuple[
                    int,
                    ExtractedLine,
                    re.Match[str] | None,
                    tuple[str, int, int, str, int, int, str],
                ]
            ] = []
            for line_index, line in row:
                bullet = _BULLET_RE.match(line.text)
                content_start = bullet.end() if bullet else 0
                leading_space = len(line.text[content_start:]) - len(
                    line.text[content_start:].lstrip()
                )
                content_start += leading_space
                content = line.text[content_start:].strip()
                attribute_inline = _PROFILE_ATTRIBUTE_INLINE_RE.match(content)
                inline = (
                    (
                        attribute_inline.group("label").strip(),
                        content_start + attribute_inline.start("label"),
                        content_start + attribute_inline.end("label"),
                        attribute_inline.group("body").strip(),
                        content_start + attribute_inline.start("body"),
                        content_start + attribute_inline.end("body"),
                        "profile_attribute_delimiter",
                    )
                    if attribute_inline is not None
                    else _skill_inline_parts(
                        content,
                        content_start,
                        allow_single_dash_body=bullet is None,
                    )
                )
                if inline is not None:
                    inline_parts.append((line_index, line, bullet, inline))

            if inline_parts:
                for line_index, line, bullet, inline in inline_parts:
                    (
                        label,
                        label_start,
                        label_end,
                        body,
                        body_start,
                        body_end,
                        method,
                    ) = inline
                    current = _grouped_section_entry()
                    entries.append(current)
                    current["subheadingLines"].append(
                        _skill_subheading_value(
                            document,
                            line,
                            label,
                            label_start,
                            label_end,
                            method,
                        )
                    )
                    current["metadataRows"].append(_row_value(row))
                    line_urls = url_entities_by_line.get(line_index, [])
                    body_value: dict[str, Any] = {
                        "text": body,
                        "page": line.page,
                        "bbox": _span_bbox(
                            document,
                            line,
                            body,
                            body_start,
                            body_end,
                        ),
                        "detectionMethod": (
                            "bullet_marker" if bullet else "geometry_default"
                        ),
                    }
                    if line_urls:
                        body_value["entities"] = line_urls
                    current["urls"].extend(line_urls)
                    attribute_type: str | None = None
                    confidence = 0.0
                    if semantic_model is not None:
                        attribute_type, confidence = classify_profile_attribute_labels(
                            semantic_model,
                            [label],
                        )[0]
                        if attribute_type is not None:
                            current["attributes"].append({
                                "type": attribute_type,
                                "text": body,
                                "page": line.page,
                                "bbox": body_value["bbox"],
                                "confidence": round(confidence, 4),
                                "detectionMethod": (
                                    "label_pattern"
                                    if confidence >= 0.9999
                                    else "minilm_label"
                                ),
                            })
                    if attribute_type is None:
                        target = "bullets" if bullet else "paragraphs"
                        current[target].append(body_value)
                        current["_lastType"] = "bullet" if bullet else "paragraph"
                    current["_lastLineBbox"] = [
                        round(number, 2) for number in line.bbox
                    ]
                continue

        dates_by_line = {
            line_index: _grouped_section_date_matches(line.text)
            for line_index, line in row
        }
        row_has_date = any(dates_by_line.values())
        title_cells: list[tuple[int, ExtractedLine]] = []
        for line_index, line in row:
            bullet = _BULLET_RE.match(line.text)
            matches = dates_by_line[line_index]
            date_only = bool(matches) and all(
                not character.strip(" ()[]{}|,;:-–—")
                for character in (
                    line.text[:matches[0].start()],
                    line.text[matches[-1].end():],
                )
            )
            if bullet or date_only:
                continue
            typographic_title = _looks_like_subheading(
                line,
                body_size=body_size,
                body_bold=body_bold,
            )
            row_title = row_has_date and len(row) >= 2
            first_title = (
                titled_section
                and
                current is None
                and len(line.text.split()) <= SETTINGS.section_router.maximum_subheading_words
                and not line.text.rstrip().endswith((".", ";", ":"))
            )
            if typographic_title or row_title or first_title:
                title_cells.append((line_index, line))

        if title_cells:
            should_continue_title = bool(
                current is not None
                and current["subheadingLines"]
                and not current["dates"]
                and not current["paragraphs"]
                and not current["bullets"]
                and title_cells[0][1].page == current["subheadingLines"][-1]["page"]
                and pymupdf.Rect(title_cells[0][1].bbox).y0
                - pymupdf.Rect(current["subheadingLines"][-1]["bbox"]).y1
                <= max(title_cells[0][1].size, 1.0) * 0.65
            )
            if not should_continue_title:
                current = _grouped_section_entry()
                entries.append(current)
            for line_index, line in title_cells:
                value: dict[str, Any] = {
                    "text": line.text,
                    "page": line.page,
                    "bbox": [round(number, 2) for number in line.bbox],
                    "detectionMethod": "geometry_typography",
                }
                line_urls = url_entities_by_line.get(line_index, [])
                if line_urls:
                    value["entities"] = line_urls
                    current["urls"].extend(line_urls)
                current["subheadingLines"].append(value)
            current["metadataRows"].append(_row_value(row))

        if row_has_date:
            if current is None:
                current = _grouped_section_entry()
                entries.append(current)
            for line_index, line in row:
                for match in dates_by_line[line_index]:
                    current["dates"].append({
                        "text": match.group(0),
                        "page": line.page,
                        "bbox": _span_bbox(
                            document,
                            line,
                            match.group(0),
                            match.start(),
                            match.end(),
                        ),
                        "detectionMethod": (
                            "date_regex" if _DATE_RANGE_RE.fullmatch(match.group(0)) else "year_regex"
                        ),
                    })
            if not title_cells:
                current["metadataRows"].append(_row_value(row))

        title_indexes = {line_index for line_index, _ in title_cells}
        for line_index, line in row:
            if line_index in title_indexes:
                continue
            matches = dates_by_line[line_index]
            if matches:
                residual = line.text
                for match in reversed(matches):
                    residual = residual[:match.start()] + residual[match.end():]
                if not residual.strip(" ()[]{}|,;:-–—"):
                    continue
            if current is None:
                current = _grouped_section_entry()
                entries.append(current)
            line_urls = url_entities_by_line.get(line_index, [])
            rounded_bbox = [round(number, 2) for number in line.bbox]
            bullet = _BULLET_RE.match(line.text)
            if bullet:
                value: dict[str, Any] = {
                    "text": line.text[bullet.end():].strip(),
                    "page": line.page,
                    "bbox": rounded_bbox,
                    "detectionMethod": "bullet_marker",
                }
                if line_urls:
                    value["entities"] = line_urls
                current["bullets"].append(value)
                current["urls"].extend(line_urls)
                current["_lastType"] = "bullet"
                current["_lastLineBbox"] = rounded_bbox
                continue
            if current["_lastType"] == "bullet" and current["_lastLineBbox"] is not None:
                previous_box = pymupdf.Rect(current["_lastLineBbox"])
                line_box = pymupdf.Rect(line.bbox)
                gap = line_box.y0 - previous_box.y1
                if (
                    -2.0 <= gap
                    <= max(previous_box.height, line_box.height)
                    * SETTINGS.section_router.paragraph_gap_multiplier
                ):
                    previous = current["bullets"][-1]
                    previous["text"] += "\n" + line.text
                    previous["bbox"] = [
                        round(number, 2)
                        for number in (pymupdf.Rect(previous["bbox"]) | line_box)
                    ]
                    if line_urls:
                        previous.setdefault("entities", []).extend(line_urls)
                    current["urls"].extend(line_urls)
                    current["_lastLineBbox"] = rounded_bbox
                    continue
            _append_skill_paragraph(
                current,
                text=line.text,
                page=line.page,
                bbox=rounded_bbox,
                entities=line_urls,
            )
            current["urls"].extend(line_urls)

    for entry in entries:
        entry.pop("_lastType", None)
        entry.pop("_lastLineBbox", None)
    next_heading = routed[position + 1] if position + 1 < len(routed) else None
    return {
        "sectionType": section_type,
        "heading": {
            "text": heading_line.text,
            "page": heading_line.page,
            "bbox": [round(value, 2) for value in heading_line.bbox],
            "similarity": round(heading.similarity, 4),
            "detectionMethod": (
                "geometry_unknown_boundary"
                if heading.line_index not in {item.line_index for item in headings}
                else "geometry_semantic"
            ),
        },
        "rows": [_row_value(row) for row in visible_rows],
        "entries": entries,
        "stoppedAtSection": (
            {"sectionType": next_heading.section_type, "text": lines[next_heading.line_index].text}
            if next_heading else None
        ),
    }


def build_projects_debug(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
    url_entities_by_line: dict[int, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    return _build_grouped_section_debug(
        document,
        lines,
        headings,
        url_entities_by_line,
        "projects",
    )


def build_supplementary_sections_debug(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
    url_entities_by_line: dict[int, list[dict[str, Any]]],
    semantic_model: EmbeddingModel,
) -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    for section_type in _SUPPLEMENTARY_SECTION_TYPES:
        if section_type == "others":
            other_sections: list[dict[str, Any]] = []
            occurrence = 0
            while True:
                section = _build_grouped_section_debug(
                    document,
                    lines,
                    headings,
                    url_entities_by_line,
                    section_type,
                    occurrence,
                    semantic_model,
                )
                if section is None:
                    break
                other_sections.append(section)
                occurrence += 1
            if other_sections:
                sections[section_type] = {
                    "sectionType": "others",
                    "sections": other_sections,
                }
            continue
        section = _build_grouped_section_debug(
            document,
            lines,
            headings,
            url_entities_by_line,
            section_type,
        )
        if section is not None:
            sections[section_type] = section
    return sections


def summary_debug_value(
    sections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not sections or sections[0]["sectionType"] != "summary":
        return None
    summary = dict(sections[0])
    summary["stoppedAtSection"] = (
        {
            "sectionType": sections[1]["sectionType"],
            "heading": sections[1]["heading"],
        }
        if len(sections) > 1
        else None
    )
    return summary


def write_summary_debug(
    pdf_path: Path,
    summary: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if summary is None:
        return
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").unlink(missing_ok=True)
    (output_directory / "summary-raw.json").write_text(
        json.dumps(
            {
                "source": pdf_path.name,
                "model": SETTINGS.model.name,
                "modelRevision": SETTINGS.model.revision,
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _pixel_box(bbox: list[float] | tuple[float, ...]) -> tuple[int, int, int, int]:
    return tuple(
        round(value * SETTINGS.debug.scale) for value in bbox
    )  # type: ignore[return-value]


def render_summary_debug_images(
    document: pymupdf.Document,
    summary: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if summary is None:
        return
    output_directory.mkdir(parents=True, exist_ok=True)
    items = [
        {
            "type": "section_heading",
            "page": summary["heading"]["page"],
            "bbox": summary["heading"]["bbox"],
        },
        *summary["content"],
    ]
    items_by_page: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        items_by_page.setdefault(int(item["page"]), []).append(item)

    colors = {
        "section_heading": SETTINGS.section_colors["summary"],
        "subheading": "#EF6C00",
        "paragraph": "#546E7A",
        "bullet": "#5C6BC0",
    }
    matrix = pymupdf.Matrix(SETTINGS.debug.scale, SETTINGS.debug.scale)
    for page_number, page_items in sorted(items_by_page.items()):
        page = document[page_number - 1]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        draw = ImageDraw.Draw(image)
        for item in page_items:
            item_type = str(item["type"])
            color = colors[item_type]
            item_box = _pixel_box(item["bbox"])
            draw.rectangle(
                item_box,
                outline=color,
                width=(
                    SETTINGS.debug.heading_stroke_width
                    if item_type == "section_heading"
                    else SETTINGS.debug.content_stroke_width
                ),
            )
            draw.text(
                (
                    item_box[0] + SETTINGS.debug.label_x_padding,
                    max(0, item_box[1] - SETTINGS.debug.label_y_offset),
                ),
                item_type,
                fill=color,
            )
        image.save(output_directory / f"page-{page_number}.png")


def write_experience_debug(
    pdf_path: Path,
    experience: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if experience is None:
        return
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "experience.json").unlink(missing_ok=True)
    (output_directory / "experience-raw.json").write_text(
        json.dumps({"source": pdf_path.name, "experience": experience}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _render_entry_debug_images(
    document: pymupdf.Document,
    items: list[dict[str, Any]],
    output_directory: Path,
    colors: dict[str, str],
    labels: dict[str, str],
    metadata_types: set[str],
) -> None:
    items_by_page: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        items_by_page.setdefault(int(item["page"]), []).append(item)
    output_directory.mkdir(parents=True, exist_ok=True)
    matrix = pymupdf.Matrix(SETTINGS.debug.scale, SETTINGS.debug.scale)
    for page_number, page_items in sorted(items_by_page.items()):
        page = document[page_number - 1]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        draw = ImageDraw.Draw(image)
        page_item_boxes = [_pixel_box(item["bbox"]) for item in page_items]
        placed_label_boxes: list[tuple[int, int, int, int]] = []
        for item_index, item in enumerate(page_items):
            item_type = item["type"]
            box = _pixel_box(item["bbox"])
            color = str(item.get("_debugColor") or colors[item_type])
            detection_method = str(item.get("detectionMethod", ""))
            model_entity = detection_method.startswith(("distilbert", "minilm"))
            draw.rectangle(
                box,
                outline=color,
                width=(
                    SETTINGS.debug.header_entity_stroke_width + 2
                    if model_entity
                    else (
                        SETTINGS.debug.heading_stroke_width
                        if item_type == "section_heading"
                        else 2
                    )
                ),
            )
            label = str(item.get("_debugLabel") or labels[item_type])
            measured_label_box = draw.textbbox((0, 0), label)
            label_width = measured_label_box[2] - measured_label_box[0]
            label_height = measured_label_box[3] - measured_label_box[1]
            if item_type in metadata_types:
                label_position = (
                    min(
                        image.width - label_width,
                        box[2] + SETTINGS.debug.label_x_padding,
                    ),
                    max(0, (box[1] + box[3] - label_height) // 2),
                )
            elif item_type in {"date", "location"}:
                label_position = (
                    max(0, min(image.width - label_width, box[2] - label_width)),
                    min(image.height - label_height, box[3] + 2),
                )
            else:
                label_position = (
                    box[0] + SETTINGS.debug.label_x_padding,
                    max(0, box[1] - SETTINGS.debug.label_y_offset),
                )
            label_box = draw.textbbox(label_position, label)
            collides = any(
                label_box[0] < other_box[2]
                and other_box[0] < label_box[2]
                and label_box[1] < other_box[3]
                and other_box[1] < label_box[3]
                for other_index, other_box in enumerate(page_item_boxes)
                if other_index != item_index
            ) or any(
                label_box[0] < other_box[2]
                and other_box[0] < label_box[2]
                and label_box[1] < other_box[3]
                and other_box[1] < label_box[3]
                for other_box in placed_label_boxes
            )
            if collides:
                label_position = (
                    max(
                        0,
                        box[0] - label_width - SETTINGS.debug.label_x_padding,
                    ),
                    max(
                        0,
                        min(
                            image.height - label_height,
                            (box[1] + box[3] - label_height) // 2,
                        ),
                    ),
                )
                label_box = draw.textbbox(label_position, label)
            draw.rectangle(label_box, fill="#FFFFFF")
            draw.text(label_position, label, fill=color)
            placed_label_boxes.append(label_box)
        image.save(output_directory / f"page-{page_number}.png")


def render_combined_debug_images(
    document: pymupdf.Document,
    header_profile: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    experience: dict[str, Any] | None,
    education: dict[str, Any] | None,
    skills: dict[str, Any] | None,
    projects: dict[str, Any] | None,
    supplementary_sections: dict[str, dict[str, Any]],
    output_directory: Path,
) -> None:
    """Render all implemented section overlays together on each source page."""
    items: list[dict[str, Any]] = []

    def add(
        item_type: str,
        value: dict[str, Any],
        color: str,
        label: str,
    ) -> None:
        items.append(
            {
                **value,
                "type": item_type,
                "_debugColor": color,
                "_debugLabel": label,
            }
        )

    if header_profile is not None:
        add(
            "header_profile",
            {
                "page": header_profile["page"],
                "bbox": header_profile["bbox"],
                "detectionMethod": "geometry_header_region",
            },
            SETTINGS.debug.header_region_color,
            "header_profile",
        )
        header_colors = dict(SETTINGS.debug.header_entity_colors)
        for entity in header_profile["entities"]:
            add(
                str(entity["type"]),
                entity,
                header_colors[str(entity["type"])],
                str(entity["type"]),
            )

    if summary is not None:
        add(
            "section_heading",
            summary["heading"],
            SETTINGS.section_colors["summary"],
            "section_heading",
        )
        for value in summary["content"]:
            item_type = str(value["type"])
            content_color = {
                "subheading": "#EF6C00",
                "paragraph": "#546E7A",
                "bullet": "#5C6BC0",
            }[item_type]
            add(
                item_type,
                value,
                content_color,
                f"{item_type}",
            )

    if experience is not None:
        add(
            "section_heading",
            experience["heading"],
            SETTINGS.section_colors["experience"],
            "section_heading",
        )
        experience_specs = {
            "experience_subheading": ("#B0BEC5", "metadata_line"),
            "job_title": ("#00897B", "MiniLM: job_title"),
            "company": ("#D81B60", "NER: company"),
            "date": ("#F9A825", "regex: date"),
            "location": ("#1E88E5", "NER: location"),
            "url": ("#6D4C41", "annotation: url"),
            "paragraph": ("#CFD8DC", "paragraph"),
            "bullet": ("#B0BEC5", "bullet"),
        }
        for entry in experience["entries"]:
            for value in entry["subheadingLines"]:
                color, label = experience_specs["experience_subheading"]
                add("experience_subheading", value, color, label)
            for key, item_type in (
                ("jobTitles", "job_title"),
                ("companies", "company"),
                ("dates", "date"),
                ("locations", "location"),
                ("urls", "url"),
                ("paragraphs", "paragraph"),
                ("bullets", "bullet"),
            ):
                color, label = experience_specs[item_type]
                for value in entry[key]:
                    item_label = (
                        "annotation: company"
                        if (
                        item_type == "company"
                        and value.get("detectionMethod") == "url_company_reconciled"
                        )
                        else label
                    )
                    add(item_type, value, color, item_label)

    if education is not None:
        add(
            "section_heading",
            education["heading"],
            SETTINGS.section_colors["education"],
            "section_heading",
        )
        education_specs = {
            "education_metadata": ("#D1C4E9", "geometry_row"),
            "education_title": ("#00897B", "pattern: education_title"),
            "institution": ("#D81B60", "entity: institution"),
            "date": ("#F9A825", "regex: date"),
            "location": ("#1E88E5", "NER: location"),
            "gpa": ("#8E24AA", "regex: GPA"),
            "skill": ("#43A047", "skill"),
            "url": ("#6D4C41", "annotation: url"),
            "paragraph": ("#CFD8DC", "paragraph"),
            "bullet": ("#B0BEC5", "bullet"),
        }
        for entry in education["entries"]:
            for value in entry["metadataRows"]:
                color, label = education_specs["education_metadata"]
                add("education_metadata", value, color, label)
            for key, item_type in (
                ("titles", "education_title"),
                ("institutions", "institution"),
                ("dates", "date"),
                ("locations", "location"),
                ("gpa", "gpa"),
                ("skills", "skill"),
                ("urls", "url"),
                ("paragraphs", "paragraph"),
                ("bullets", "bullet"),
            ):
                color, label = education_specs[item_type]
                for value in entry[key]:
                    add(item_type, value, color, label)

    if skills is not None:
        add(
            "section_heading",
            skills["heading"],
            SETTINGS.section_colors["skills"],
            "section_heading",
        )
        for value in skills["rows"]:
            add("skill_row", value, "#CFD8DC", "geometry_row")
        for group in skills["groups"]:
            if group["subheading"] is not None:
                add(
                    "skill_subheading",
                    group["subheading"],
                    "#00897B",
                    "skill_group",
                )
            for key, item_type, color in (
                ("paragraphs", "paragraph", "#90A4AE"),
                ("bullets", "bullet", "#5C6BC0"),
                ("urls", "url", "#6D4C41"),
            ):
                for value in group[key]:
                    add(item_type, value, color, f"{item_type}" if item_type != "url" else "annotation: url")

    grouped_sections: list[tuple[str, dict[str, Any]]] = []
    if projects is not None:
        grouped_sections.append(("projects", projects))
    for section_type, section in supplementary_sections.items():
        for section_part in (
            section["sections"] if section_type == "others" else [section]
        ):
            grouped_sections.append((section_type, section_part))

    for section_type, section in grouped_sections:
        section_color = SETTINGS.section_colors[section_type]
        add(
            "section_heading",
            section["heading"],
            section_color,
            "section_heading",
        )
        for value in section["rows"]:
            add("grouped_row", value, "#ECEFF1", "geometry_row")
        for entry in section["entries"]:
            for value in entry["subheadingLines"]:
                add("grouped_subheading", value, section_color, "subheading")
            for value in entry.get("attributes", []):
                attribute_type = str(value["type"])
                add(
                    "profile_attribute",
                    value,
                    dict(SETTINGS.debug.header_entity_colors).get(
                        attribute_type,
                        section_color,
                    ),
                    f"attribute: {attribute_type}",
                )
            for key, item_type, color, label in (
                ("dates", "date", "#F9A825", "regex: date"),
                ("urls", "url", "#6D4C41", "annotation: url"),
                ("paragraphs", "paragraph", "#90A4AE", "paragraph"),
                ("bullets", "bullet", "#5C6BC0", "bullet"),
            ):
                for value in entry[key]:
                    add(item_type, value, color, label)

    if not items:
        return
    item_types = {str(item["type"]) for item in items}
    _render_entry_debug_images(
        document,
        items,
        output_directory,
        {item_type: "#000000" for item_type in item_types},
        {item_type: item_type for item_type in item_types},
        {
            "name",
            "experience_subheading",
            "education_metadata",
            "skill_row",
            "grouped_row",
        },
    )


def render_experience_debug_images(
    document: pymupdf.Document,
    experience: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if experience is None:
        return
    items: list[dict[str, Any]] = [{"type": "section_heading", **experience["heading"]}]
    for entry in experience["entries"]:
        items.extend({"type": "experience_subheading", **item} for item in entry["subheadingLines"])
        for key, item_type in (
            ("jobTitles", "job_title"),
            ("companies", "company"),
            ("dates", "date"),
            ("locations", "location"),
            ("urls", "url"),
        ):
            items.extend(
                {
                    "type": item_type,
                    **item,
                    **(
                        {"_debugLabel": "annotation: company"}
                        if item_type == "company"
                        and item.get("detectionMethod") == "url_company_reconciled"
                        else {}
                    ),
                }
                for item in entry[key]
            )
        items.extend({"type": "paragraph", **item} for item in entry["paragraphs"])
        items.extend({"type": "bullet", **item} for item in entry["bullets"])
    _render_entry_debug_images(
        document,
        items,
        output_directory,
        {
            "section_heading": SETTINGS.section_colors["experience"],
            "experience_subheading": "#B0BEC5",
            "job_title": "#00897B",
            "company": "#D81B60",
            "date": "#F9A825",
            "location": "#1E88E5",
            "url": "#6D4C41",
            "paragraph": "#CFD8DC",
            "bullet": "#B0BEC5",
        },
        {
            "section_heading": "section_heading",
            "experience_subheading": "metadata_line",
            "job_title": "MiniLM: job_title",
            "company": "NER: company",
            "date": "regex: date",
            "location": "NER: location",
            "url": "annotation: url",
            "paragraph": "paragraph",
            "bullet": "bullet",
        },
        {"experience_subheading"},
    )


def write_education_debug(
    pdf_path: Path,
    education: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if education is None:
        return
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "education.json").unlink(missing_ok=True)
    (output_directory / "education-raw.json").write_text(
        json.dumps({"source": pdf_path.name, "education": education}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def render_education_debug_images(
    document: pymupdf.Document,
    education: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if education is None:
        return
    items: list[dict[str, Any]] = [{"type": "section_heading", **education["heading"]}]
    for entry in education["entries"]:
        items.extend({"type": "education_metadata", **item} for item in entry["metadataRows"])
        for key, item_type in (
            ("titles", "education_title"),
            ("institutions", "institution"),
            ("dates", "date"),
            ("locations", "location"),
            ("gpa", "gpa"),
            ("skills", "skill"),
            ("urls", "url"),
        ):
            items.extend({"type": item_type, **item} for item in entry[key])
        items.extend({"type": "paragraph", **item} for item in entry["paragraphs"])
        items.extend({"type": "bullet", **item} for item in entry["bullets"])
    _render_entry_debug_images(
        document,
        items,
        output_directory,
        {
            "section_heading": SETTINGS.section_colors["education"],
            "education_metadata": "#D1C4E9",
            "education_title": "#00897B",
            "institution": "#D81B60",
            "date": "#F9A825",
            "location": "#1E88E5",
            "gpa": "#8E24AA",
            "skill": "#43A047",
            "url": "#6D4C41",
            "paragraph": "#CFD8DC",
            "bullet": "#B0BEC5",
        },
        {
            "section_heading": "section_heading",
            "education_metadata": "geometry_row",
            "education_title": "pattern: education_title",
            "institution": "entity: institution",
            "date": "regex: date",
            "location": "NER: location",
            "gpa": "regex: GPA",
            "skill": "skill",
            "url": "annotation: url",
            "paragraph": "paragraph",
            "bullet": "bullet",
        },
        {"education_metadata"},
    )


def write_skills_debug(
    pdf_path: Path,
    skills: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if skills is None:
        return
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "skills.json").unlink(missing_ok=True)
    (output_directory / "skills-raw.json").write_text(
        json.dumps({"source": pdf_path.name, "skills": skills}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def render_skills_debug_images(
    document: pymupdf.Document,
    skills: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if skills is None:
        return
    items: list[dict[str, Any]] = [{"type": "section_heading", **skills["heading"]}]
    items.extend({"type": "skill_row", **item} for item in skills["rows"])
    for group in skills["groups"]:
        if group["subheading"] is not None:
            items.append({"type": "skill_subheading", **group["subheading"]})
        items.extend({"type": "paragraph", **item} for item in group["paragraphs"])
        items.extend({"type": "bullet", **item} for item in group["bullets"])
        items.extend({"type": "url", **item} for item in group["urls"])
    _render_entry_debug_images(
        document,
        items,
        output_directory,
        {
            "section_heading": SETTINGS.section_colors["skills"],
            "skill_row": "#CFD8DC",
            "skill_subheading": "#00897B",
            "paragraph": "#90A4AE",
            "bullet": "#5C6BC0",
            "url": "#6D4C41",
        },
        {
            "section_heading": "section_heading",
            "skill_row": "geometry_row",
            "skill_subheading": "skill_group",
            "paragraph": "paragraph",
            "bullet": "bullet",
            "url": "annotation: url",
        },
        {"skill_row"},
    )


def write_projects_debug(
    pdf_path: Path,
    projects: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if projects is None:
        return
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "projects.json").unlink(missing_ok=True)
    (output_directory / "projects-raw.json").write_text(
        json.dumps({"source": pdf_path.name, "projects": projects}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def render_projects_debug_images(
    document: pymupdf.Document,
    projects: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    if projects is None:
        return
    items: list[dict[str, Any]] = [{"type": "section_heading", **projects["heading"]}]
    items.extend({"type": "project_row", **item} for item in projects["rows"])
    for entry in projects["entries"]:
        items.extend({"type": "project_subheading", **item} for item in entry["subheadingLines"])
        items.extend({"type": "date", **item} for item in entry["dates"])
        items.extend({"type": "url", **item} for item in entry["urls"])
        items.extend({"type": "paragraph", **item} for item in entry["paragraphs"])
        items.extend({"type": "bullet", **item} for item in entry["bullets"])
    _render_entry_debug_images(
        document,
        items,
        output_directory,
        {
            "section_heading": SETTINGS.section_colors["projects"],
            "project_row": "#FFE0B2",
            "project_subheading": "#EF6C00",
            "date": "#F9A825",
            "url": "#6D4C41",
            "paragraph": "#90A4AE",
            "bullet": "#5C6BC0",
        },
        {
            "section_heading": "section_heading",
            "project_row": "geometry_row",
            "project_subheading": "project_subheading",
            "date": "regex: date",
            "url": "annotation: url",
            "paragraph": "paragraph",
            "bullet": "bullet",
        },
        {"project_row"},
    )


def write_supplementary_sections_debug(
    pdf_path: Path,
    sections: dict[str, dict[str, Any]],
    debug_directory: Path,
) -> None:
    for section_type, section in sections.items():
        output_directory = debug_directory / section_type
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / f"{section_type}.json").unlink(missing_ok=True)
        (output_directory / f"{section_type}-raw.json").write_text(
            json.dumps(
                {"source": pdf_path.name, section_type: section},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


def render_supplementary_sections_debug_images(
    document: pymupdf.Document,
    sections: dict[str, dict[str, Any]],
    debug_directory: Path,
) -> None:
    for section_type, section in sections.items():
        section_color = SETTINGS.section_colors[section_type]
        section_parts = (
            section["sections"]
            if section_type == "others"
            else [section]
        )
        items: list[dict[str, Any]] = []
        for section_part in section_parts:
            items.append({"type": "section_heading", **section_part["heading"]})
            items.extend(
                {"type": "grouped_row", **item}
                for item in section_part["rows"]
            )
            for entry in section_part["entries"]:
                items.extend(
                    {"type": "grouped_subheading", **item}
                    for item in entry["subheadingLines"]
                )
                items.extend({"type": "date", **item} for item in entry["dates"])
                items.extend(
                    {
                        **item,
                        "type": "profile_attribute",
                        "_debugColor": dict(
                            SETTINGS.debug.header_entity_colors
                        ).get(str(item["type"]), section_color),
                        "_debugLabel": f"attribute: {item['type']}",
                    }
                    for item in entry.get("attributes", [])
                )
                items.extend({"type": "url", **item} for item in entry["urls"])
                items.extend(
                    {"type": "paragraph", **item}
                    for item in entry["paragraphs"]
                )
                items.extend({"type": "bullet", **item} for item in entry["bullets"])
        _render_entry_debug_images(
            document,
            items,
            debug_directory / section_type,
            {
                "section_heading": section_color,
                "grouped_row": "#ECEFF1",
                "grouped_subheading": section_color,
                "date": "#F9A825",
                "profile_attribute": section_color,
                "url": "#6D4C41",
                "paragraph": "#90A4AE",
                "bullet": "#5C6BC0",
            },
            {
                "section_heading": "section_heading",
                "grouped_row": "geometry_row",
                "grouped_subheading": "subheading",
                "date": "regex: date",
                "profile_attribute": "attribute",
                "url": "annotation: url",
                "paragraph": "paragraph",
                "bullet": "bullet",
            },
            {"grouped_row"},
        )
