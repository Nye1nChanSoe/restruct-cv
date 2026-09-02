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
)


_BULLET_RE = re.compile(
    r"^\s*(?:[-+*•●▪◦‣]|\d+[.)])"
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
    rf"\b(?:From\s+)?{_DATED_YEAR_PATTERN}"
    rf"(?:\s+-\s+|\s*[\u2013\u2014]\s*|\s+to\s+)"
    rf"(?:{_DATED_YEAR_PATTERN}|Present|Current|Now)\b",
    re.IGNORECASE,
)
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
_EDUCATION_SKILLS_RE = re.compile(
    r"^\s*(?:relevant\s+)?(?:coursework|courses?|subjects?|modules?|skills?|"
    r"technologies|tools)\s*(?::|-|included\b)?\s*(?P<items>.+)$",
    re.IGNORECASE,
)
_PAGE_FOOTER_RE = re.compile(r"\bpage\s+\d+\s*$", re.IGNORECASE)


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
    """Route lines into typographic subheadings or combined paragraph blocks."""
    content_lines = [lines[index] for index in line_indexes]
    body_size, body_bold = _section_body_style(content_lines)
    blocks: list[dict[str, Any]] = []

    for line_index in line_indexes:
        line = lines[line_index]
        line_url_entities = url_entities_by_line.get(line_index, [])
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
    routed_headings = sorted(
        (
            heading
            for heading in headings
            if heading.line_index >= first_boundary.line_index
            and _is_reliable_section_heading(lines[heading.line_index], heading)
        ),
        key=lambda heading: heading.line_index,
    )

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
            "detectionMethod": "geometry_semantic",
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
    title_spans: list[tuple[int, int]] = []
    for (text, start, end), (accepted, confidence) in zip(candidates, classifications, strict=True):
        if accepted:
            title_spans.append((start, end))
            result["jobTitles"].append({
                "text": text, "page": line.page,
                "bbox": _span_bbox(document, line, text, start, end),
                "confidence": round(confidence, 4),
                "detectionMethod": "minilm_reconciled",
            })

    company_markers = re.compile(
        r"\b(?:co\.?|company|ltd\.?|limited|inc\.?|corp\.?|corporation|llc|plc)\b",
        re.IGNORECASE,
    )
    for segment, start, end in candidates:
        if any(left < end and start < right for left, right in title_spans):
            continue
        predictions = ner_model.predict_entities(
            segment,
            ["organization", "location"],
            SETTINGS.ner.minimum_confidence,
        )
        organizations = [item for item in predictions if item["label"] == "organization"]
        locations = [item for item in predictions if item["label"] == "location"]
        if organizations and company_markers.search(segment):
            value: dict[str, Any] = {
                "text": segment,
                "page": line.page,
                "bbox": _span_bbox(document, line, segment, start, end),
                "confidence": round(max(float(item["score"]) for item in organizations), 4),
                "detectionMethod": "distilbert_ner_reconciled",
            }
            if urls:
                value["urls"] = urls
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
        for prediction in predictions:
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
            if key == "companies" and urls:
                value["urls"] = urls
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
    routed = sorted(
        (
            heading for heading in headings
            if _is_reliable_section_heading(lines[heading.line_index], heading)
        ),
        key=lambda item: item.line_index,
    )
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


def _education_skills(
    document: pymupdf.Document,
    line: ExtractedLine,
) -> list[dict[str, Any]]:
    match = _EDUCATION_SKILLS_RE.match(line.text)
    if match is None:
        return []
    raw_items = match.group("items").strip()
    if "," not in raw_items and ";" not in raw_items:
        return []
    return _education_skill_items(
        document,
        line,
        raw_items,
        match.start("items"),
    )


def _education_skill_items(
    document: pymupdf.Document,
    line: ExtractedLine,
    raw_items: str,
    source_start: int = 0,
) -> list[dict[str, Any]]:
    pieces = re.split(r"\s*[,;]\s*", raw_items)
    if pieces and re.search(r"\s+and\s+", pieces[-1], re.IGNORECASE):
        pieces[-1:] = re.split(r"\s+and\s+", pieces[-1], maxsplit=1, flags=re.IGNORECASE)
    skills: list[dict[str, Any]] = []
    search_from = source_start
    for piece in pieces:
        text = piece.strip(" .")
        if not text:
            continue
        start = line.text.casefold().find(text.casefold(), search_from)
        if start < 0:
            start = search_from
        end = start + len(text)
        search_from = end
        skills.append({
            "text": text,
            "page": line.page,
            "bbox": _span_bbox(document, line, text, start, end),
            "detectionMethod": "labeled_comma_list",
        })
    return skills


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

        skills = _education_skills(document, line)
        if skills:
            result["skills"].extend(skills)
            result["handledLineIndexes"].add(line_index)
            continue

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
    routed = sorted(
        (
            heading for heading in headings
            if _is_reliable_section_heading(lines[heading.line_index], heading)
        ),
        key=lambda item: item.line_index,
    )
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
        entities = _education_row_entities(document, row, ner_model, url_entities_by_line)
        primary = bool(entities["titles"] or entities["institutions"] or entities["dates"])
        structured_body = bool(entities["skills"])
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

        if structured_body:
            current = current or new_entry()
            current["_bodyStarted"] = True
            current["skills"].extend(entities["skills"])
            current["urls"].extend(entities["urls"])
            current["_lastBodyType"] = "skill"
            current["_lastLineBbox"] = [round(value, 2) for value in row[-1][1].bbox]
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
            if current["_lastBodyType"] == "skill" and current["_lastLineBbox"] is not None:
                previous_box = pymupdf.Rect(current["_lastLineBbox"])
                gap = line_box.y0 - previous_box.y1
                if (
                    -2.0 <= gap <= max(previous_box.height, line_box.height) * SETTINGS.section_router.paragraph_gap_multiplier
                    and not bullet
                ):
                    current["skills"].extend(
                        _education_skill_items(document, line, line.text)
                    )
                    current["_lastLineBbox"] = rounded_bbox
                    continue
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
    (output_directory / "summary.json").write_text(
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
    (output_directory / "experience.json").write_text(
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
        for item_index, item in enumerate(page_items):
            item_type = item["type"]
            box = _pixel_box(item["bbox"])
            detection_method = str(item.get("detectionMethod", ""))
            model_entity = detection_method.startswith(("distilbert", "minilm"))
            draw.rectangle(
                box,
                outline=colors[item_type],
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
            label_level = item_index % 3 + 1
            label = labels[item_type]
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
                    max(
                        0,
                        box[1] - SETTINGS.debug.label_y_offset * label_level,
                    ),
                )
            label_box = draw.textbbox(label_position, label)
            draw.rectangle(label_box, fill="#FFFFFF")
            draw.text(label_position, label, fill=colors[item_type])
        image.save(output_directory / f"page-{page_number}.png")


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
            items.extend({"type": item_type, **item} for item in entry[key])
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
            "section_heading": "route: section_heading",
            "experience_subheading": "route: metadata_line",
            "job_title": "MiniLM: job_title",
            "company": "NER: company",
            "date": "regex: date",
            "location": "NER: location",
            "url": "annotation: url",
            "paragraph": "route: paragraph",
            "bullet": "route: bullet",
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
    (output_directory / "education.json").write_text(
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
            "section_heading": "route: section_heading",
            "education_metadata": "route: geometry_row",
            "education_title": "pattern: education_title",
            "institution": "entity: institution",
            "date": "regex: date",
            "location": "NER: location",
            "gpa": "regex: GPA",
            "skill": "route: skill",
            "url": "annotation: url",
            "paragraph": "route: paragraph",
            "bullet": "route: bullet",
        },
        {"education_metadata"},
    )
