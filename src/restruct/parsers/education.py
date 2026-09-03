"""Education records: degrees, institutions, dates, grades and coursework."""
from __future__ import annotations

from typing import Any

import pymupdf

from restruct.configs import SETTINGS
from restruct.document.stats import DocumentStatistics
from restruct.document.types import DetectedHeading, ExtractedLine
from restruct.geometry import resolve_span_box, rounded
from restruct.layout.blocks import continues_block, extend_block
from restruct.layout.rows import _row_value, _visual_rows
from restruct.model import DistilBertNerPredictor
from restruct.patterns.bullets import BULLET_RE
from restruct.patterns.dates import DATE_RANGE_RE
from restruct.patterns.education import COURSEWORK_RE, DEGREE_RE, GPA_RE, INSTITUTION_RE
from restruct.patterns.layout import PAGE_FOOTER_RE
from restruct.patterns.separators import METADATA_SEPARATOR_RE
from restruct.structure.headings import _routed_section_headings
from restruct.structure.metadata import _metadata_candidates


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
        if BULLET_RE.match(line.text):
            continue
        date_spans = list(DATE_RANGE_RE.finditer(line.text))
        for match in date_spans:
            result["dates"].append({
                "text": match.group(0),
                "page": line.page,
                "bbox": resolve_span_box(document, line, match.group(0), match.start(), match.end()),
                "detectionMethod": "date_regex",
            })

        gpa_matches = list(GPA_RE.finditer(line.text))
        for match in gpa_matches:
            result["gpa"].append({
                "text": match.group(0),
                "value": float(match.group("value")),
                "scale": float(match.group("scale")) if match.group("scale") else None,
                "page": line.page,
                "bbox": resolve_span_box(document, line, match.group(0), match.start(), match.end()),
                "detectionMethod": "gpa_regex",
            })
        if gpa_matches:
            result["handledLineIndexes"].add(line_index)

        candidates = _metadata_candidates(
            line.text,
            date_spans,
            METADATA_SEPARATOR_RE,
            preserve_education_hyphens=True,
        )
        for segment, start, end in candidates:
            if any(match.start() < end and start < match.end() for match in gpa_matches):
                continue
            has_title = DEGREE_RE.search(segment) is not None
            has_institution = INSTITUTION_RE.search(segment) is not None
            if has_title and not has_institution:
                result["titles"].append({
                    "text": segment,
                    "page": line.page,
                    "bbox": resolve_span_box(document, line, segment, start, end),
                    "detectionMethod": "education_title_pattern",
                })
                continue
            if has_institution:
                value: dict[str, Any] = {
                    "text": segment,
                    "page": line.page,
                    "bbox": resolve_span_box(document, line, segment, start, end),
                    "detectionMethod": "institution_pattern",
                }
                if line_urls:
                    value["urls"] = line_urls
                result["institutions"].append(value)
                if has_title:
                    result["titles"].append({
                        "text": segment,
                        "page": line.page,
                        "bbox": resolve_span_box(document, line, segment, start, end),
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
                    "bbox": resolve_span_box(document, line, segment, start, end),
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
                    "bbox": resolve_span_box(document, line, text, prediction_start, prediction_end),
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
    statistics: DocumentStatistics,
) -> dict[str, Any] | None:
    """Reconstruct education entries from visual rows and grounded entities."""
    routed = _routed_section_headings(lines, headings)
    position = next((i for i, item in enumerate(routed) if item.section_type == "education"), None)
    if position is None:
        return None
    heading = routed[position]
    end = routed[position + 1].line_index if position + 1 < len(routed) else len(lines)
    heading_line = lines[heading.line_index]
    rows = _visual_rows(lines, range(heading.line_index + 1, end), statistics)
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
            PAGE_FOOTER_RE.search(line.text) is not None
            and line.bbox[1] >= document[line.page - 1].rect.height * 0.88
            for _, line in row
        ):
            continue
        explicit_metadata = any(
            DATE_RANGE_RE.search(line.text)
            or GPA_RE.search(line.text)
            or DEGREE_RE.search(line.text)
            or INSTITUTION_RE.search(line.text)
            for _, line in row
        )
        metadata_shape = explicit_metadata and any(
            line.bold
            or METADATA_SEPARATOR_RE.search(line.text)
            or len(line.text.split()) <= SETTINGS.section_router.maximum_subheading_words
            for _, line in row
        )
        prose_row = any(
            COURSEWORK_RE.search(line.text)
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
            bullet = BULLET_RE.match(line.text)
            line_box = pymupdf.Rect(line.bbox)
            rounded_bbox = rounded(line.bbox)
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
                if continues_block(
                    current["_lastLineBbox"],
                    line_box,
                    same_page=True,
                    require_horizontal_overlap=False,
                ):
                    extend_block(current["bullets"][-1], text=line.text, box=line_box)
                    current["_lastLineBbox"] = rounded_bbox
                    continue
            if current["paragraphs"] and current["paragraphs"][-1]["page"] == line.page:
                previous = current["paragraphs"][-1]
                if continues_block(
                    previous["bbox"],
                    line_box,
                    same_page=True,
                    require_horizontal_overlap=True,
                ):
                    extend_block(previous, text=line.text, box=line_box)
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
            "bbox": rounded(heading_line.bbox),
            "similarity": round(heading.similarity, 4),
            "detectionMethod": "geometry_semantic",
        },
        "entries": entries,
        "stoppedAtSection": (
            {"sectionType": next_heading.section_type, "text": lines[next_heading.line_index].text}
            if next_heading else None
        ),
    }
