"""Experience records: job titles, employers, dates, locations and body text."""
from __future__ import annotations

import re
from typing import Any

import pymupdf

from restruct.configs import SETTINGS
from restruct.document.types import DetectedHeading, ExtractedLine
from restruct.geometry import resolve_span_box, rounded
from restruct.layout.blocks import continues_block, extend_block
from restruct.model import DistilBertNerPredictor, EmbeddingModel, classify_job_title_candidates
from restruct.patterns.bullets import BULLET_RE
from restruct.patterns.dates import DATE_RANGE_RE
from restruct.patterns.organizations import COMPANY_MARKER_RE
from restruct.patterns.separators import METADATA_SEPARATOR_RE
from restruct.structure.headings import _routed_section_headings
from restruct.structure.metadata import _metadata_candidates


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
    date_spans = list(DATE_RANGE_RE.finditer(line.text))
    for match in date_spans:
        result["dates"].append({
            "text": match.group(0), "page": line.page,
            "bbox": resolve_span_box(document, line, match.group(0), match.start(), match.end()),
            "detectionMethod": "date_regex",
        })

    candidates = _metadata_candidates(
        line.text,
        date_spans,
        METADATA_SEPARATOR_RE,
    )
    classifications = classify_job_title_candidates(model, [item[0] for item in candidates])
    def candidate_urls(
        segment: str,
        start: int,
        end: int,
    ) -> list[dict[str, Any]]:
        normalized_segment = " ".join(
            segment.replace("\u200b", "").replace("\ufeff", "").casefold().split()
        ).strip(" |•·-–—")
        segment_box = pymupdf.Rect(resolve_span_box(document, line, segment, start, end))
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
                "companyMarker": COMPANY_MARKER_RE.search(segment) is not None,
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
                "bbox": resolve_span_box(document, line, segment, start, end),
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
                        "bbox": resolve_span_box(
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
                        "bbox": resolve_span_box(
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
                "bbox": resolve_span_box(document, line, segment, start, end),
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
                "bbox": resolve_span_box(document, line, segment, start, end),
                "confidence": round(max(float(item["score"]) for item in locations), 4),
                "detectionMethod": "distilbert_ner_reconciled",
            })
            continue

        if item["titleAccepted"]:
            result["jobTitles"].append({
                "text": segment,
                "page": line.page,
                "bbox": resolve_span_box(document, line, segment, start, end),
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
                "bbox": resolve_span_box(document, line, text, prediction_start, prediction_end),
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
        bullet = BULLET_RE.match(line.text)
        if bullet:
            current = current or new_entry()
            current["_bodyStarted"] = True
            current["bullets"].append({
                "text": line.text[bullet.end():].strip(), "page": line.page,
                "bbox": rounded(line.bbox),
                "detectionMethod": "bullet_marker",
            })
            current["_lastBodyType"] = "bullet"
            current["_lastLineBbox"] = rounded(line.bbox)
            continue

        can_be_metadata = (
            current is None
            or not current["_bodyStarted"]
            or line.bold
            or DATE_RANGE_RE.search(line.text) is not None
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
                "bbox": rounded(line.bbox),
                "detectionMethod": "experience_metadata",
            })
            for key in ("jobTitles", "companies", "dates", "locations", "urls"):
                current[key].extend(entities[key])
            continue

        current = current or new_entry()
        current["_bodyStarted"] = True
        current_box = pymupdf.Rect(line.bbox)
        if current["_lastBodyType"] == "bullet" and current["bullets"]:
            # Deliberately unconditional: a bullet may continue across a page
            # break, unlike the paragraph rules below.
            if continues_block(
                current["_lastLineBbox"],
                line.bbox,
                same_page=True,
                require_horizontal_overlap=False,
            ):
                extend_block(current["bullets"][-1], text=line.text, box=line.bbox)
                current["_lastLineBbox"] = rounded(line.bbox)
                continue
        paragraph = {
            "text": line.text, "page": line.page,
            "bbox": rounded(line.bbox),
            "detectionMethod": "geometry_default",
        }
        if current["paragraphs"] and current["paragraphs"][-1]["page"] == line.page:
            previous = current["paragraphs"][-1]
            # Anchored on the accumulated block box rather than the last line,
            # and with no horizontal-overlap requirement, unlike every other
            # paragraph continuation.
            if continues_block(
                previous["bbox"],
                line.bbox,
                same_page=True,
                require_horizontal_overlap=False,
            ):
                extend_block(previous, text=line.text, box=line.bbox)
                current["_lastBodyType"] = "paragraph"
                current["_lastLineBbox"] = rounded(line.bbox)
                continue
        current["paragraphs"].append(paragraph)
        current["_lastBodyType"] = "paragraph"
        current["_lastLineBbox"] = rounded(line.bbox)

    for entry in entries:
        entry.pop("_bodyStarted", None)
        entry.pop("_lastBodyType", None)
        entry.pop("_lastLineBbox", None)
    next_heading = routed[position + 1] if position + 1 < len(routed) else None
    return {
        "sectionType": "experience",
        "heading": {
            "text": heading_line.text, "page": heading_line.page,
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
