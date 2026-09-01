"""Minimal PyMuPDF -> MiniLM heading/section extraction experiment."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from gliner import GLiNER
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer

from extractor_v1.configs import SETTINGS


@dataclass(frozen=True)
class ExtractedLine:
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    bold: bool
    used_ocr: bool


@dataclass(frozen=True)
class DetectedHeading:
    line_index: int
    section_type: str
    similarity: float
    runner_up_similarity: float


@dataclass(frozen=True)
class HeaderEntityMatch:
    kind: str
    text: str
    line_index: int
    start: int
    end: int
    detection_method: str
    confidence: float | None = None


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d ()\-.]{6,}\d(?!\w)")
_URL_RE = re.compile(
    r"(?:"
    r"https?://[^\s|,;)]+"
    r"|www\.[^\s|,;)]+"
    r"|(?<![@\w.-])(?:linkedin|github)\.com/[^\s|,;)]+"
    r"|(?<![@\w.-])[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z]{2,})+(?:/[^\s|,;)]*)?"
    r")",
    re.IGNORECASE,
)
_CONTACT_LABEL_RE = re.compile(
    r"\b(?:linkedin|github|portfolio|website|personal site)\b",
    re.IGNORECASE,
)
_HEADER_SEGMENT_RE = re.compile(r"[^|•·]+")
_JOB_TITLE_SEPARATOR_RE = re.compile(r"[|•·]|\s+/\s+")
_LOCATION_SEGMENT_RE = re.compile(
    r"^[^\d@]{2,48},\s*[^\d@]{2,48}$",
    re.UNICODE,
)
_NATIONALITY_CONTEXT_RE = re.compile(
    r"\b(?:citizen|citizenship|national|nationality)\b",
    re.IGNORECASE,
)
_NATIONALITY_PHRASE_RE = re.compile(
    r"(?:"
    r"\b[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*)?\s+"
    r"(?:citizen|national)\b"
    r"|\b(?:citizenship|nationality)\s*:\s*"
    r"[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*)?\b"
    r")",
    re.IGNORECASE,
)

def _meaningful_character_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _line_from_pdf_dict(
    raw_line: dict[str, Any],
    *,
    page_number: int,
    used_ocr: bool,
) -> ExtractedLine | None:
    spans = raw_line.get("spans", [])
    text = "".join(str(span.get("text", "")) for span in spans).strip()
    if not text:
        return None

    sizes = [float(span.get("size", 0.0)) for span in spans]
    fonts = [str(span.get("font", "")).casefold() for span in spans]
    flags = [int(span.get("flags", 0)) for span in spans]
    return ExtractedLine(
        page=page_number,
        text=text,
        bbox=tuple(float(value) for value in raw_line["bbox"]),
        size=max(sizes, default=0.0),
        bold=any(flag & 16 for flag in flags)
        or any("bold" in font or "black" in font for font in fonts),
        used_ocr=used_ocr,
    )


def extract_lines(
    document: pymupdf.Document,
) -> tuple[list[ExtractedLine], list[dict[str, Any]], list[dict[str, Any]]]:
    """Capture native PyMuPDF blocks, then use OCR only when native text is absent."""
    lines: list[ExtractedLine] = []
    raw_pages: list[dict[str, Any]] = []
    ocr_pages: list[dict[str, Any]] = []
    for page_index, page in enumerate(document):
        native_page_dict = page.get_text("dict", sort=True)
        native_text_blocks = [
            block for block in native_page_dict.get("blocks", []) if block.get("type") == 0
        ]
        raw_pages.append(
            {
                "page": page_index + 1,
                "width": page.rect.width,
                "height": page.rect.height,
                "blocks": native_text_blocks,
            }
        )
        native_text = "".join(
            str(span.get("text", ""))
            for block in native_text_blocks
            for raw_line in block.get("lines", [])
            for span in raw_line.get("spans", [])
        )
        used_ocr = (
            SETTINGS.ocr.enabled
            and _meaningful_character_count(native_text)
            < SETTINGS.ocr.native_text_min_characters
        )

        if used_ocr:
            text_page = page.get_textpage_ocr(
                language=SETTINGS.ocr.language,
                dpi=SETTINGS.ocr.dpi,
                full=True,
            )
            page_dict = page.get_text("dict", textpage=text_page, sort=True)
            ocr_pages.append(
                {
                    "page": page_index + 1,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "blocks": [
                        block
                        for block in page_dict.get("blocks", [])
                        if block.get("type") == 0
                    ],
                }
            )
        else:
            page_dict = native_page_dict

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for raw_line in block.get("lines", []):
                line = _line_from_pdf_dict(
                    raw_line,
                    page_number=page_index + 1,
                    used_ocr=used_ocr,
                )
                if line is not None:
                    lines.append(line)
    return lines, raw_pages, ocr_pages


def write_raw_extraction(
    pdf_path: Path,
    raw_pages: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Persist the untouched native text blocks captured before OCR or MiniLM."""
    if not SETTINGS.debug.raw_extraction_enabled:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "source": pdf_path.name,
                "pages": raw_pages,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_ocr_extraction(
    pdf_path: Path,
    ocr_pages: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Persist only the page dictionaries produced by the existing OCR fallback."""
    if not SETTINGS.debug.ocr_extraction_enabled or not ocr_pages:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "source": pdf_path.name,
                "pages": ocr_pages,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _looks_like_heading(line: ExtractedLine, page_median_size: float) -> bool:
    text = line.text.strip()
    words = text.split()
    if not 1 <= len(words) <= SETTINGS.heading.maximum_words:
        return False
    if len(text) > SETTINGS.heading.maximum_characters:
        return False
    if "@" in text or text.startswith(("http://", "https://", "www.")):
        return False
    if text.endswith((".", ",", ";", ":")):
        return False

    letters = [character for character in text if character.isalpha()]
    uppercase = (
        bool(letters)
        and sum(character.isupper() for character in letters) / len(letters)
        >= SETTINGS.heading.uppercase_ratio
    )
    larger = (
        page_median_size > 0
        and line.size >= page_median_size * SETTINGS.heading.font_size_multiplier
    )
    return line.bold or uppercase or larger


def detect_headings(
    lines: list[ExtractedLine],
    model: SentenceTransformer,
) -> list[DetectedHeading]:
    """Classify conservative visual candidates and abstain on weak/ambiguous matches."""
    sizes_by_page: dict[int, list[float]] = {}
    for line in lines:
        if line.size > 0:
            sizes_by_page.setdefault(line.page, []).append(line.size)
    median_by_page = {
        page: statistics.median(sizes) for page, sizes in sizes_by_page.items()
    }

    candidate_indexes = [
        index
        for index, line in enumerate(lines)
        if _looks_like_heading(line, median_by_page.get(line.page, line.size))
    ]
    if not candidate_indexes:
        return []

    reference_texts: list[str] = []
    reference_types: list[str] = []
    for section_type, examples in SETTINGS.section_references.items():
        for example in examples:
            reference_texts.append(example)
            reference_types.append(section_type)

    reference_embeddings = model.encode(reference_texts, normalize_embeddings=True)
    candidate_embeddings = model.encode(
        [lines[index].text for index in candidate_indexes],
        normalize_embeddings=True,
    )

    accepted: list[DetectedHeading] = []
    for line_index, candidate_embedding in zip(candidate_indexes, candidate_embeddings, strict=True):
        raw_scores = candidate_embedding @ reference_embeddings.T
        best_by_type: dict[str, float] = {}
        for section_type, score in zip(reference_types, raw_scores, strict=True):
            best_by_type[section_type] = max(best_by_type.get(section_type, -1.0), float(score))

        ranked = sorted(best_by_type.items(), key=lambda item: item[1], reverse=True)
        (winner_type, winner_score), (_, runner_up_score) = ranked[:2]
        if winner_score < SETTINGS.heading.similarity_threshold:
            continue
        if winner_score - runner_up_score < SETTINGS.heading.winner_margin:
            continue
        accepted.append(
            DetectedHeading(
                line_index=line_index,
                section_type=winner_type,
                similarity=winner_score,
                runner_up_similarity=runner_up_score,
            )
        )
    return accepted


def _first_header_boundary(
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
) -> DetectedHeading | None:
    """Return the first high-confidence section heading on page one."""
    exact_references = {
        reference.casefold().strip()
        for references in SETTINGS.section_references.values()
        for reference in references
    }
    for heading in headings:
        line = lines[heading.line_index]
        if line.page != 1 or heading.line_index == 0:
            continue
        exact_match = line.text.casefold().strip() in exact_references
        strong_semantic_match = (
            heading.similarity
            >= SETTINGS.header_profile.boundary_similarity_threshold
            and heading.similarity - heading.runner_up_similarity
            >= SETTINGS.header_profile.boundary_winner_margin
        )
        if exact_match or strong_semantic_match:
            return heading
    return None


def _append_regex_matches(
    matches: list[HeaderEntityMatch],
    *,
    line_index: int,
    text: str,
    kind: str,
    pattern: re.Pattern[str],
) -> None:
    for match in pattern.finditer(text):
        if _overlaps_existing(
            [
                existing
                for existing in matches
                if existing.kind in {"email", "phone", "url"}
            ],
            line_index=line_index,
            start=match.start(),
            end=match.end(),
        ):
            continue
        matches.append(
            HeaderEntityMatch(
                kind=kind,
                text=match.group(0),
                line_index=line_index,
                start=match.start(),
                end=match.end(),
                detection_method="regex",
            )
        )


def _overlaps_existing(
    matches: list[HeaderEntityMatch],
    *,
    line_index: int,
    start: int,
    end: int,
) -> bool:
    return any(
        match.line_index == line_index
        and match.start < end
        and start < match.end
        for match in matches
    )


def _entity_box(
    document: pymupdf.Document,
    line: ExtractedLine,
    entity_text: str,
    start: int,
    end: int,
) -> list[float]:
    """Resolve a substring box, using character offsets when OCR search fails."""
    page = document[line.page - 1]
    found = page.search_for(entity_text, clip=pymupdf.Rect(line.bbox))
    estimated_rectangle = None
    if line.text and 0 <= start < end <= len(line.text):
        line_rectangle = pymupdf.Rect(line.bbox)
        character_width = line_rectangle.width / len(line.text)
        estimated_rectangle = pymupdf.Rect(
            line_rectangle.x0 + character_width * start,
            line_rectangle.y0,
            line_rectangle.x0 + character_width * end,
            line_rectangle.y1,
        )
    if found and estimated_rectangle is not None:
        rectangle = min(
            found,
            key=lambda candidate: abs(
                (candidate.x0 + candidate.x1)
                - (estimated_rectangle.x0 + estimated_rectangle.x1)
            ),
        )
    elif found:
        rectangle = found[0]
    elif estimated_rectangle is not None:
        rectangle = estimated_rectangle
    else:
        rectangle = pymupdf.Rect(line.bbox)
    return [round(value, 2) for value in rectangle]


def _expand_location_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand a LOC token to its short comma-separated header segment."""
    for segment_match in _HEADER_SEGMENT_RE.finditer(text):
        if not (segment_match.start() <= start and end <= segment_match.end()):
            continue
        segment = segment_match.group(0)
        stripped = segment.strip(" \t,;:-\u200b")
        if not stripped or not _LOCATION_SEGMENT_RE.fullmatch(stripped):
            return start, end
        relative_start = segment.find(stripped)
        return (
            segment_match.start() + relative_start,
            segment_match.start() + relative_start + len(stripped),
        )
    return start, end


def _expand_nationality_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand a nationality token to its contextual header segment."""
    for segment_match in _HEADER_SEGMENT_RE.finditer(text):
        if not (segment_match.start() <= start and end <= segment_match.end()):
            continue
        segment = segment_match.group(0)
        if _NATIONALITY_CONTEXT_RE.search(segment) is None:
            return start, end
        stripped = segment.strip(" \t,;:-\u200b")
        relative_start = segment.find(stripped)
        return (
            segment_match.start() + relative_start,
            segment_match.start() + relative_start + len(stripped),
        )
    return start, end


def _masked_line_for_gliner(
    text: str,
    line_index: int,
    existing_matches: list[HeaderEntityMatch],
) -> str:
    """Hide deterministic contacts while preserving character offsets."""
    characters = list(text)
    for match in existing_matches:
        if match.line_index != line_index:
            continue
        if match.kind not in {"email", "phone", "url"}:
            continue
        for position in range(max(0, match.start), min(len(characters), match.end)):
            characters[position] = " "
    return "".join(characters)


def _looks_like_complete_name_line(text: str) -> bool:
    stripped = text.strip()
    return (
        1 <= len(stripped.split()) <= 6
        and len(stripped) <= 80
        and any(character.isalpha() for character in stripped)
        and not any(character.isdigit() for character in stripped)
        and not any(separator in stripped for separator in ("@", "|", "•", "·"))
    )


def _gliner_matches_for_profile(
    ner_model: GLiNER,
    lines: list[ExtractedLine],
    profile_indexes: list[int],
    existing_matches: list[HeaderEntityMatch],
) -> list[HeaderEntityMatch]:
    """Run GLiNER independently on each contact-masked header line."""
    label_to_kind = {
        "person name": "name",
        "location": "location",
        "nationality": "nationality",
    }
    accepted: list[HeaderEntityMatch] = []
    for line_index in profile_indexes:
        text = lines[line_index].text
        masked_text = _masked_line_for_gliner(text, line_index, existing_matches)
        if not masked_text.strip():
            continue
        predictions = ner_model.predict_entities(
            masked_text,
            list(SETTINGS.ner.labels),
            threshold=SETTINGS.ner.minimum_confidence,
        )
        for prediction in predictions:
            kind = label_to_kind.get(str(prediction.get("label", "")).casefold())
            if kind is None:
                continue
            start = max(0, int(prediction.get("start", 0)))
            end = min(len(text), int(prediction.get("end", start)))
            if start >= end or _overlaps_existing(
                existing_matches,
                line_index=line_index,
                start=start,
                end=end,
            ):
                continue
            if kind == "location":
                start, end = _expand_location_span(text, start, end)
            elif kind == "nationality":
                start, end = _expand_nationality_span(text, start, end)
            elif kind == "name" and _looks_like_complete_name_line(text):
                stripped_line = text.strip()
                start = text.find(stripped_line)
                end = start + len(stripped_line)
            entity_text = text[start:end].strip()
            if not entity_text:
                continue
            start += len(text[start:end]) - len(text[start:end].lstrip())
            end = start + len(entity_text)
            accepted.append(
                HeaderEntityMatch(
                    kind=kind,
                    text=entity_text,
                    line_index=line_index,
                    start=start,
                    end=end,
                    detection_method="gliner",
                    confidence=float(prediction.get("score", 0.0)),
                )
            )

    locations = [match for match in accepted if match.kind == "location"]
    return [
        match
        for match in accepted
        if match.kind != "nationality"
        or (
            _NATIONALITY_CONTEXT_RE.search(lines[match.line_index].text) is not None
            and not _overlaps_existing(
                locations,
                line_index=match.line_index,
                start=match.start,
                end=match.end,
            )
        )
    ]


def _job_title_segments(text: str) -> list[tuple[str, int, int]]:
    """Return up to the first configured header segments with source offsets."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for separator in _JOB_TITLE_SEPARATOR_RE.finditer(text):
        spans.append((cursor, separator.start()))
        cursor = separator.end()
    spans.append((cursor, len(text)))

    segments: list[tuple[str, int, int]] = []
    for raw_start, raw_end in spans:
        raw_segment = text[raw_start:raw_end]
        stripped = raw_segment.strip(" \t,;:-\u200b")
        if not stripped:
            continue
        start = raw_start + raw_segment.find(stripped)
        segments.append((stripped, start, start + len(stripped)))
        if len(segments) >= SETTINGS.job_title.maximum_segments_per_line:
            break
    return segments


def _semantic_job_title_matches(
    model: SentenceTransformer,
    lines: list[ExtractedLine],
    profile_indexes: list[int],
    existing_matches: list[HeaderEntityMatch],
) -> list[HeaderEntityMatch]:
    """Classify unmatched header segments as titles while preserving source text."""
    title_references = sorted(
        {
            reference
            for references in SETTINGS.job_title_references.values()
            for reference in references
        },
        key=len,
        reverse=True,
    )
    phrase_matches: list[HeaderEntityMatch] = []
    candidates: list[HeaderEntityMatch] = []
    for line_index in profile_indexes:
        text = lines[line_index].text
        for segment, start, end in _job_title_segments(text):
            if _overlaps_existing(
                existing_matches,
                line_index=line_index,
                start=start,
                end=end,
            ):
                continue
            for reference in title_references:
                for phrase in re.finditer(
                    rf"(?<!\w){re.escape(reference)}(?!\w)",
                    segment,
                    re.IGNORECASE,
                ):
                    phrase_start = start + phrase.start()
                    phrase_end = start + phrase.end()
                    if _overlaps_existing(
                        phrase_matches,
                        line_index=line_index,
                        start=phrase_start,
                        end=phrase_end,
                    ):
                        continue
                    phrase_matches.append(
                        HeaderEntityMatch(
                            kind="job_title",
                            text=text[phrase_start:phrase_end],
                            line_index=line_index,
                            start=phrase_start,
                            end=phrase_end,
                            detection_method="job_title_phrase",
                        )
                    )
            if not 1 <= len(segment.split()) <= SETTINGS.job_title.maximum_words:
                continue
            if not any(character.isalpha() for character in segment):
                continue
            if segment.endswith((".", ";")):
                continue
            candidates.append(
                HeaderEntityMatch(
                    kind="job_title",
                    text=segment,
                    line_index=line_index,
                    start=start,
                    end=end,
                    detection_method="semantic_similarity",
                )
            )

    if not candidates:
        return phrase_matches

    positive_references = title_references
    negative_references = [
        reference
        for references in SETTINGS.job_title_negative_references.values()
        for reference in references
    ]
    positive_embeddings = model.encode(
        positive_references,
        normalize_embeddings=True,
    )
    negative_embeddings = model.encode(
        negative_references,
        normalize_embeddings=True,
    )
    candidate_embeddings = model.encode(
        [candidate.text for candidate in candidates],
        normalize_embeddings=True,
    )

    accepted: list[HeaderEntityMatch] = []
    for candidate, embedding in zip(candidates, candidate_embeddings, strict=True):
        positive_score = float((embedding @ positive_embeddings.T).max())
        negative_score = float((embedding @ negative_embeddings.T).max())
        if positive_score < SETTINGS.job_title.similarity_threshold:
            continue
        if positive_score - negative_score < SETTINGS.job_title.winner_margin:
            continue
        accepted.append(
            HeaderEntityMatch(
                kind=candidate.kind,
                text=candidate.text,
                line_index=candidate.line_index,
                start=candidate.start,
                end=candidate.end,
                detection_method=candidate.detection_method,
                confidence=positive_score,
            )
        )
    for phrase_match in phrase_matches:
        if _overlaps_existing(
            accepted,
            line_index=phrase_match.line_index,
            start=phrase_match.start,
            end=phrase_match.end,
        ):
            continue
        accepted.append(phrase_match)
    return accepted


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
    semantic_model: SentenceTransformer,
    ner_model: Any,
) -> dict[str, Any] | None:
    """Detect the identity/contact region before the first likely section."""
    boundary = _first_header_boundary(lines, headings)
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

    # Claim deterministic contact spans first so semantic models never see email
    # usernames, domains, or phone fragments as possible named entities.
    matches: list[HeaderEntityMatch] = []
    for line_index in profile_indexes:
        text = lines[line_index].text
        _append_regex_matches(
            matches,
            line_index=line_index,
            text=text,
            kind="email",
            pattern=_EMAIL_RE,
        )
        _append_regex_matches(
            matches,
            line_index=line_index,
            text=text,
            kind="phone",
            pattern=_PHONE_RE,
        )
        _append_regex_matches(
            matches,
            line_index=line_index,
            text=text,
            kind="url",
            pattern=_URL_RE,
        )
        _append_regex_matches(
            matches,
            line_index=line_index,
            text=text,
            kind="url",
            pattern=_CONTACT_LABEL_RE,
        )

    # GLiNER receives each line separately with contact spans replaced by spaces.
    ner_matches = _gliner_matches_for_profile(
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
                    bool(_LOCATION_SEGMENT_RE.fullmatch(match.text)),
                    -match.line_index,
                    match.confidence or 0.0,
                ),
            )
        )

    # Resolve titles before geometry guesses a missing name. This prevents a
    # title-only OCR header from being mislabeled as a person.
    matches.extend(
        _semantic_job_title_matches(
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
                    for segment_match in _HEADER_SEGMENT_RE.finditer(text)
                    if _LOCATION_SEGMENT_RE.fullmatch(
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
            nationality_match = next(_NATIONALITY_PHRASE_RE.finditer(text), None)
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
    region_box = _union_boxes(profile_lines)[0]["bbox"]
    entities = []
    for match in unique_matches:
        line = lines[match.line_index]
        entity: dict[str, Any] = {
            "type": match.kind,
            "text": match.text,
            "page": line.page,
            "bbox": _entity_box(
                document,
                line,
                match.text,
                match.start,
                match.end,
            ),
            "detectionMethod": match.detection_method,
        }
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
            "bbox": [round(value, 2) for value in boundary_line.bbox],
            "similarity": round(boundary.similarity, 4),
        }

    return {
        "page": 1,
        "bbox": region_box,
        "text": "\n".join(line.text for line in profile_lines),
        "stoppedAtSection": boundary_value,
        "entities": entities,
    }


def _union_boxes(lines: list[ExtractedLine]) -> list[dict[str, Any]]:
    boxes_by_page: dict[int, pymupdf.Rect] = {}
    for line in lines:
        rectangle = pymupdf.Rect(line.bbox)
        boxes_by_page[line.page] = boxes_by_page.get(line.page, rectangle) | rectangle
    return [
        {"page": page, "bbox": [round(value, 2) for value in rectangle]}
        for page, rectangle in sorted(boxes_by_page.items())
    ]


def _pixel_box(bbox: list[float] | tuple[float, ...]) -> tuple[int, int, int, int]:
    return tuple(
        round(value * SETTINGS.debug.scale) for value in bbox
    )  # type: ignore[return-value]


def render_debug_images(
    document: pymupdf.Document,
    header_profile: dict[str, Any] | None,
    output_directory: Path,
) -> None:
    """Draw only the top profile region and its detected entities."""
    if header_profile is None:
        return
    output_directory.mkdir(parents=True, exist_ok=True)
    matrix = pymupdf.Matrix(SETTINGS.debug.scale, SETTINGS.debug.scale)
    page_number = header_profile["page"]
    page = document[page_number - 1]
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    draw = ImageDraw.Draw(image)

    profile_box = _pixel_box(header_profile["bbox"])
    draw.rectangle(
        profile_box,
        outline=SETTINGS.debug.header_region_color,
        width=SETTINGS.debug.header_region_stroke_width,
    )
    draw.text(
        (
            profile_box[0] + SETTINGS.debug.label_x_padding,
            max(0, profile_box[1] - SETTINGS.debug.label_y_offset),
        ),
        "header_profile",
        fill=SETTINGS.debug.header_region_color,
    )

    entity_colors = dict(SETTINGS.debug.header_entity_colors)
    for entity in header_profile["entities"]:
        color = entity_colors[entity["type"]]
        entity_box = _pixel_box(entity["bbox"])
        draw.rectangle(
            entity_box,
            outline=color,
            width=SETTINGS.debug.header_entity_stroke_width,
        )
        if entity["type"] == "name":
            label_position = (
                entity_box[2] + SETTINGS.debug.label_x_padding,
                entity_box[1],
            )
            label = "name"
        else:
            label_position = (
                entity_box[0] + SETTINGS.debug.label_x_padding,
                max(0, entity_box[1] - SETTINGS.debug.label_y_offset),
            )
            label = entity["type"]
        draw.text(
            label_position,
            label,
            fill=color,
        )

    image.save(output_directory / f"page-{page_number}.png")


def extract_resume(
    pdf_path: Path,
    output_directory: Path,
    raw_debug_path: Path,
    ocr_debug_path: Path,
    model: SentenceTransformer,
    ner_model: Any,
) -> None:
    with pymupdf.open(pdf_path) as document:
        lines, raw_pages, ocr_pages = extract_lines(document)
        write_raw_extraction(pdf_path, raw_pages, raw_debug_path)
        write_ocr_extraction(pdf_path, ocr_pages, ocr_debug_path)
        headings = detect_headings(lines, model)
        header_profile = build_header_profile(
            document,
            lines,
            headings,
            model,
            ner_model,
        )

        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "header.json").write_text(
            json.dumps(
                {
                    "source": pdf_path.name,
                    "model": SETTINGS.model.name,
                    "modelRevision": SETTINGS.model.revision,
                    "nerModel": SETTINGS.ner.name,
                    "nerModelRevision": SETTINGS.ner.revision,
                    "headerProfile": header_profile,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        render_debug_images(
            document,
            header_profile,
            output_directory / "debug" / "header",
        )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the top resume profile region.")
    parser.add_argument(
        "--truths",
        action="store_true",
        help="Process only PDFs in resume-truths/ and write them under results/0-truths/.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    project_root = Path(__file__).resolve().parents[2]
    if arguments.truths:
        input_directory = project_root / SETTINGS.paths.truths_input_directory
        output_root = project_root / SETTINGS.paths.truths_results_directory
    else:
        input_directory = project_root / SETTINGS.paths.input_directory
        output_root = project_root / SETTINGS.paths.results_directory
    raw_debug_directory = project_root / SETTINGS.debug.raw_extraction_directory
    ocr_debug_directory = project_root / SETTINGS.debug.ocr_extraction_directory
    model = SentenceTransformer(
        SETTINGS.model.name,
        revision=SETTINGS.model.revision,
    )
    ner_model = GLiNER.from_pretrained(SETTINGS.ner.name)

    for pdf_path in sorted(input_directory.glob("*.pdf")):
        resume_output = output_root / pdf_path.stem
        raw_debug_path = (
            resume_output / "raw-pymupdf.json"
            if arguments.truths
            else raw_debug_directory / f"{pdf_path.stem}.raw-pymupdf.json"
        )
        ocr_debug_path = (
            resume_output / "debug" / "ocr" / "raw-pymupdf.json"
            if arguments.truths
            else ocr_debug_directory / f"{pdf_path.stem}.ocr-pymupdf.json"
        )
        extract_resume(
            pdf_path,
            resume_output,
            raw_debug_path,
            ocr_debug_path,
            model,
            ner_model,
        )
        print(f"extracted: {pdf_path.name}")
