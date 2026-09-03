"""Minimal PyMuPDF -> MiniLM heading/section extraction experiment."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageDraw

from restruct.configs import SETTINGS
from restruct.model import (
    DetectedHeading,
    DistilBertNerPredictor,
    EmbeddingModel,
    ExtractedLine,
    HEADER_SEGMENT_RE as _HEADER_SEGMENT_RE,
    HeaderEntityMatch,
    LOCATION_SEGMENT_RE as _LOCATION_SEGMENT_RE,
    NATIONALITY_PHRASE_RE as _NATIONALITY_PHRASE_RE,
    detect_headings,
    classify_profile_attribute_labels,
    load_embedding_model,
    load_ner_model,
    ner_matches_for_profile,
    overlaps_existing as _overlaps_existing,
    semantic_job_title_matches,
)
from restruct.routing import (
    build_education_debug,
    build_experience_debug,
    build_projects_debug,
    build_sections,
    build_skills_debug,
    build_supplementary_sections_debug,
    first_header_boundary,
    render_combined_debug_images,
    render_education_debug_images,
    render_experience_debug_images,
    render_projects_debug_images,
    render_skills_debug_images,
    render_supplementary_sections_debug_images,
    render_summary_debug_images,
    summary_debug_value,
    write_education_debug,
    write_experience_debug,
    write_projects_debug,
    write_skills_debug,
    write_supplementary_sections_debug,
    write_summary_debug,
)
from restruct.schema import build_v1_resume, write_v1_resume


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d ()\-.]{6,}\d(?!\w)")
_HEADER_ATTRIBUTE_LABEL_PATTERN = (
    r"date\s+of\s+birth|birth\s+date|d\.?\s*o\.?\s*b\.?|dob|"
    r"current\s+age|age|"
    r"gender|sex|marital\s+status|martial\s+status|civil\s+status|marital|"
    r"visa\s+status|visa\s+type|work\s+visa|immigration\s+status|residency\s+visa|"
    r"work\s+authorization|right\s+to\s+work|visa|"
    r"nationality|citizenship|"
    r"current\s+residen(?:ce|t)|current\s+location|place\s+of\s+residence|"
    r"residen(?:ce|t)"
)
_GENERIC_HEADER_ATTRIBUTE_RE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z .'/]{1,40}?)"
    r"\s*(?::|[-\u2013\u2014])\s*"
    r"(?P<value>.+?)(?=\s*[|\u2022\u00b7]|$)",
    re.IGNORECASE,
)
_HEADER_ATTRIBUTE_RE = re.compile(
    rf"(?P<label>\b(?:{_HEADER_ATTRIBUTE_LABEL_PATTERN})\b)"
    rf"(?:\s*(?::|[-\u2013\u2014])\s*|\t+|\s+)"
    rf"(?P<value>.+?)"
    rf"(?=\s*(?:[|\u2022\u00b7]|\b(?:{_HEADER_ATTRIBUTE_LABEL_PATTERN})\b)|$)",
    re.IGNORECASE,
)
_URL_RE = re.compile(
    r"(?:"
    r"https?://[^\s|,;)]+"
    r"|www\.[^\s|,;)]+"
    r"|(?<![@\w.-])(?:linkedin|github)\.com/[^\s|,;)]+"
    r"|(?<![@\w.-])[a-z0-9](?:[a-z0-9-]{0,59}[a-z0-9])"
    r"(?:\.[a-z]{2,})+(?:/[^\s|,;)]*)?"
    r"(?![@\w-]|\.[a-z0-9])"
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


def _run_ocr_command(arguments: list[str], program_name: str) -> subprocess.CompletedProcess[str]:
    """Run one required native OCR command with a focused error message."""
    try:
        return subprocess.run(
            arguments,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"required OCR program is not installed: {program_name}"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "unknown error"
        raise RuntimeError(f"{program_name} failed: {detail}") from error


def _tesseract_page_dict(
    page: pymupdf.Page,
    page_index: int,
    temporary_directory: Path,
) -> dict[str, Any]:
    """Render one PDF page with PyMuPDF and rebuild line geometry from TSV."""
    page_number = page_index + 1
    image_path = temporary_directory / f"page-{page_number}.png"
    pixmap = page.get_pixmap(dpi=SETTINGS.ocr.dpi, alpha=False)
    pixmap.save(image_path)
    pixel_width, pixel_height = pixmap.width, pixmap.height
    x_scale = page.rect.width / pixel_width
    y_scale = page.rect.height / pixel_height

    tesseract_result = _run_ocr_command(
        [
            SETTINGS.ocr.tesseract_command,
            str(image_path),
            "stdout",
            "-l",
            SETTINGS.ocr.language,
            "--oem",
            str(SETTINGS.ocr.engine_mode),
            "--psm",
            str(SETTINGS.ocr.page_segmentation_mode),
            "tsv",
        ],
        SETTINGS.ocr.tesseract_command,
    )

    words_by_line: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    reader = csv.DictReader(
        io.StringIO(tesseract_result.stdout),
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
    )
    for row in reader:
        text = str(row.get("text") or "").strip()
        if row.get("level") != "5" or not text:
            continue
        left = int(row["left"])
        top = int(row["top"])
        width = int(row["width"])
        height = int(row["height"])
        word = {
            "text": text,
            "bbox": (
                left * x_scale,
                top * y_scale,
                (left + width) * x_scale,
                (top + height) * y_scale,
            ),
            "confidence": float(row.get("conf") or -1.0),
            "size": height * y_scale,
        }
        line_key = (
            int(row["block_num"]),
            int(row["par_num"]),
            int(row["line_num"]),
        )
        words_by_line.setdefault(line_key, []).append(word)

    lines_by_block: dict[int, list[dict[str, Any]]] = {}
    for (block_number, _, _), words in words_by_line.items():
        rectangle = pymupdf.Rect(words[0]["bbox"])
        for word in words[1:]:
            rectangle |= pymupdf.Rect(word["bbox"])
        line_text = " ".join(str(word["text"]) for word in words)
        confidence_values = [
            float(word["confidence"])
            for word in words
            if float(word["confidence"]) >= 0
        ]
        raw_line = {
            "bbox": tuple(float(value) for value in rectangle),
            "spans": [
                {
                    "text": line_text,
                    "bbox": tuple(float(value) for value in rectangle),
                    "size": statistics.median(
                        float(word["size"]) for word in words
                    ),
                    "font": "TesseractOCR",
                    "flags": 0,
                }
            ],
            "ocrConfidence": (
                sum(confidence_values) / len(confidence_values)
                if confidence_values
                else None
            ),
        }
        lines_by_block.setdefault(block_number, []).append(raw_line)

    blocks: list[dict[str, Any]] = []
    for block_number, block_lines in lines_by_block.items():
        block_rectangle = pymupdf.Rect(block_lines[0]["bbox"])
        for raw_line in block_lines[1:]:
            block_rectangle |= pymupdf.Rect(raw_line["bbox"])
        blocks.append(
            {
                "type": 0,
                "number": block_number,
                "bbox": tuple(float(value) for value in block_rectangle),
                "lines": block_lines,
            }
        )
    return {
        "page": page_number,
        "width": page.rect.width,
        "height": page.rect.height,
        "engine": "tesseract_cli",
        "renderer": "pymupdf_pixmap",
        "dpi": SETTINGS.ocr.dpi,
        "pageSegmentationMode": SETTINGS.ocr.page_segmentation_mode,
        "blocks": blocks,
    }


def extract_lines(
    document: pymupdf.Document,
) -> tuple[list[ExtractedLine], list[dict[str, Any]], list[dict[str, Any]]]:
    """Use PyMuPDF for native text/rendering and Tesseract TSV for OCR pages."""
    lines: list[ExtractedLine] = []
    raw_pages: list[dict[str, Any]] = []
    ocr_pages: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="restruct-ocr-") as temporary_name:
        temporary_directory = Path(temporary_name)
        for page_index, page in enumerate(document):
            native_page_dict = page.get_text("dict", sort=True)
            native_text_blocks = [
                block
                for block in native_page_dict.get("blocks", [])
                if block.get("type") == 0
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
                page_dict = _tesseract_page_dict(
                    page,
                    page_index,
                    temporary_directory,
                )
                ocr_pages.append(page_dict)
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
    """Persist reconstructed blocks produced from Tesseract TSV output."""
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
            matches,
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
                url=match.group(0) if kind == "url" else None,
            )
        )


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
        for match in _HEADER_ATTRIBUTE_RE.finditer(line.text):
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
                        _entity_box(
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
        for match in _GENERIC_HEADER_ATTRIBUTE_RE.finditer(line.text):
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
                    _entity_box(
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


def _annotation_text_span(
    line: ExtractedLine,
    annotation_rectangle: pymupdf.Rect,
    page_words: list[tuple[Any, ...]],
) -> tuple[str, int, int] | None:
    """Map the visible words under a PDF link annotation back to one line."""
    line_rectangle = pymupdf.Rect(line.bbox)
    selected_words = [
        str(word[4])
        for word in page_words
        if len(word) >= 5
        and pymupdf.Rect(word[:4]).intersects(annotation_rectangle)
        and pymupdf.Rect(word[:4]).intersects(line_rectangle)
    ]
    if selected_words:
        folded_line = line.text.casefold()
        cursor = 0
        positions: list[tuple[int, int]] = []
        for word in selected_words:
            position = folded_line.find(word.casefold(), cursor)
            if position < 0:
                position = folded_line.find(word.casefold())
            if position < 0:
                continue
            positions.append((position, position + len(word)))
            cursor = position + len(word)
        if positions:
            start = min(position[0] for position in positions)
            end = max(position[1] for position in positions)
            return line.text[start:end], start, end

    intersection = line_rectangle & annotation_rectangle
    if intersection.is_empty or line_rectangle.width <= 0 or not line.text:
        return None
    start = round(
        len(line.text)
        * max(0.0, intersection.x0 - line_rectangle.x0)
        / line_rectangle.width
    )
    end = round(
        len(line.text)
        * min(line_rectangle.width, intersection.x1 - line_rectangle.x0)
        / line_rectangle.width
    )
    raw_text = line.text[start:end]
    visible_text = raw_text.strip()
    if not visible_text:
        return None
    start += len(raw_text) - len(raw_text.lstrip())
    return visible_text, start, start + len(visible_text)


def _annotation_url_matches(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    line_indexes: list[int],
) -> list[HeaderEntityMatch]:
    """Match PDF link rectangles to visible lines with a small bbox tolerance."""
    indexes_by_page: dict[int, list[int]] = {}
    for line_index in line_indexes:
        indexes_by_page.setdefault(lines[line_index].page, []).append(line_index)

    matches: list[HeaderEntityMatch] = []
    tolerance = SETTINGS.url.annotation_bbox_tolerance
    for page_number, page_line_indexes in indexes_by_page.items():
        page = document[page_number - 1]
        page_words = list(page.get_text("words", sort=True))
        for link in page.get_links():
            url = str(link.get("uri") or "").strip()
            if not url or url.casefold().startswith(("mailto:", "tel:")):
                continue
            raw_rectangle = link.get("from")
            if raw_rectangle is None:
                continue
            annotation_rectangle = pymupdf.Rect(raw_rectangle)
            matching_rectangle = pymupdf.Rect(
                annotation_rectangle.x0 - tolerance,
                annotation_rectangle.y0 - tolerance,
                annotation_rectangle.x1 + tolerance,
                annotation_rectangle.y1 + tolerance,
            )
            exact_candidates: list[tuple[int, float]] = []
            tolerant_candidates: list[tuple[int, float]] = []
            for line_index in page_line_indexes:
                line = lines[line_index]
                line_rectangle = pymupdf.Rect(line.bbox)
                center_distance = abs(
                    (line_rectangle.y0 + line_rectangle.y1)
                    - (annotation_rectangle.y0 + annotation_rectangle.y1)
                )
                if line_rectangle.intersects(annotation_rectangle):
                    exact_candidates.append((line_index, center_distance))
                elif line_rectangle.intersects(matching_rectangle):
                    tolerant_candidates.append((line_index, center_distance))

            candidates = sorted(
                exact_candidates or tolerant_candidates,
                key=lambda candidate: candidate[1],
            )[:1]
            for line_index, _ in candidates:
                line = lines[line_index]
                line_rectangle = pymupdf.Rect(line.bbox)
                span = _annotation_text_span(
                    line,
                    matching_rectangle,
                    page_words,
                )
                if span is None:
                    continue
                text, start, end = span
                matches.append(
                    HeaderEntityMatch(
                        kind="url",
                        text=text,
                        line_index=line_index,
                        start=start,
                        end=end,
                        detection_method="pdf_annotation",
                        url=url,
                        bbox=tuple(
                            float(value)
                            for value in (
                                line_rectangle
                                & (
                                    annotation_rectangle
                                    if line_rectangle.intersects(annotation_rectangle)
                                    else matching_rectangle
                                )
                            )
                        ),
                    )
                )
    return matches


def _url_matches_for_lines(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    line_indexes: list[int],
) -> list[HeaderEntityMatch]:
    """Prefer annotation targets, then fill remaining visible URLs with regex."""
    matches = _annotation_url_matches(document, lines, line_indexes)
    for line_index in line_indexes:
        _append_regex_matches(
            matches,
            line_index=line_index,
            text=lines[line_index].text,
            kind="url",
            pattern=_URL_RE,
        )
    return matches


def _url_entity_value(
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    match: HeaderEntityMatch,
) -> dict[str, Any]:
    """Serialize one URL match consistently for headers and later sections."""
    line = lines[match.line_index]
    return {
        "type": "url",
        "text": match.text,
        "url": match.url or match.text,
        "page": line.page,
        "bbox": (
            [round(value, 2) for value in match.bbox]
            if match.bbox is not None
            else _entity_box(
                document,
                line,
                match.text,
                match.start,
                match.end,
            )
        ),
        "detectionMethod": match.detection_method,
    }


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
                    bool(_LOCATION_SEGMENT_RE.fullmatch(match.text)),
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
            "bbox": (
                [round(value, 2) for value in match.bbox]
                if match.bbox is not None
                else _entity_box(
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
    model: EmbeddingModel,
    ner_model: DistilBertNerPredictor,
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
        url_entities_by_line: dict[int, list[dict[str, Any]]] = {}
        section_boundary = first_header_boundary(lines, headings)
        section_line_indexes = (
            list(range(section_boundary.line_index, len(lines)))
            if section_boundary is not None
            else []
        )
        for match in _url_matches_for_lines(document, lines, section_line_indexes):
            url_entities_by_line.setdefault(match.line_index, []).append(
                _url_entity_value(document, lines, match)
            )
        sections = build_sections(lines, headings, url_entities_by_line)
        summary = summary_debug_value(sections)
        experience = build_experience_debug(
            document,
            lines,
            headings,
            model,
            ner_model,
            url_entities_by_line,
        )
        education = build_education_debug(
            document,
            lines,
            headings,
            ner_model,
            url_entities_by_line,
        )
        skills = build_skills_debug(
            document,
            lines,
            headings,
            url_entities_by_line,
        )
        projects = build_projects_debug(
            document,
            lines,
            headings,
            url_entities_by_line,
        )
        supplementary_sections = build_supplementary_sections_debug(
            document,
            lines,
            headings,
            url_entities_by_line,
            model,
        )

        output_directory.mkdir(parents=True, exist_ok=True)
        header_debug_directory = output_directory / "debug" / "header"
        summary_debug_directory = output_directory / "debug" / "summary"
        experience_debug_directory = output_directory / "debug" / "experience"
        education_debug_directory = output_directory / "debug" / "education"
        skills_debug_directory = output_directory / "debug" / "skills"
        projects_debug_directory = output_directory / "debug" / "projects"
        header_debug_directory.mkdir(parents=True, exist_ok=True)
        (header_debug_directory / "header.json").unlink(missing_ok=True)
        (header_debug_directory / "header-raw.json").write_text(
            json.dumps(
                {
                    "source": pdf_path.name,
                    "model": SETTINGS.model.name,
                    "modelRevision": SETTINGS.model.revision,
                    "nerBackend": "distilbert",
                    "nerModel": SETTINGS.ner.distilbert_name,
                    "nerModelRevision": SETTINGS.ner.distilbert_revision,
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
            header_debug_directory,
        )
        write_summary_debug(
            pdf_path,
            summary,
            summary_debug_directory,
        )
        render_summary_debug_images(
            document,
            summary,
            summary_debug_directory,
        )
        write_experience_debug(
            pdf_path,
            experience,
            experience_debug_directory,
        )
        render_experience_debug_images(
            document,
            experience,
            experience_debug_directory,
        )
        write_education_debug(
            pdf_path,
            education,
            education_debug_directory,
        )
        render_education_debug_images(
            document,
            education,
            education_debug_directory,
        )
        write_skills_debug(
            pdf_path,
            skills,
            skills_debug_directory,
        )
        render_skills_debug_images(
            document,
            skills,
            skills_debug_directory,
        )
        write_projects_debug(
            pdf_path,
            projects,
            projects_debug_directory,
        )
        render_projects_debug_images(
            document,
            projects,
            projects_debug_directory,
        )
        write_supplementary_sections_debug(
            pdf_path,
            supplementary_sections,
            output_directory / "debug",
        )
        render_supplementary_sections_debug_images(
            document,
            supplementary_sections,
            output_directory / "debug",
        )
        render_combined_debug_images(
            document,
            header_profile,
            summary,
            experience,
            education,
            skills,
            projects,
            supplementary_sections,
            output_directory / "debug",
        )
        write_v1_resume(
            output_directory,
            build_v1_resume(
                header_profile=header_profile,
                summary=summary,
                experience=experience,
                education=education,
                skills=skills,
                projects=projects,
                supplementary_sections=supplementary_sections,
            ),
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
    model = load_embedding_model(project_root)
    ner_model = load_ner_model(project_root)

    for pdf_path in sorted(input_directory.glob("*.pdf")):
        resume_output = output_root / pdf_path.stem
        raw_debug_path = (
            resume_output / "raw-pymupdf.json"
            if arguments.truths
            else raw_debug_directory / f"{pdf_path.stem}.raw-pymupdf.json"
        )
        ocr_debug_path = (
            resume_output / "debug" / "ocr" / "raw-tesseract.json"
            if arguments.truths
            else ocr_debug_directory / f"{pdf_path.stem}.ocr-tesseract.json"
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
