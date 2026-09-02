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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

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
    url: str | None = None
    bbox: tuple[float, float, float, float] | None = None


class DistilBertNerPredictor:
    """Adapt fixed CoNLL entities to the resume header entity interface."""

    _LABEL_TO_TYPE = {
        "LABEL_0": "O",
        "LABEL_1": "PER",
        "LABEL_2": "PER",
        "LABEL_3": "ORG",
        "LABEL_4": "ORG",
        "LABEL_5": "LOC",
        "LABEL_6": "LOC",
        "LABEL_7": "MISC",
        "LABEL_8": "MISC",
    }
    _TYPE_TO_LABEL = {
        "PER": "person name",
        "LOC": "location",
        "MISC": "nationality",
    }

    def __init__(self, model_directory: Path) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_directory,
            local_files_only=True,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_directory,
            local_files_only=True,
        )
        self._pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=-1,
        )

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float,
    ) -> list[dict[str, Any]]:
        requested_labels = set(labels)
        predictions: list[dict[str, Any]] = []
        for prediction in self._pipeline(text):
            raw_type = str(
                prediction.get("entity_group", prediction.get("entity", ""))
            ).upper()
            entity_type = self._LABEL_TO_TYPE.get(
                raw_type,
                raw_type.removeprefix("B-").removeprefix("I-"),
            )
            label = self._TYPE_TO_LABEL.get(entity_type)
            score = float(prediction.get("score", 0.0))
            if label not in requested_labels or score < threshold:
                continue
            predictions.append(
                {
                    "label": label,
                    "text": text[
                        int(prediction.get("start", 0)) : int(
                            prediction.get("end", 0)
                        )
                    ],
                    "start": int(prediction.get("start", 0)),
                    "end": int(prediction.get("end", 0)),
                    "score": score,
                }
            )
        return predictions


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d ()\-.]{6,}\d(?!\w)")
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
    with tempfile.TemporaryDirectory(prefix="extractor-v1-ocr-") as temporary_name:
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
                url=match.group(0) if kind == "url" else None,
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


def _masked_line_for_ner(
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


def _ner_matches_for_profile(
    ner_model: DistilBertNerPredictor,
    lines: list[ExtractedLine],
    profile_indexes: list[int],
    existing_matches: list[HeaderEntityMatch],
) -> list[HeaderEntityMatch]:
    """Run DistilBERT NER on each contact-masked header line."""
    label_to_kind = {
        "person name": "name",
        "location": "location",
        "nationality": "nationality",
    }
    accepted: list[HeaderEntityMatch] = []
    for line_index in profile_indexes:
        text = lines[line_index].text
        masked_text = _masked_line_for_ner(text, line_index, existing_matches)
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
                    detection_method="distilbert_ner",
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

    for url_match in _url_matches_for_lines(document, lines, profile_indexes):
        if _overlaps_existing(
            matches,
            line_index=url_match.line_index,
            start=url_match.start,
            end=url_match.end,
        ):
            continue
        matches.append(url_match)

    # The selected NER backend receives contact-masked lines independently.
    ner_matches = _ner_matches_for_profile(
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


def _section_body_style(
    content_lines: list[ExtractedLine],
) -> tuple[float, bool]:
    """Infer the ordinary prose style before testing possible subheadings."""
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
    """Require a real typographic contrast; plain body text stays paragraph text."""
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
    """Route section lines into typographic subheadings or paragraph blocks."""
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
    document: pymupdf.Document,
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
) -> list[dict[str, Any]]:
    """Use MiniLM-confirmed headings as boundaries and geometry within sections."""
    first_boundary = _first_header_boundary(lines, headings)
    if first_boundary is None:
        return []

    routed_headings = sorted(
        (
            heading
            for heading in headings
            if heading.line_index >= first_boundary.line_index
        ),
        key=lambda heading: heading.line_index,
    )
    routed_line_indexes = list(range(first_boundary.line_index, len(lines)))
    url_entities_by_line: dict[int, list[dict[str, Any]]] = {}
    for match in _url_matches_for_lines(document, lines, routed_line_indexes):
        url_entities_by_line.setdefault(match.line_index, []).append(
            _url_entity_value(document, lines, match)
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
        sections = build_sections(document, lines, headings)

        output_directory.mkdir(parents=True, exist_ok=True)
        header_debug_directory = output_directory / "debug" / "header"
        header_debug_directory.mkdir(parents=True, exist_ok=True)
        (header_debug_directory / "header.json").write_text(
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
        (output_directory / "sections-debug.json").write_text(
            json.dumps(
                {
                    "source": pdf_path.name,
                    "model": SETTINGS.model.name,
                    "modelRevision": SETTINGS.model.revision,
                    "sections": sections,
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


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the top resume profile region.")
    parser.add_argument(
        "--truths",
        action="store_true",
        help="Process only PDFs in resume-truths/ and write them under results/0-truths/.",
    )
    return parser.parse_args()


def _require_local_model(project_root: Path, relative_directory: str) -> Path:
    model_directory = project_root / relative_directory
    if not model_directory.is_dir():
        raise FileNotFoundError(
            f"local model directory is missing: {model_directory}"
        )
    return model_directory


def _load_ner_model(project_root: Path) -> DistilBertNerPredictor:
    model_directory = _require_local_model(
        project_root,
        SETTINGS.ner.distilbert_local_directory,
    )
    return DistilBertNerPredictor(model_directory)


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
        str(
            _require_local_model(
                project_root,
                SETTINGS.model.local_directory,
            )
        ),
    )
    ner_model = _load_ner_model(project_root)

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
