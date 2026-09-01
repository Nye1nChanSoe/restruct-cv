"""Minimal PyMuPDF -> MiniLM heading/section extraction experiment."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# Conservative acceptance avoids forcing names, job titles, or body lines into sections.
SIMILARITY_THRESHOLD = 0.55
WINNER_MARGIN = 0.06
NATIVE_TEXT_MIN_CHARACTERS = 20
OCR_DPI = 200
DEBUG_SCALE = 2.0

SECTION_REFERENCES: dict[str, tuple[str, ...]] = {
    "summary": (
        "Professional Summary",
        "Career Profile",
        "Career Objective",
        "About Me",
    ),
    "experience": (
        "Work Experience",
        "Professional Experience",
        "Employment History",
        "Career History",
    ),
    "education": (
        "Education",
        "Academic Background",
        "Academic Qualifications",
    ),
    "skills": (
        "Skills",
        "Technical Skills",
        "Core Competencies",
        "Technologies and Tools",
    ),
    "projects": (
        "Projects",
        "Selected Projects",
        "Personal Projects",
        "Portfolio",
    ),
    "certifications": (
        "Certifications",
        "Certificates and Training",
        "Professional Certifications",
    ),
    "languages": (
        "Languages",
        "Language Proficiency",
    ),
    "volunteering": (
        "Volunteer Experience",
        "Community Involvement",
        "Community Service",
    ),
    "awards": (
        "Awards and Honors",
        "Achievements",
    ),
    "publications": (
        "Publications",
        "Research and Publications",
    ),
}

SECTION_COLORS: dict[str, str] = {
    "summary": "#2E7D32",
    "experience": "#1565C0",
    "education": "#7B1FA2",
    "skills": "#00897B",
    "projects": "#EF6C00",
    "certifications": "#C2185B",
    "languages": "#558B2F",
    "volunteering": "#00838F",
    "awards": "#F9A825",
    "publications": "#6D4C41",
}


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


def extract_lines(document: pymupdf.Document) -> list[ExtractedLine]:
    """Extract ordered lines, using OCR only for pages without native text."""
    lines: list[ExtractedLine] = []
    for page_index, page in enumerate(document):
        native_text = page.get_text("text")
        used_ocr = _meaningful_character_count(native_text) < NATIVE_TEXT_MIN_CHARACTERS

        if used_ocr:
            text_page = page.get_textpage_ocr(language="eng", dpi=OCR_DPI, full=True)
            page_dict = page.get_text("dict", textpage=text_page, sort=True)
        else:
            page_dict = page.get_text("dict", sort=True)

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
    return lines


def _looks_like_heading(line: ExtractedLine, page_median_size: float) -> bool:
    text = line.text.strip()
    words = text.split()
    if not 1 <= len(words) <= 8 or len(text) > 80:
        return False
    if "@" in text or text.startswith(("http://", "https://", "www.")):
        return False
    if text.endswith((".", ",", ";", ":")):
        return False

    letters = [character for character in text if character.isalpha()]
    uppercase = bool(letters) and sum(character.isupper() for character in letters) / len(letters) >= 0.8
    larger = page_median_size > 0 and line.size >= page_median_size * 1.12
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
    for section_type, examples in SECTION_REFERENCES.items():
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
        if winner_score < SIMILARITY_THRESHOLD:
            continue
        if winner_score - runner_up_score < WINNER_MARGIN:
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


def _union_boxes(lines: list[ExtractedLine]) -> list[dict[str, Any]]:
    boxes_by_page: dict[int, pymupdf.Rect] = {}
    for line in lines:
        rectangle = pymupdf.Rect(line.bbox)
        boxes_by_page[line.page] = boxes_by_page.get(line.page, rectangle) | rectangle
    return [
        {"page": page, "bbox": [round(value, 2) for value in rectangle]}
        for page, rectangle in sorted(boxes_by_page.items())
    ]


def build_sections(
    lines: list[ExtractedLine],
    headings: list[DetectedHeading],
) -> list[dict[str, Any]]:
    """Copy every extracted line between consecutive accepted headings."""
    sections: list[dict[str, Any]] = []
    for position, heading in enumerate(headings):
        next_index = (
            headings[position + 1].line_index
            if position + 1 < len(headings)
            else len(lines)
        )
        heading_line = lines[heading.line_index]
        content_lines = lines[heading.line_index + 1 : next_index]
        sections.append(
            {
                "heading": heading_line.text,
                "sectionType": heading.section_type,
                "similarity": round(heading.similarity, 4),
                "runnerUpSimilarity": round(heading.runner_up_similarity, 4),
                "page": heading_line.page,
                "headingBox": [round(value, 2) for value in heading_line.bbox],
                "contentBoxes": _union_boxes(content_lines),
                "content": "\n".join(line.text for line in content_lines),
                "usedOcr": heading_line.used_ocr
                or any(line.used_ocr for line in content_lines),
            }
        )
    return sections


def _pixel_box(bbox: list[float] | tuple[float, ...]) -> tuple[int, int, int, int]:
    return tuple(round(value * DEBUG_SCALE) for value in bbox)  # type: ignore[return-value]


def render_debug_images(
    document: pymupdf.Document,
    sections: list[dict[str, Any]],
    output_directory: Path,
) -> None:
    """Draw only heading boxes and their same-colored content regions."""
    output_directory.mkdir(parents=True, exist_ok=True)
    matrix = pymupdf.Matrix(DEBUG_SCALE, DEBUG_SCALE)

    for page_index, page in enumerate(document):
        page_number = page_index + 1
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        draw = ImageDraw.Draw(image)

        for section in sections:
            color = SECTION_COLORS[section["sectionType"]]
            if section["page"] == page_number:
                heading_box = _pixel_box(section["headingBox"])
                draw.rectangle(heading_box, outline=color, width=5)
                draw.text(
                    (heading_box[0] + 4, max(0, heading_box[1] - 14)),
                    section["sectionType"],
                    fill=color,
                )

            for content_box in section["contentBoxes"]:
                if content_box["page"] == page_number:
                    draw.rectangle(
                        _pixel_box(content_box["bbox"]),
                        outline=color,
                        width=3,
                    )

        image.save(output_directory / f"page-{page_number}.png")


def extract_resume(
    pdf_path: Path,
    output_directory: Path,
    model: SentenceTransformer,
) -> None:
    with pymupdf.open(pdf_path) as document:
        lines = extract_lines(document)
        headings = detect_headings(lines, model)
        sections = build_sections(lines, headings)

        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "sections.json").write_text(
            json.dumps(
                {
                    "source": pdf_path.name,
                    "model": MODEL_NAME,
                    "modelRevision": MODEL_REVISION,
                    "sections": sections,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        render_debug_images(document, sections, output_directory / "debug")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_directory = project_root / "resumes-synthetic"
    output_root = project_root / "results"
    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)

    for pdf_path in sorted(input_directory.glob("*.pdf")):
        extract_resume(pdf_path, output_root / pdf_path.stem, model)
        print(f"extracted: {pdf_path.name}")
