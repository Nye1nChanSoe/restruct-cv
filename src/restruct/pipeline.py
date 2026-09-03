"""Pipeline orchestration.

The only place that knows the order of the stages. It holds no extraction logic
of its own, so the same engine can be driven from Python without the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf

from restruct.configs import SETTINGS
from restruct.debug.artifacts import (
    write_education_debug,
    write_experience_debug,
    write_ocr_extraction,
    write_projects_debug,
    write_raw_extraction,
    write_skills_debug,
    write_summary_debug,
    write_supplementary_sections_debug,
)
from restruct.debug.render import (
    render_combined_debug_images,
    render_debug_images,
    render_education_debug_images,
    render_experience_debug_images,
    render_projects_debug_images,
    render_skills_debug_images,
    render_summary_debug_images,
    render_supplementary_sections_debug_images,
)
from restruct.document.stats import measure
from restruct.ingestion.native import extracted_lines, read_document
from restruct.layout.words import reconstruct_words
from restruct.model import DistilBertNerPredictor, EmbeddingModel, detect_headings
from restruct.parsers.education import build_education_debug
from restruct.parsers.experience import build_experience_debug
from restruct.parsers.grouped import (
    build_projects_debug,
    build_supplementary_sections_debug,
)
from restruct.parsers.header import build_header_profile
from restruct.parsers.skills import build_skills_debug
from restruct.parsers.urls import _url_entity_value, _url_matches_for_lines
from restruct.schema import build_v1_resume, write_v1_resume
from restruct.structure.headings import first_header_boundary
from restruct.structure.sections import build_sections, summary_debug_value


def extract_resume(
    pdf_path: Path,
    output_directory: Path,
    raw_debug_path: Path,
    ocr_debug_path: Path,
    model: EmbeddingModel,
    ner_model: DistilBertNerPredictor,
) -> None:
    with pymupdf.open(pdf_path) as document:
        # Pass 1: read the document once into the shared representation.
        physical = read_document(document)
        statistics = measure(physical)
        # Pass 2: group characters into words using those measurements.
        physical = reconstruct_words(physical, statistics)

        write_raw_extraction(pdf_path, list(physical.raw_pages), raw_debug_path)
        write_ocr_extraction(pdf_path, list(physical.ocr_pages), ocr_debug_path)

        # Passes 3-5 still consume the flat line view; the bridge is removed as
        # each of them moves onto the physical representation.
        lines = extracted_lines(physical)
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
