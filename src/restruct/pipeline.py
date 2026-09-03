"""Pipeline orchestration.

The only place that knows the order of the stages. It holds no extraction logic
of its own, so the same engine can be driven from Python without the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf

from restruct.configs import SETTINGS
from restruct.debug.artifacts import (
    write_layout_warnings,
    write_ocr_extraction,
    write_raw_evidence,
    write_raw_extraction,
    write_supplementary_raw_evidence,
)
from restruct.debug.stages import render_sections, render_stage_overlays
from restruct.debug.render import render_combined_debug_images
from restruct.document.stats import measure
from restruct.ingestion.native import extracted_lines, read_document
from restruct.layout.unsupported import detect_unsupported_layouts
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
from restruct.stages import ALL_STAGES, DEFAULT_DEBUG_STAGES  # noqa: F401  (re-export)
from restruct.structure.headings import first_header_boundary
from restruct.structure.sections import build_sections, summary_debug_value


def extract_resume(
    pdf_path: Path,
    output_directory: Path,
    raw_debug_path: Path,
    ocr_debug_path: Path,
    model: EmbeddingModel,
    ner_model: DistilBertNerPredictor,
    *,
    stages: frozenset[int] = ALL_STAGES,
) -> None:
    """Extract one resume, writing ``resume.json`` and the requested artifacts.

    ``stages`` selects debug output only. An empty set writes nothing but the
    result, which is what the single-file CLI does unless asked otherwise.
    """
    with pymupdf.open(pdf_path) as document:
        # Pass 1: read the document once into the shared representation.
        physical = read_document(document)
        statistics = measure(physical)
        # Pass 2: group characters into words using those measurements.
        physical = reconstruct_words(physical, statistics)
        # Recorded before anything reads the lines in order, because every
        # later pass assumes that order is the author's.
        layout_warnings = detect_unsupported_layouts(physical, statistics)

        if stages:
            # The untouched PyMuPDF and Tesseract dumps are debug output like
            # everything else: asking for no artifacts must leave no artifacts.
            write_raw_extraction(pdf_path, list(physical.raw_pages), raw_debug_path)
            write_ocr_extraction(pdf_path, list(physical.ocr_pages), ocr_debug_path)
        physical_stages = tuple(sorted(stages & {1, 2, 3}))
        if physical_stages:
            # Passes 1-3 render images and no JSON: their output is geometry,
            # which a dump cannot usefully convey.
            render_stage_overlays(
                document,
                physical,
                statistics,
                output_directory / "debug",
                stages=physical_stages,
                warnings=layout_warnings,
            )

        # Passes 3-5 still consume the flat line view; the bridge is removed as
        # each of them moves onto the physical representation.
        lines = extracted_lines(physical, statistics)
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
        sections = build_sections(lines, headings, url_entities_by_line, statistics)
        if 4 in stages:
            # Pass 4 is geometry too: a compound heading can split into the
            # right destinations while the blocks land under the wrong one,
            # and only the overlay shows which.
            render_sections(
                document,
                sections,
                output_directory / "debug" / "pass-4-sections",
            )
        summary = summary_debug_value(sections)
        experience = build_experience_debug(
            document,
            lines,
            headings,
            model,
            ner_model,
            url_entities_by_line,
            statistics,
        )
        education = build_education_debug(
            document,
            lines,
            headings,
            ner_model,
            url_entities_by_line,
            statistics,
        )
        skills = build_skills_debug(
            document,
            lines,
            headings,
            url_entities_by_line,
            statistics,
        )
        projects = build_projects_debug(
            document,
            lines,
            headings,
            url_entities_by_line,
            statistics,
        )
        supplementary_sections = build_supplementary_sections_debug(
            document,
            lines,
            headings,
            url_entities_by_line,
            model,
            statistics,
        )

        output_directory.mkdir(parents=True, exist_ok=True)

        # Two tracks, kept apart on disk as well as in shape. raw/ holds the
        # evidence -- boxes, fonts, confidences, detection methods -- and
        # debug/ holds only images. resume.json, alongside both, stays lean.
        raw_directory = output_directory / "raw"
        # Written whenever anything is: a warning is about the document, not
        # about a pass, and a reader inspecting any stage wants to know the
        # reading order may not be recoverable.
        if stages:
            write_layout_warnings(pdf_path, layout_warnings, raw_directory)
        if 5 in stages:
            write_raw_evidence(
                pdf_path,
                raw_directory,
                "headerProfile",
                header_profile,
                model=SETTINGS.model.name,
                modelRevision=SETTINGS.model.revision,
                nerBackend="distilbert",
                nerModel=SETTINGS.ner.distilbert_name,
                nerModelRevision=SETTINGS.ner.distilbert_revision,
            )
            write_raw_evidence(
                pdf_path,
                raw_directory,
                "summary",
                summary,
                model=SETTINGS.model.name,
                modelRevision=SETTINGS.model.revision,
            )
            write_raw_evidence(pdf_path, raw_directory, "experience", experience)
            write_raw_evidence(pdf_path, raw_directory, "education", education)
            write_raw_evidence(pdf_path, raw_directory, "skills", skills)
            write_raw_evidence(pdf_path, raw_directory, "projects", projects)
            write_supplementary_raw_evidence(
                pdf_path,
                raw_directory,
                supplementary_sections,
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
