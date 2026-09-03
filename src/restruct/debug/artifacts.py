"""Debug JSON artifacts.

These record the evidence behind every decision -- bounding boxes, fonts,
similarity scores, detection methods -- which the production schema deliberately
excludes. Nothing here should ever be read by the clean-output path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from restruct.configs import SETTINGS
from restruct.geometry import rounded


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




def write_layout_warnings(
    pdf_path: Path,
    warnings: tuple[Any, ...],
    output_directory: Path,
) -> None:
    """Record the unsupported layouts found, and that the check ran at all.

    Written even when nothing was found, so an empty list is a positive
    statement that the layout was examined rather than an absent file that
    could equally mean the detector never ran.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "layout-warnings.json").write_text(
        json.dumps(
            {
                "source": pdf_path.name,
                "supported": not warnings,
                "warnings": [
                    {
                        "kind": warning.kind,
                        "page": warning.page,
                        "detail": warning.detail,
                        "bbox": rounded(warning.bbox) if warning.bbox else None,
                    }
                    for warning in warnings
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
