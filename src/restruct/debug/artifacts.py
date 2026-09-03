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


def write_raw_evidence(
    pdf_path: Path,
    raw_directory: Path,
    name: str,
    payload: Any,
    **extra: Any,
) -> None:
    """Write one section's raw evidence to ``results/<resume>/raw/<name>.json``.

    Seven near-identical copies of this stood here before, one per section,
    differing only in the key they wrote under. The evidence track is deliberate
    -- ``resume.json`` stays lean and metadata-free, and every bbox, font,
    confidence and detection method lives here instead.
    """
    if payload is None:
        return
    raw_directory.mkdir(parents=True, exist_ok=True)
    (raw_directory / f"{name}.json").write_text(
        json.dumps(
            {"source": pdf_path.name, **extra, name: payload},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_supplementary_raw_evidence(
    pdf_path: Path,
    raw_directory: Path,
    sections: dict[str, dict[str, Any]],
) -> None:
    """One file per minor section that produced anything."""
    for section_type, section in sections.items():
        write_raw_evidence(pdf_path, raw_directory, section_type, section)


def write_layout_warnings(
    pdf_path: Path,
    warnings: tuple[Any, ...],
    raw_directory: Path,
) -> None:
    """Record the unsupported layouts found, and that the check ran at all.

    Written even when nothing was found, so an empty list is a positive
    statement that the layout was examined rather than an absent file that
    could equally mean the detector never ran.
    """
    raw_directory.mkdir(parents=True, exist_ok=True)
    (raw_directory / "layout-warnings.json").write_text(
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
