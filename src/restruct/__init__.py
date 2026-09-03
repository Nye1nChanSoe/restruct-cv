"""Restruct: explainable resume extraction from PDF and scanned documents.

This module is the public Python API. It re-exports the pipeline entry points
and the shared document types; everything else lives in the stage packages:

    ingestion/   physical extraction, native and OCR
    document/    shared types and document representation
    layout/      rows, paragraphs and bullet reconstruction
    structure/   headings, key-value pairs and section routing
    parsers/     one module per section shape
    models/      NER and embedding adapters
    patterns/    deterministic evidence
    debug/       artifacts and overlay rendering
    schema/      the versioned clean output
"""

from restruct.cli import main
from restruct.document.types import (
    DetectedHeading,
    ExtractedLine,
    HeaderEntityMatch,
)
from restruct.pipeline import extract_resume
from restruct.schema import build_v1_resume, write_v1_resume

__all__ = [
    "DetectedHeading",
    "ExtractedLine",
    "HeaderEntityMatch",
    "build_v1_resume",
    "extract_resume",
    "main",
    "write_v1_resume",
]
