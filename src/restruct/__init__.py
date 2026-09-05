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

from restruct.document.types import (
    DetectedHeading,
    ExtractedLine,
    HeaderEntityMatch,
)
from restruct.schema import build_v1_resume, write_v1_resume

# ``main`` and ``extract_resume`` reach the model libraries, which cost about
# four seconds to import. Re-exporting them eagerly made every entry into this
# package pay that -- including `restruct --help` and a run that fails
# validation before a model is ever consulted. PEP 562 keeps the public names
# where they were and defers the cost to the first use.
# ``install_models`` is deferred for a different reason: it is the one entry
# point that opens a socket, and a library import should not pull the network
# stack in for a caller that only ever reads local weights.
_LAZY_EXPORTS = {
    "main": ("restruct.cli", "main"),
    "extract_resume": ("restruct.pipeline", "extract_resume"),
    "install_models": ("restruct.install", "install_models"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attribute = target
    return getattr(importlib.import_module(module_name), attribute)


def __dir__() -> list[str]:
    return sorted(__all__)

__all__ = [
    "DetectedHeading",
    "ExtractedLine",
    "HeaderEntityMatch",
    "build_v1_resume",
    "extract_resume",
    "install_models",
    "main",
    "write_v1_resume",
]
