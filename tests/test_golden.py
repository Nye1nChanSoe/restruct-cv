"""Byte-for-byte regression guard on the clean resume.json output.

The refactor moves thousands of lines between modules. These snapshots are what
makes each of those moves verifiable: for a pure refactor the diff must be
empty, and a non-empty diff means the commit changed behavior.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import require_tesseract_for
from tests.helpers import (
    SYNTHETIC_DIRECTORY,
    dump_json,
    golden_path,
    run_pipeline,
    synthetic_stems,
)

STEMS = synthetic_stems()

# The 16 fixed v1 destinations, in the order the schema must always expose.
EXPECTED_SECTION_ORDER = (
    "header_profile",
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
    "licenses",
    "tools_equipment",
    "languages",
    "volunteering",
    "awards",
    "publications",
    "references",
    "interests",
    "others",
)


@pytest.fixture(scope="session")
def extracted(workspace: Path, models: Any) -> dict[str, dict[str, Any]]:
    """Run each fixture once per session and cache the clean output."""
    return {}


def _result(
    stem: str,
    extracted: dict[str, dict[str, Any]],
    workspace: Path,
    models: Any,
) -> dict[str, Any]:
    if stem not in extracted:
        extracted[stem] = run_pipeline(
            SYNTHETIC_DIRECTORY / f"{stem}.pdf",
            workspace,
            models,
        )
    return extracted[stem]


@pytest.mark.parametrize("stem", STEMS)
def test_matches_golden_snapshot(
    stem: str,
    extracted: dict[str, dict[str, Any]],
    workspace: Path,
    models: Any,
    update_golden: bool,
) -> None:
    require_tesseract_for(stem)
    actual = dump_json(_result(stem, extracted, workspace, models))
    path = golden_path(stem)

    if update_golden:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        pytest.skip(f"rebaselined {path.name}")

    if not path.exists():
        pytest.fail(
            f"missing golden snapshot {path.name}; "
            f"create it with: uv run pytest --update-golden"
        )

    expected = path.read_text(encoding="utf-8")
    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=f"golden/{path.name}",
                tofile=f"actual/{stem}",
            )
        )
        pytest.fail(
            f"{stem}.pdf output changed.\n\n{diff}\n"
            "If this change is intended, review every line above, then "
            "re-baseline with: uv run pytest --update-golden"
        )


@pytest.mark.parametrize("stem", STEMS)
def test_schema_exposes_every_section_in_order(
    stem: str,
    extracted: dict[str, dict[str, Any]],
    workspace: Path,
    models: Any,
) -> None:
    """All 16 destinations are always present, in the same order."""
    require_tesseract_for(stem)
    result = _result(stem, extracted, workspace, models)
    assert tuple(result.keys()) == EXPECTED_SECTION_ORDER


@pytest.mark.parametrize("stem", STEMS)
def test_output_carries_no_debug_metadata(
    stem: str,
    extracted: dict[str, dict[str, Any]],
    workspace: Path,
    models: Any,
) -> None:
    """Production JSON must stay lean: no geometry, models, or confidences."""
    require_tesseract_for(stem)
    result = _result(stem, extracted, workspace, models)
    forbidden = {
        "bbox",
        "page",
        "size",
        "bold",
        "similarity",
        "confidence",
        "detectionMethod",
        "model",
        "modelRevision",
        "entities",
        "stoppedAtSection",
        "metadataRows",
        "subheadingLines",
        "rows",
    }
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            found.update(forbidden.intersection(value.keys()))
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(result)
    assert not found, f"{stem}.pdf leaked debug metadata into clean output: {sorted(found)}"
