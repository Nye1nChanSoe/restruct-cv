"""Session fixtures. Loading both models costs seconds, so it happens once."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.helpers import (
    PROJECT_ROOT,
    models_available,
    tesseract_available,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite tests/golden/ from the current pipeline output.",
    )


@dataclass(frozen=True)
class LoadedModels:
    embedding: Any
    ner: Any


@pytest.fixture(scope="session")
def models() -> LoadedModels:
    if not models_available():
        pytest.skip("local models/ weights are absent; see README for setup")
    from extractor_v1.model import load_embedding_model, load_ner_model

    return LoadedModels(
        embedding=load_embedding_model(PROJECT_ROOT),
        ner=load_ner_model(PROJECT_ROOT),
    )


@pytest.fixture(scope="session")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One shared scratch directory, kept out of the repository."""
    return tmp_path_factory.mktemp("restruct-pipeline")


@pytest.fixture(scope="session")
def update_golden(pytestconfig: pytest.Config) -> bool:
    return bool(pytestconfig.getoption("--update-golden"))


def require_tesseract_for(stem: str) -> None:
    """Skip a scanned fixture when the OCR binary is not installed."""
    from tests.helpers import OCR_STEMS

    if stem in OCR_STEMS and not tesseract_available():
        pytest.skip(f"{stem} needs OCR and tesseract is not installed")
