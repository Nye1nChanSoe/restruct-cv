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
    parser.addoption(
        "--fresh-clone",
        action="store_true",
        default=False,
        help="Pretend this machine has neither an OCR engine nor model weights.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Make a developed machine look like the runner a contributor arrives on.

    CONTRIBUTING.md claims a fresh clone is green because the tests that need
    weights or Tesseract *skip*. That claim is unfalsifiable on the machine of
    anyone who works on this: they have both installed, so a test that should
    have skipped runs and passes, and the break only appears in CI. It has
    twice.

    Patched in ``pytest_configure``, before any test module is imported. It
    takes away what ``find_tesseract`` *consults* rather than replacing the
    function, because two tests in ``test_ocr.py`` are about that function
    finding a binary, and they set ``shutil.which`` themselves -- replacing the
    lookup would make them assert against this fixture instead of against the
    code. One patch still covers the whole suite: the pipeline calls
    ``find_tesseract``, and ``tesseract_available`` delegates to it.
    """
    if not config.getoption("--fresh-clone"):
        return

    from restruct.ingestion import ocr
    import tests.helpers as helpers

    ocr.shutil.which = lambda command: None
    ocr._known_install_locations = tuple
    helpers.models_available = lambda: False


@dataclass(frozen=True)
class LoadedModels:
    embedding: Any
    ner: Any


@pytest.fixture(scope="session")
def models() -> LoadedModels:
    if not models_available():
        pytest.skip("local models/ weights are absent; see README for setup")
    from restruct.model import load_embedding_model, load_ner_model

    return LoadedModels(
        embedding=load_embedding_model(PROJECT_ROOT / "models"),
        ner=load_ner_model(PROJECT_ROOT / "models"),
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
