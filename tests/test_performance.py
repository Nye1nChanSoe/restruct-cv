"""Guards on the properties that make a run fast.

A timing assertion in a test suite is a flaky test. These pin the *structure*
that produced the speed-up instead -- a fixed reference set is embedded once, a
model is read only when something needs it, and importing the package does not
drag in the model libraries. Each of those can regress silently and each is
cheap to check.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers import PROJECT_ROOT, SYNTHETIC_DIRECTORY, models_available

needs_models = pytest.mark.skipif(
    not models_available(),
    reason="local models/ weights are absent; see README for setup",
)


class CountingModel:
    """Records what it was asked to embed."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(self, sentences, **kwargs):
        import numpy

        self.calls.append(tuple(sentences))
        return numpy.zeros((len(sentences), 4), dtype="float32")


def test_a_reference_set_is_embedded_once_per_model() -> None:
    """Re-encoding the reference sets per call was 48% of a run: roughly two
    hundred fixed phrases, re-embedded once per experience metadata line."""
    from restruct.model import encode_references

    model = CountingModel()
    references = ("Software Engineer", "Data Analyst")
    first = encode_references(model, references)
    second = encode_references(model, references)

    assert len(model.calls) == 1
    assert first is second


def test_a_different_reference_set_is_embedded_separately() -> None:
    from restruct.model import encode_references

    model = CountingModel()
    encode_references(model, ("Software Engineer",))
    encode_references(model, ("Warehouse Operative",))
    assert len(model.calls) == 2


def test_two_models_do_not_share_a_cache() -> None:
    """The cache is keyed on the model, because two models embedding the same
    phrase do not produce the same vector."""
    from restruct.model import encode_references

    first, second = CountingModel(), CountingModel()
    references = ("Software Engineer",)
    encode_references(first, references)
    encode_references(second, references)
    assert len(first.calls) == 1 and len(second.calls) == 1


def test_candidate_text_is_never_cached() -> None:
    """Candidates are per-document and unbounded; caching them would be a leak
    rather than a saving. Only fixed configuration goes through the cache."""
    from restruct import model as model_module

    source = (PROJECT_ROOT / "src" / "restruct" / "model.py").read_text(
        encoding="utf-8"
    )
    assert "encode_references(model, candidates)" not in source
    assert "_REFERENCE_EMBEDDINGS" in dir(model_module)


@needs_models
def test_a_model_is_not_read_until_something_needs_it() -> None:
    from restruct.model import LazyEmbeddingModel, LazyNerPredictor

    embedding = LazyEmbeddingModel(PROJECT_ROOT / "models")
    ner = LazyNerPredictor(PROJECT_ROOT / "models")
    assert not embedding.loaded and not ner.loaded

    embedding.encode(["a heading"], normalize_embeddings=True)
    assert embedding.loaded
    assert not ner.loaded, "encoding text must not drag in the NER model"


def test_importing_the_package_does_not_import_the_model_libraries() -> None:
    """`restruct --help` and a run that fails validation used to pay four
    seconds for libraries they never touch, because __init__ re-exported the
    pipeline eagerly.

    The torch names stay in the probe after the ONNX swap: they are no longer
    dependencies, so finding one in ``sys.modules`` would mean something had
    started importing an installed copy again -- the export group's, most
    likely -- and the whole point of the swap is that a run never does."""
    probe = (
        "import sys, restruct, restruct.cli;"
        "heavy = [n for n in ('torch', 'transformers', 'sentence_transformers',"
        " 'onnxruntime', 'tokenizers')"
        " if n in sys.modules];"
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=True,
    )
    assert result.stdout.strip() == "", (
        f"importing restruct.cli pulled in {result.stdout.strip()}"
    )


def test_the_public_api_still_resolves_after_being_made_lazy() -> None:
    """PEP 562 keeps the names where they were; a typo in the lazy table would
    turn a public export into an AttributeError only callers would find."""
    import restruct

    assert restruct.extract_resume.__name__ == "extract_resume"
    assert restruct.main.__name__ == "main"
    assert set(restruct.__all__) <= set(dir(restruct))
    with pytest.raises(AttributeError):
        restruct.no_such_export
