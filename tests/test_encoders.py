"""The ONNX adapters: pooling, grouping, and what they promise their callers.

The golden snapshots prove the whole pipeline is unchanged by the swap off
torch. These are the properties underneath that, which a snapshot would only
report as a diff somewhere else entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

from restruct.encoders import DistilBertNerPredictor, MiniLmEncoder, _group_of, _tag_of
from tests.helpers import PROJECT_ROOT, models_available

needs_models = pytest.mark.skipif(
    not models_available(), reason="local ONNX weights are not present"
)


def _encoder() -> MiniLmEncoder:
    return MiniLmEncoder(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2")


@needs_models
def test_normalized_embeddings_are_unit_length() -> None:
    """Every similarity in the package is a dot product, which is only a
    cosine while the vectors are normalized."""
    embeddings = _encoder().encode(
        ["work experience", "education"], normalize_embeddings=True
    )
    assert embeddings.shape == (2, 384)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)


@needs_models
def test_a_single_string_returns_one_vector() -> None:
    """`sentence-transformers` collapsed the batch dimension for a bare
    string, and a caller that stopped getting that would index into a row."""
    encoder = _encoder()
    single = encoder.encode("work experience", normalize_embeddings=True)
    batched = encoder.encode(["work experience"], normalize_embeddings=True)
    assert single.shape == (384,)
    assert np.allclose(single, batched[0], atol=1e-5)


@needs_models
def test_padding_does_not_change_a_short_embedding() -> None:
    """Mean pooling is over the unmasked tokens only. A batch pads to its
    longest member, so a bug here would make an embedding depend on what it
    happened to be encoded alongside."""
    encoder = _encoder()
    alone = encoder.encode(["skills"], normalize_embeddings=True)[0]
    padded = encoder.encode(
        ["skills", "a considerably longer line of text to pad the batch out"],
        normalize_embeddings=True,
    )[0]
    assert np.allclose(alone, padded, atol=1e-4)


@needs_models
def test_presentation_keywords_are_accepted_and_ignored() -> None:
    """Callers pass `sentence-transformers` options that mean nothing here;
    rejecting them would make this a drop-in that is not one."""
    embeddings = _encoder().encode(
        ["skills"], normalize_embeddings=True, show_progress_bar=False
    )
    assert embeddings.shape == (1, 384)


@needs_models
def test_the_ner_span_indexes_the_original_text() -> None:
    """Every header entity is later reversed back onto the source line, so an
    offset that indexes anything but the original string is a silent
    mis-split rather than a visible failure."""
    predictor = DistilBertNerPredictor(PROJECT_ROOT / "models" / "distilbert-NER")
    text = "Alex Morgan | Bangkok, Thailand"
    predictions = predictor.predict_entities(text, ["person name"], threshold=0.5)
    assert predictions, "the model should find a person in a plain name line"
    for prediction in predictions:
        assert text[prediction["start"]:prediction["end"]] == prediction["text"]


def test_an_unprefixed_label_continues_rather_than_starts() -> None:
    """`O` carries no BIO prefix. Treating it as a beginning would break the
    run into one group per token and change every score."""
    assert _tag_of("B-PER") == ("B", "PER")
    assert _tag_of("I-PER") == ("I", "PER")
    assert _tag_of("O") == ("I", "O")


def test_a_group_spans_its_members_and_averages_their_scores() -> None:
    """This is the transformers pipeline's `simple` aggregation, and the
    thresholds in SETTINGS were tuned against the scores it produces."""
    group = _group_of(
        [
            {"label": "B-LOC", "score": 0.9, "start": 0, "end": 3},
            {"label": "I-LOC", "score": 0.7, "start": 3, "end": 7},
        ]
    )
    assert group == {
        "entity_group": "LOC",
        "score": pytest.approx(0.8),
        "start": 0,
        "end": 7,
    }
