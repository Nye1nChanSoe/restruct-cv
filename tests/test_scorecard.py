"""Guards on the accuracy scorecard itself, and the no-regression floor.

Golden snapshots prove output did not *change*. These tests prove it did not get
*worse*: every field's F1 must stay at or above the frozen baseline, so a commit
that silently trades one field's accuracy for another cannot pass unnoticed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests import scorecard
from tests.helpers import tesseract_available

# Floating-point noise only; a real accuracy drop is far larger than this.
TOLERANCE = 1e-6


@pytest.mark.parametrize(
    ("expected", "actual", "matches"),
    [
        ("Thai citizen", "Thai citizen", True),
        ("March 2022 – Present", "March 2022 - Present", True),
        ("  SOMCHAI  RATTANAKUL ", "somchai rattanakul", True),
        ("Thai citizen", "Thai", False),
        ("Bangkok, Thailand", "Bangkok", True),
        ("Software Engineer", "Data Analyst", False),
        ("Present", "P", False),
        ("", "anything", False),
    ],
)
def test_values_match(expected: str, actual: str, matches: bool) -> None:
    assert (
        scorecard.values_match(scorecard.normalize(expected), scorecard.normalize(actual))
        is matches
    )


def test_normalize_strips_invisible_and_folds_dashes() -> None:
    assert scorecard.normalize("Jan​ 2024 – Present") == "jan 2024 - present"


def test_score_sets_counts_each_side_once() -> None:
    counts = scorecard.score_sets(
        ["Alpha Corp", "Beta Ltd", "Gamma Inc"],
        ["Beta Ltd", "Delta LLC"],
    )
    assert (counts.true_positives, counts.false_positives, counts.false_negatives) == (1, 1, 2)


def test_score_scalar_treats_null_as_no_prediction() -> None:
    assert scorecard.score_scalar(None, None).supported is False
    assert scorecard.score_scalar("Thai", None).false_negatives == 1
    assert scorecard.score_scalar(None, "Thai").false_positives == 1


def test_every_synthetic_fixture_has_a_label() -> None:
    """A new fixture without a label would silently escape the scorecard."""
    from tests.helpers import label_path, synthetic_stems

    missing = [stem for stem in synthetic_stems() if not label_path(stem).exists()]
    assert not missing, f"synthetic fixtures without ground-truth labels: {missing}"


@pytest.fixture(scope="session")
def scores(models: Any) -> dict[str, scorecard.Counts]:
    if not tesseract_available():
        pytest.skip("scorecard needs tesseract for the scanned fixtures")
    totals, _, _ = scorecard.evaluate(models)
    return totals


def test_no_field_regressed_below_baseline(scores: dict[str, scorecard.Counts]) -> None:
    if not scorecard.BASELINE_PATH.exists():
        pytest.skip(
            "no baseline yet; create one with: "
            "uv run python -m tests.scorecard --update-baseline"
        )
    baseline = json.loads(scorecard.BASELINE_PATH.read_text(encoding="utf-8"))
    current = scorecard.baseline_scores(scores)

    regressions = {
        name: (floor, current.get(name))
        for name, floor in baseline.items()
        if current.get(name, 0.0) < floor - TOLERANCE
    }
    assert not regressions, (
        "accuracy regressed against tests/baseline_scores.json "
        f"(field: baseline -> now): {regressions}\n"
        "If the drop is an accepted trade-off, re-freeze with: "
        "uv run python -m tests.scorecard --update-baseline"
    )


def test_baseline_covers_every_scored_field(scores: dict[str, scorecard.Counts]) -> None:
    """A field that stops being scored should be noticed, not silently dropped."""
    if not scorecard.BASELINE_PATH.exists():
        pytest.skip("no baseline yet")
    baseline = json.loads(scorecard.BASELINE_PATH.read_text(encoding="utf-8"))
    missing = sorted(set(baseline) - set(scorecard.baseline_scores(scores)))
    assert not missing, f"fields present in the baseline are no longer scored: {missing}"
