"""Field-level accuracy scorecard for the resume pipeline.

The golden snapshots in ``tests/golden/`` catch *changes*; they cannot tell a
fix from a regression. This module answers the other question: how close is the
extraction to hand-written ground truth, field by field.

Labels live in ``tests/labels/`` and were written by reading the resumes, not by
copying pipeline output, so a perfect score is not expected.

Run it directly::

    uv run python -m tests.scorecard                    # print the scorecard
    uv run python -m tests.scorecard --update-baseline  # freeze current scores
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tests.helpers import (
    LABEL_DIRECTORY,
    PROJECT_ROOT,
    SYNTHETIC_DIRECTORY,
    TRUTHS_DIRECTORY,
    TRUTHS_LABEL_DIRECTORY,
    OCR_STEMS,
    fixture_path,
    label_path,
    models_available,
    run_pipeline,
    synthetic_stems,
    tesseract_available,
)

BASELINE_PATH = Path(__file__).resolve().parent / "baseline_scores.json"
REPORT_PATH = Path(__file__).resolve().parent / "scorecard.md"

HEADER_SCALARS = (
    "name",
    "location",
    "date_of_birth",
    "age",
    "gender",
    "marital_status",
    "visa_status",
    "nationality",
    "current_residence",
)
HEADER_LISTS = ("job_titles", "emails", "phones", "urls")
EXPERIENCE_FIELDS = ("job_titles", "companies", "dates", "locations")
EDUCATION_FIELDS = ("titles", "institutions", "dates")

# Every v1 destination except header_profile, which is scored on its own.
ROUTED_SECTIONS = (
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

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍﻿­"), None)

# Below this length a substring match is coincidence rather than agreement.
_MINIMUM_CONTAINMENT_LENGTH = 6


def normalize(value: Any) -> str:
    """Fold a text span to a form that ignores cosmetic source differences."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.translate(_INVISIBLE).translate(_DASHES)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t.,;:|-•·").casefold()


def values_match(expected: str, actual: str) -> bool:
    """Exact after normalization, or one span fully contains the other.

    Containment is allowed because the pipeline preserves original source text:
    a label of 'Thai citizen' and an extraction of 'Thai' describe the same
    evidence. The length guard stops short fragments matching by accident.
    """
    if not expected or not actual:
        return False
    if expected == actual:
        return True
    shorter, longer = sorted((expected, actual), key=len)
    return len(shorter) >= _MINIMUM_CONTAINMENT_LENGTH and shorter in longer


@dataclass
class Counts:
    """True/false positives and negatives for one scored field."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    misses: list[str] = field(default_factory=list)
    spurious: list[str] = field(default_factory=list)

    def add(self, other: "Counts") -> None:
        self.true_positives += other.true_positives
        self.false_positives += other.false_positives
        self.false_negatives += other.false_negatives
        self.misses.extend(other.misses)
        self.spurious.extend(other.spurious)

    @property
    def precision(self) -> float:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 1.0

    @property
    def recall(self) -> float:
        expected = self.true_positives + self.false_negatives
        return self.true_positives / expected if expected else 1.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        total = precision + recall
        return 2 * precision * recall / total if total else 0.0

    @property
    def supported(self) -> bool:
        """Whether this field was exercised at all."""
        return bool(self.true_positives or self.false_positives or self.false_negatives)


def score_sets(expected: Iterable[Any], actual: Iterable[Any]) -> Counts:
    """Greedy one-to-one matching between two bags of text spans."""
    expected_values = [text for item in expected if (text := normalize(item))]
    actual_values = [text for item in actual if (text := normalize(item))]
    unclaimed = list(actual_values)
    counts = Counts()
    for wanted in expected_values:
        hit = next(
            (candidate for candidate in unclaimed if values_match(wanted, candidate)),
            None,
        )
        if hit is None:
            counts.false_negatives += 1
            counts.misses.append(wanted)
        else:
            unclaimed.remove(hit)
            counts.true_positives += 1
    counts.false_positives += len(unclaimed)
    counts.spurious.extend(unclaimed)
    return counts


def score_scalar(expected: Any, actual: Any) -> Counts:
    """A scalar field contributes at most one prediction and one expectation."""
    return score_sets(
        [expected] if expected is not None else [],
        [actual] if actual is not None else [],
    )


def _url_values(urls: Iterable[Any]) -> list[str]:
    """Output URLs are {text, url} pairs; either side may carry the evidence."""
    values: list[str] = []
    for item in urls:
        if isinstance(item, dict):
            values.append(str(item.get("url") or item.get("text") or ""))
        else:
            values.append(str(item))
    return values


def _section_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, dict, str)):
        return bool(value)
    return True


def _flatten(entries: Iterable[dict[str, Any]], key: str) -> list[Any]:
    return [item for entry in entries for item in entry.get(key, [])]


def score_resume(label: dict[str, Any], result: dict[str, Any]) -> dict[str, Counts]:
    """Score one extracted resume against its hand-written label."""
    scores: dict[str, Counts] = {}
    expected_profile = label.get("header_profile", {})
    actual_profile = result.get("header_profile") or {}

    for name in HEADER_SCALARS:
        scores[f"header.{name}"] = score_scalar(
            expected_profile.get(name),
            actual_profile.get(name),
        )
    for name in HEADER_LISTS:
        expected = expected_profile.get(name, [])
        actual = actual_profile.get(name, []) or []
        if name == "urls":
            expected, actual = _url_values(expected), _url_values(actual)
        scores[f"header.{name}"] = score_sets(expected, actual)

    scores["section_routing"] = score_sets(
        label.get("sections_present", []),
        [name for name in ROUTED_SECTIONS if _section_is_present(result.get(name))],
    )

    for section, fields in (
        ("experience", EXPERIENCE_FIELDS),
        ("education", EDUCATION_FIELDS),
    ):
        expected_entries = label.get(section, [])
        actual_entries = result.get(section) or []
        for name in fields:
            scores[f"{section}.{name}"] = score_sets(
                _flatten(expected_entries, name),
                _flatten(actual_entries, name),
            )
        # An entry-count mismatch means records were split or merged wrongly,
        # which set-based field scores alone would hide.
        scores[f"{section}.entry_count"] = score_sets(
            [f"entry-{index}" for index in range(len(expected_entries))],
            [f"entry-{index}" for index in range(len(actual_entries))],
        )
    return scores


def collect_cases() -> list[tuple[str, Path, Path]]:
    """Committed synthetic fixtures, plus local-only real CVs when present."""
    cases = [
        (stem, fixture_path(stem), label_path(stem))
        for stem in synthetic_stems()
        if label_path(stem).exists()
    ]
    if TRUTHS_LABEL_DIRECTORY.is_dir():
        for path in sorted(TRUTHS_LABEL_DIRECTORY.glob("*.json")):
            pdf = TRUTHS_DIRECTORY / f"{path.stem}.pdf"
            if pdf.exists():
                cases.append((path.stem, pdf, path))
    return cases


@dataclass(frozen=True)
class _Models:
    embedding: Any
    ner: Any


def load_models() -> Any:
    from restruct.model import load_embedding_model, load_ner_model

    return _Models(
        embedding=load_embedding_model(PROJECT_ROOT),
        ner=load_ner_model(PROJECT_ROOT),
    )


def evaluate(
    models: Any | None = None,
) -> tuple[dict[str, Counts], dict[str, dict[str, Counts]], list[str]]:
    """Run every labelled case and aggregate per-field counts.

    ``models`` lets a caller reuse already-loaded weights; loading them costs
    several seconds, so pytest passes its session-scoped pair.
    """
    models = models or load_models()

    totals: dict[str, Counts] = {}
    per_resume: dict[str, dict[str, Counts]] = {}
    skipped: list[str] = []
    with tempfile.TemporaryDirectory(prefix="restruct-scorecard-") as name:
        workspace = Path(name)
        for stem, pdf, labels in collect_cases():
            if stem in OCR_STEMS and not tesseract_available():
                skipped.append(stem)
                continue
            label = json.loads(labels.read_text(encoding="utf-8"))
            result = run_pipeline(pdf, workspace, models)
            scores = score_resume(label, result)
            per_resume[stem] = scores
            for name_, counts in scores.items():
                # Tag every example with its resume so the report says where
                # to look, not just that something is wrong somewhere.
                totals.setdefault(name_, Counts()).add(
                    Counts(
                        true_positives=counts.true_positives,
                        false_positives=counts.false_positives,
                        false_negatives=counts.false_negatives,
                        misses=[f"{stem}: {value}" for value in counts.misses],
                        spurious=[f"{stem}: {value}" for value in counts.spurious],
                    )
                )
    return totals, per_resume, skipped


def render_report(
    totals: dict[str, Counts],
    per_resume: dict[str, dict[str, Counts]],
    skipped: list[str],
) -> str:
    lines = [
        "# Restruct accuracy scorecard",
        "",
        "Scored against the hand-written labels in `tests/labels/`, which were",
        "derived by reading each resume rather than from pipeline output. A span",
        "counts as correct when it matches after normalization, or when one side",
        "fully contains the other.",
        "",
        f"Resumes scored: {len(per_resume)}"
        + (f" (skipped, no tesseract: {', '.join(skipped)})" if skipped else ""),
        "",
        "| Field | Precision | Recall | F1 | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, counts in totals.items():
        if not counts.supported:
            continue
        lines.append(
            f"| `{name}` | {counts.precision:.2f} | {counts.recall:.2f} | "
            f"{counts.f1:.2f} | {counts.true_positives} | "
            f"{counts.false_positives} | {counts.false_negatives} |"
        )

    macro = [counts.f1 for counts in totals.values() if counts.supported]
    lines += ["", f"**Macro F1 across scored fields: {sum(macro) / len(macro):.3f}**", ""]

    lines += ["## Misses and spurious values", ""]
    for name, counts in totals.items():
        if not (counts.misses or counts.spurious):
            continue
        lines.append(f"- `{name}`")
        for value in counts.misses:
            lines.append(f"  - missed: `{value}`")
        for value in counts.spurious:
            lines.append(f"  - spurious: `{value}`")
    return "\n".join(lines) + "\n"


def baseline_scores(totals: dict[str, Counts]) -> dict[str, float]:
    return {
        name: round(counts.f1, 6)
        for name, counts in totals.items()
        if counts.supported
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Freeze the current per-field F1 scores as the regression floor.",
    )
    arguments = parser.parse_args(argv)

    if not models_available():
        print("local models/ weights are absent; see README for setup", file=sys.stderr)
        return 2

    totals, per_resume, skipped = evaluate()
    report = render_report(totals, per_resume, skipped)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    if arguments.update_baseline:
        BASELINE_PATH.write_text(
            json.dumps(baseline_scores(totals), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"baseline written to {BASELINE_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
