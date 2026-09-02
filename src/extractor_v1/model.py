"""Local NER and MiniLM inference for resume extraction."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from extractor_v1.configs import SETTINGS


@dataclass(frozen=True)
class ExtractedLine:
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    bold: bool
    used_ocr: bool


@dataclass(frozen=True)
class DetectedHeading:
    line_index: int
    section_type: str
    similarity: float
    runner_up_similarity: float


@dataclass(frozen=True)
class HeaderEntityMatch:
    kind: str
    text: str
    line_index: int
    start: int
    end: int
    detection_method: str
    confidence: float | None = None
    url: str | None = None
    bbox: tuple[float, float, float, float] | None = None


class EmbeddingModel(Protocol):
    def encode(self, sentences: Any, **kwargs: Any) -> Any: ...


class DistilBertNerPredictor:
    """Adapt fixed CoNLL entities to the resume header entity interface."""

    _LABEL_TO_TYPE = {
        "LABEL_0": "O",
        "LABEL_1": "PER",
        "LABEL_2": "PER",
        "LABEL_3": "ORG",
        "LABEL_4": "ORG",
        "LABEL_5": "LOC",
        "LABEL_6": "LOC",
        "LABEL_7": "MISC",
        "LABEL_8": "MISC",
    }
    _TYPE_TO_LABEL = {
        "PER": "person name",
        "ORG": "organization",
        "LOC": "location",
        "MISC": "nationality",
    }

    def __init__(self, model_directory: Path) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_directory,
            local_files_only=True,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_directory,
            local_files_only=True,
        )
        self._pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=-1,
        )

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float,
    ) -> list[dict[str, Any]]:
        requested_labels = set(labels)
        predictions: list[dict[str, Any]] = []
        for prediction in self._pipeline(text):
            raw_type = str(
                prediction.get("entity_group", prediction.get("entity", ""))
            ).upper()
            entity_type = self._LABEL_TO_TYPE.get(
                raw_type,
                raw_type.removeprefix("B-").removeprefix("I-"),
            )
            label = self._TYPE_TO_LABEL.get(entity_type)
            score = float(prediction.get("score", 0.0))
            if label not in requested_labels or score < threshold:
                continue
            start = int(prediction.get("start", 0))
            end = int(prediction.get("end", 0))
            predictions.append(
                {
                    "label": label,
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "score": score,
                }
            )
        return predictions


def _require_local_model(project_root: Path, relative_directory: str) -> Path:
    model_directory = project_root / relative_directory
    if not model_directory.is_dir():
        raise FileNotFoundError(
            f"local model directory is missing: {model_directory}"
        )
    return model_directory


def load_embedding_model(project_root: Path) -> SentenceTransformer:
    return SentenceTransformer(
        str(_require_local_model(project_root, SETTINGS.model.local_directory))
    )


def load_ner_model(project_root: Path) -> DistilBertNerPredictor:
    model_directory = _require_local_model(
        project_root,
        SETTINGS.ner.distilbert_local_directory,
    )
    return DistilBertNerPredictor(model_directory)


def _looks_like_heading(line: ExtractedLine, page_median_size: float) -> bool:
    text = line.text.strip()
    words = text.split()
    if not 1 <= len(words) <= SETTINGS.heading.maximum_words:
        return False
    if len(text) > SETTINGS.heading.maximum_characters:
        return False
    if "@" in text or text.startswith(("http://", "https://", "www.")):
        return False
    if text.endswith((".", ",", ";", ":")):
        return False

    letters = [character for character in text if character.isalpha()]
    uppercase = (
        bool(letters)
        and sum(character.isupper() for character in letters) / len(letters)
        >= SETTINGS.heading.uppercase_ratio
    )
    larger = (
        page_median_size > 0
        and line.size >= page_median_size * SETTINGS.heading.font_size_multiplier
    )
    return line.bold or uppercase or larger


def detect_headings(
    lines: list[ExtractedLine],
    model: EmbeddingModel,
) -> list[DetectedHeading]:
    """Classify conservative visual candidates with MiniLM."""
    sizes_by_page: dict[int, list[float]] = {}
    for line in lines:
        if line.size > 0:
            sizes_by_page.setdefault(line.page, []).append(line.size)
    median_by_page = {
        page: statistics.median(sizes) for page, sizes in sizes_by_page.items()
    }
    candidate_indexes = [
        index
        for index, line in enumerate(lines)
        if _looks_like_heading(line, median_by_page.get(line.page, line.size))
    ]
    if not candidate_indexes:
        return []

    reference_texts: list[str] = []
    reference_types: list[str] = []
    for section_type, examples in SETTINGS.section_references.items():
        for example in examples:
            reference_texts.append(example)
            reference_types.append(section_type)

    reference_embeddings = model.encode(reference_texts, normalize_embeddings=True)
    candidate_embeddings = model.encode(
        [lines[index].text for index in candidate_indexes],
        normalize_embeddings=True,
    )
    accepted: list[DetectedHeading] = []
    for line_index, candidate_embedding in zip(
        candidate_indexes,
        candidate_embeddings,
        strict=True,
    ):
        raw_scores = candidate_embedding @ reference_embeddings.T
        best_by_type: dict[str, float] = {}
        for section_type, score in zip(reference_types, raw_scores, strict=True):
            best_by_type[section_type] = max(
                best_by_type.get(section_type, -1.0),
                float(score),
            )
        ranked = sorted(best_by_type.items(), key=lambda item: item[1], reverse=True)
        (winner_type, winner_score), (_, runner_up_score) = ranked[:2]
        if winner_score < SETTINGS.heading.similarity_threshold:
            continue
        if winner_score - runner_up_score < SETTINGS.heading.winner_margin:
            continue
        accepted.append(
            DetectedHeading(
                line_index=line_index,
                section_type=winner_type,
                similarity=winner_score,
                runner_up_similarity=runner_up_score,
            )
        )
    return accepted


_JOB_TITLE_SEPARATOR_RE = re.compile(r"[|•·]|\s+/\s+")
HEADER_SEGMENT_RE = re.compile(r"[^|•·]+")
LOCATION_SEGMENT_RE = re.compile(
    r"^[^\d@]{2,48},\s*[^\d@]{2,48}$",
    re.UNICODE,
)
NATIONALITY_CONTEXT_RE = re.compile(
    r"\b(?:citizen|citizenship|national|nationality)\b",
    re.IGNORECASE,
)
NATIONALITY_PHRASE_RE = re.compile(
    r"(?:"
    r"\b[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*)?\s+"
    r"(?:citizen|national)\b"
    r"|\b(?:citizenship|nationality)\s*:\s*"
    r"[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*)?\b"
    r")",
    re.IGNORECASE,
)


def overlaps_existing(
    matches: list[HeaderEntityMatch],
    *,
    line_index: int,
    start: int,
    end: int,
) -> bool:
    return any(
        match.line_index == line_index
        and match.start < end
        and start < match.end
        for match in matches
    )


def _expand_location_span(text: str, start: int, end: int) -> tuple[int, int]:
    for segment_match in HEADER_SEGMENT_RE.finditer(text):
        if not (segment_match.start() <= start and end <= segment_match.end()):
            continue
        segment = segment_match.group(0)
        stripped = segment.strip(" \t,;:-\u200b")
        if not stripped or not LOCATION_SEGMENT_RE.fullmatch(stripped):
            return start, end
        relative_start = segment.find(stripped)
        return (
            segment_match.start() + relative_start,
            segment_match.start() + relative_start + len(stripped),
        )
    return start, end


def _expand_nationality_span(text: str, start: int, end: int) -> tuple[int, int]:
    for segment_match in HEADER_SEGMENT_RE.finditer(text):
        if not (segment_match.start() <= start and end <= segment_match.end()):
            continue
        segment = segment_match.group(0)
        if NATIONALITY_CONTEXT_RE.search(segment) is None:
            return start, end
        stripped = segment.strip(" \t,;:-\u200b")
        relative_start = segment.find(stripped)
        return (
            segment_match.start() + relative_start,
            segment_match.start() + relative_start + len(stripped),
        )
    return start, end


def _masked_line_for_ner(
    text: str,
    line_index: int,
    existing_matches: list[HeaderEntityMatch],
) -> str:
    characters = list(text)
    for match in existing_matches:
        if match.line_index != line_index:
            continue
        if match.kind not in {"email", "phone", "url"}:
            continue
        for position in range(max(0, match.start), min(len(characters), match.end)):
            characters[position] = " "
    return "".join(characters)


def _looks_like_complete_name_line(text: str) -> bool:
    stripped = text.strip()
    return (
        1 <= len(stripped.split()) <= 6
        and len(stripped) <= 80
        and any(character.isalpha() for character in stripped)
        and not any(character.isdigit() for character in stripped)
        and not any(separator in stripped for separator in ("@", "|", "•", "·"))
    )


def ner_matches_for_profile(
    ner_model: DistilBertNerPredictor,
    lines: list[ExtractedLine],
    profile_indexes: list[int],
    existing_matches: list[HeaderEntityMatch],
) -> list[HeaderEntityMatch]:
    """Run DistilBERT on each contact-masked header line."""
    label_to_kind = {
        "person name": "name",
        "location": "location",
        "nationality": "nationality",
    }
    accepted: list[HeaderEntityMatch] = []
    for line_index in profile_indexes:
        text = lines[line_index].text
        masked_text = _masked_line_for_ner(text, line_index, existing_matches)
        if not masked_text.strip():
            continue
        predictions = ner_model.predict_entities(
            masked_text,
            list(SETTINGS.ner.labels),
            threshold=SETTINGS.ner.minimum_confidence,
        )
        for prediction in predictions:
            kind = label_to_kind.get(str(prediction.get("label", "")).casefold())
            if kind is None:
                continue
            start = max(0, int(prediction.get("start", 0)))
            end = min(len(text), int(prediction.get("end", start)))
            if start >= end or overlaps_existing(
                existing_matches,
                line_index=line_index,
                start=start,
                end=end,
            ):
                continue
            if kind == "location":
                start, end = _expand_location_span(text, start, end)
            elif kind == "nationality":
                start, end = _expand_nationality_span(text, start, end)
            elif kind == "name" and _looks_like_complete_name_line(text):
                stripped_line = text.strip()
                start = text.find(stripped_line)
                end = start + len(stripped_line)
            entity_text = text[start:end].strip()
            if not entity_text:
                continue
            start += len(text[start:end]) - len(text[start:end].lstrip())
            end = start + len(entity_text)
            accepted.append(
                HeaderEntityMatch(
                    kind=kind,
                    text=entity_text,
                    line_index=line_index,
                    start=start,
                    end=end,
                    detection_method="distilbert_ner",
                    confidence=float(prediction.get("score", 0.0)),
                )
            )

    locations = [match for match in accepted if match.kind == "location"]
    return [
        match
        for match in accepted
        if match.kind != "nationality"
        or (
            NATIONALITY_CONTEXT_RE.search(lines[match.line_index].text) is not None
            and not overlaps_existing(
                locations,
                line_index=match.line_index,
                start=match.start,
                end=match.end,
            )
        )
    ]


def _job_title_segments(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for separator in _JOB_TITLE_SEPARATOR_RE.finditer(text):
        spans.append((cursor, separator.start()))
        cursor = separator.end()
    spans.append((cursor, len(text)))

    segments: list[tuple[str, int, int]] = []
    for raw_start, raw_end in spans:
        raw_segment = text[raw_start:raw_end]
        stripped = raw_segment.strip(" \t,;:-\u200b")
        if not stripped:
            continue
        start = raw_start + raw_segment.find(stripped)
        segments.append((stripped, start, start + len(stripped)))
        if len(segments) >= SETTINGS.job_title.maximum_segments_per_line:
            break
    return segments


def semantic_job_title_matches(
    model: EmbeddingModel,
    lines: list[ExtractedLine],
    profile_indexes: list[int],
    existing_matches: list[HeaderEntityMatch],
) -> list[HeaderEntityMatch]:
    """Classify unmatched header segments as titles while preserving source text."""
    title_references = sorted(
        {
            reference
            for references in SETTINGS.job_title_references.values()
            for reference in references
        },
        key=len,
        reverse=True,
    )
    phrase_matches: list[HeaderEntityMatch] = []
    candidates: list[HeaderEntityMatch] = []
    for line_index in profile_indexes:
        text = lines[line_index].text
        for segment, start, end in _job_title_segments(text):
            if overlaps_existing(
                existing_matches,
                line_index=line_index,
                start=start,
                end=end,
            ):
                continue
            for reference in title_references:
                for phrase in re.finditer(
                    rf"(?<!\w){re.escape(reference)}(?!\w)",
                    segment,
                    re.IGNORECASE,
                ):
                    phrase_start = start + phrase.start()
                    phrase_end = start + phrase.end()
                    if overlaps_existing(
                        phrase_matches,
                        line_index=line_index,
                        start=phrase_start,
                        end=phrase_end,
                    ):
                        continue
                    phrase_matches.append(
                        HeaderEntityMatch(
                            kind="job_title",
                            text=text[phrase_start:phrase_end],
                            line_index=line_index,
                            start=phrase_start,
                            end=phrase_end,
                            detection_method="job_title_phrase",
                        )
                    )
            if not 1 <= len(segment.split()) <= SETTINGS.job_title.maximum_words:
                continue
            if not any(character.isalpha() for character in segment):
                continue
            if segment.endswith((".", ";")):
                continue
            candidates.append(
                HeaderEntityMatch(
                    kind="job_title",
                    text=segment,
                    line_index=line_index,
                    start=start,
                    end=end,
                    detection_method="semantic_similarity",
                )
            )

    if not candidates:
        return phrase_matches

    positive_embeddings = model.encode(
        title_references,
        normalize_embeddings=True,
    )
    negative_references = [
        reference
        for references in SETTINGS.job_title_negative_references.values()
        for reference in references
    ]
    negative_embeddings = model.encode(
        negative_references,
        normalize_embeddings=True,
    )
    candidate_embeddings = model.encode(
        [candidate.text for candidate in candidates],
        normalize_embeddings=True,
    )

    accepted: list[HeaderEntityMatch] = []
    for candidate, embedding in zip(candidates, candidate_embeddings, strict=True):
        positive_score = float((embedding @ positive_embeddings.T).max())
        negative_score = float((embedding @ negative_embeddings.T).max())
        if positive_score < SETTINGS.job_title.similarity_threshold:
            continue
        if positive_score - negative_score < SETTINGS.job_title.winner_margin:
            continue
        accepted.append(
            HeaderEntityMatch(
                kind=candidate.kind,
                text=candidate.text,
                line_index=candidate.line_index,
                start=candidate.start,
                end=candidate.end,
                detection_method=candidate.detection_method,
                confidence=positive_score,
            )
        )
    for phrase_match in phrase_matches:
        if overlaps_existing(
            accepted,
            line_index=phrase_match.line_index,
            start=phrase_match.start,
            end=phrase_match.end,
        ):
            continue
        accepted.append(phrase_match)
    return accepted


def classify_job_title_candidates(
    model: EmbeddingModel,
    candidates: list[str],
) -> list[tuple[bool, float]]:
    """Classify short experience metadata segments as possible job titles."""
    if not candidates:
        return []
    positive_references = [
        reference
        for references in SETTINGS.job_title_references.values()
        for reference in references
    ]
    negative_references = [
        reference
        for references in SETTINGS.job_title_negative_references.values()
        for reference in references
    ]
    positive_embeddings = model.encode(
        positive_references,
        normalize_embeddings=True,
    )
    negative_embeddings = model.encode(
        negative_references,
        normalize_embeddings=True,
    )
    candidate_embeddings = model.encode(candidates, normalize_embeddings=True)
    classified: list[tuple[bool, float]] = []
    for embedding in candidate_embeddings:
        positive_score = float((embedding @ positive_embeddings.T).max())
        negative_score = float((embedding @ negative_embeddings.T).max())
        classified.append(
            (
                positive_score >= SETTINGS.job_title.similarity_threshold
                and positive_score - negative_score
                >= SETTINGS.job_title.winner_margin,
                positive_score,
            )
        )
    return classified
