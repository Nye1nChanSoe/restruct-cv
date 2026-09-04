"""ONNX Runtime adapters for the two local models.

Both models are forward passes over small encoders -- nothing here trains
anything -- so they run on ONNX Runtime rather than torch. That is the whole
of the difference: `torch`, `transformers`, `sentence-transformers`, `scipy`
and `scikit-learn` were about 725 MB of a 932 MB install, and on Linux the
torch wheel resolves to the CUDA build and drags the `nvidia-*` wheels with
it. `onnxruntime` + `tokenizers` + `numpy` is about 40 MB and has no GPU
variant to resolve into by accident.

The weights are the `model.onnx` written into each model directory by
`tools/export_onnx.py`, quantized to int8. Tokenization is the same
`tokenizer.json` the torch path used, read directly by `tokenizers`.

Two behaviours are reimplemented here because they used to come from a
library, and both are the *published* behaviour of the model they belong to:

- MiniLM's sentence embedding is mean pooling over the unmasked tokens
  followed by L2 normalization -- what `modules.json` and `1_Pooling/` in the
  model directory state, read here rather than interpreted by
  `sentence-transformers`.
- The NER pipeline's `aggregation_strategy="simple"`: argmax per token, then
  consecutive tokens of one tag grouped, scored by the mean of their members.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnxruntime
from tokenizers import Tokenizer

from restruct.configs import SETTINGS

# A resume is a handful of short lines, not a training corpus. One thread
# avoids ONNX Runtime spinning up a pool per session for work that finishes
# faster than the pool costs, and keeps a library caller's own parallelism
# intact.
_SESSION_OPTIONS = onnxruntime.SessionOptions()
_SESSION_OPTIONS.intra_op_num_threads = 1
_SESSION_OPTIONS.inter_op_num_threads = 1
_SESSION_OPTIONS.log_severity_level = 3

_DEFAULT_MAXIMUM_TOKENS = 512


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_session(model_directory: Path) -> onnxruntime.InferenceSession:
    weights = model_directory / "model.onnx"
    if not weights.is_file():
        raise FileNotFoundError(
            f"local model is missing its ONNX weights: {weights}"
        )
    return onnxruntime.InferenceSession(
        str(weights),
        sess_options=_SESSION_OPTIONS,
        providers=["CPUExecutionProvider"],
    )


def _load_tokenizer(model_directory: Path, maximum_tokens: int) -> Tokenizer:
    path = model_directory / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"local model is missing its tokenizer: {path}"
        )
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.enable_truncation(max_length=maximum_tokens)
    return tokenizer


def _pad_token(model_directory: Path) -> tuple[str, int]:
    configuration = _read_json(model_directory / "tokenizer_config.json")
    token = configuration.get("pad_token") or "[PAD]"
    if isinstance(token, dict):  # some exports write the full AddedToken
        token = token.get("content", "[PAD]")
    return str(token), 0


class MiniLmEncoder:
    """Sentence embeddings, mean-pooled and L2-normalized, as float32."""

    def __init__(self, model_directory: Path) -> None:
        self._session = _load_session(model_directory)
        sentence_configuration = _read_json(
            model_directory / "sentence_bert_config.json"
        )
        maximum_tokens = int(
            sentence_configuration.get("max_seq_length") or _DEFAULT_MAXIMUM_TOKENS
        )
        self._tokenizer = _load_tokenizer(model_directory, maximum_tokens)
        pad_token, pad_id = _pad_token(model_directory)
        self._tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
        self._input_names = {input.name for input in self._session.get_inputs()}

    def encode(
        self,
        sentences: Any,
        *,
        normalize_embeddings: bool = False,
        **_: Any,
    ) -> np.ndarray:
        """Embed one string or a sequence of them.

        Keyword arguments beyond ``normalize_embeddings`` are accepted and
        ignored: they are ``sentence-transformers`` presentation options
        (progress bars, tensor type) with no meaning here, and swallowing them
        keeps this a drop-in for the ``EmbeddingModel`` protocol.
        """
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
        attention_mask = np.array(
            [encoding.attention_mask for encoding in encodings], dtype=np.int64
        )
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            inputs["token_type_ids"] = np.array(
                [encoding.type_ids for encoding in encodings], dtype=np.int64
            )

        token_embeddings = self._session.run(None, inputs)[0]
        mask = attention_mask.astype(np.float32)[..., None]
        summed = (token_embeddings * mask).sum(axis=1)
        # Clamped the way sentence-transformers clamps it: a line that
        # tokenizes to nothing would otherwise divide by zero.
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = (summed / counts).astype(np.float32)
        if normalize_embeddings:
            norms = np.clip(
                np.linalg.norm(embeddings, axis=1, keepdims=True),
                a_min=1e-12,
                a_max=None,
            )
            embeddings = embeddings / norms
        return embeddings[0] if single else embeddings


def _tag_of(label: str) -> tuple[str, str]:
    """Split a BIO label into its prefix and its tag.

    A label with no prefix -- `O`, or a model that writes bare types -- counts
    as a continuation, so consecutive ones group rather than each starting a
    new entity. That is what the transformers pipeline does, and the grouping
    is what our spans are made of.
    """
    if label.startswith("B-"):
        return "B", label[2:]
    if label.startswith("I-"):
        return "I", label[2:]
    return "I", label


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
        self._session = _load_session(model_directory)
        configuration = _read_json(model_directory / "config.json")
        raw_labels = configuration.get("id2label") or {}
        self._id_to_label = {
            int(identifier): str(label) for identifier, label in raw_labels.items()
        }
        tokenizer_configuration = _read_json(
            model_directory / "tokenizer_config.json"
        )
        maximum_tokens = int(
            tokenizer_configuration.get("model_max_length") or _DEFAULT_MAXIMUM_TOKENS
        )
        self._tokenizer = _load_tokenizer(
            model_directory,
            min(maximum_tokens, _DEFAULT_MAXIMUM_TOKENS),
        )

    def _token_entities(self, text: str) -> list[dict[str, Any]]:
        encoding = self._tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        logits = self._session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0][0]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exponentiated = np.exp(shifted)
        scores = exponentiated / exponentiated.sum(axis=-1, keepdims=True)

        entities: list[dict[str, Any]] = []
        for index, (special, (start, end)) in enumerate(
            zip(encoding.special_tokens_mask, encoding.offsets, strict=True)
        ):
            if special:
                continue
            best = int(scores[index].argmax())
            entities.append(
                {
                    "label": self._id_to_label.get(best, f"LABEL_{best}"),
                    "score": float(scores[index][best]),
                    "start": int(start),
                    "end": int(end),
                }
            )
        return entities

    def _grouped(self, text: str) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        current: list[dict[str, Any]] = []
        for entity in self._token_entities(text):
            if current:
                prefix, tag = _tag_of(entity["label"])
                _, previous_tag = _tag_of(current[-1]["label"])
                if tag == previous_tag and prefix != "B":
                    current.append(entity)
                    continue
                groups.append(_group_of(current))
                current = [entity]
            else:
                current = [entity]
        if current:
            groups.append(_group_of(current))
        return groups

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float,
    ) -> list[dict[str, Any]]:
        requested_labels = set(labels)
        predictions: list[dict[str, Any]] = []
        for group in self._grouped(text):
            raw_type = str(group["entity_group"]).upper()
            entity_type = self._LABEL_TO_TYPE.get(
                raw_type,
                raw_type.removeprefix("B-").removeprefix("I-"),
            )
            label = self._TYPE_TO_LABEL.get(entity_type)
            score = float(group["score"])
            if label not in requested_labels or score < threshold:
                continue
            start = int(group["start"])
            end = int(group["end"])
            predictions.append(
                {
                    "label": label,
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "score": score,
                }
            )
        merged: list[dict[str, Any]] = []
        for prediction in predictions:
            if merged and merged[-1]["label"] == prediction["label"]:
                gap = text[int(merged[-1]["end"]):int(prediction["start"])]
                if len(gap) <= 3 and not any(character.isalnum() for character in gap):
                    merged[-1]["end"] = prediction["end"]
                    merged[-1]["text"] = text[int(merged[-1]["start"]):int(prediction["end"])]
                    merged[-1]["score"] = min(
                        float(merged[-1]["score"]), float(prediction["score"])
                    )
                    continue
            merged.append(prediction)
        return merged


def _group_of(entities: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One entity from a run of tokens, scored by the mean of its members."""
    first = entities[0]
    return {
        "entity_group": str(first["label"]).split("-", 1)[-1],
        "score": float(np.mean([entity["score"] for entity in entities])),
        "start": first["start"],
        "end": entities[-1]["end"],
    }
