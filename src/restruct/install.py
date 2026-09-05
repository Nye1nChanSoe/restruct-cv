"""Fetch the two ONNX model directories from their source repositories.

This is the one place in the package that reaches the network, and it runs
only when a person asks for it -- either `restruct --install-models`, or by
answering the prompt a missing-weights run puts up on a terminal. An
extraction never downloads anything: a batch that quietly pulled 350 MB on a
metered connection is worse than one that says what is missing and stops.

**Nothing here needs torch.** Both source repositories publish an fp32
`onnx/model.onnx` of their own, so the install is four HTTPS gets per model
rather than a re-export. Those published exports were checked against the ones
`tools/export_onnx.py` writes: every golden snapshot is byte-identical and the
scorecard's macro F1 is unchanged at 0.966. That equivalence is what makes
downloading them legitimate rather than convenient -- the graphs differ in
size, and the numbers they produce do not.

Only the files the loaders actually open are fetched. `encoders.py` reads
`model.onnx`, `tokenizer.json`, `tokenizer_config.json` and one config each
(`sentence_bert_config.json` for MiniLM, `config.json` for the NER model); the
safetensors, the training runs and the quantized variants in those repositories
are not read by anything here, so pulling them would cost a few hundred
megabytes to store what nothing loads.

Every file is pinned by revision *and* by SHA-256. The revision says which
commit, and the digest says the bytes arrived intact and unaltered -- a
revision alone trusts the transport and the host to agree with what was
measured here.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from restruct.configs import SETTINGS
from restruct.errors import ModelDownloadFailed

_HUGGINGFACE_ENDPOINT_VARIABLE = "RESTRUCT_MODELS_ENDPOINT"
_DEFAULT_ENDPOINT = "https://huggingface.co"

# Read in 1 MB blocks: large enough that the digest and the progress callback
# are not the cost of the loop, small enough that a stalled connection is
# noticed within a block rather than at the end of a 260 MB file.
_CHUNK_BYTES = 1024 * 1024

_NETWORK_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class RemoteFile:
    """One file to fetch, and what it must turn out to be.

    ``remote_path`` and ``local_name`` differ because both repositories publish
    the export under `onnx/`, while the loaders expect it beside the tokenizer.
    """

    remote_path: str
    local_name: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ModelAsset:
    """One local model directory and where its contents come from."""

    local_directory: str
    repository: str
    revision: str
    files: tuple[RemoteFile, ...]

    @property
    def size(self) -> int:
        return sum(remote.size for remote in self.files)


# The revisions come from SETTINGS rather than being retyped, so the commit an
# install pulls is the same one the rest of the package names.
MODEL_ASSETS: tuple[ModelAsset, ...] = (
    ModelAsset(
        local_directory=SETTINGS.model.local_directory,
        repository=SETTINGS.model.name,
        revision=SETTINGS.model.revision,
        files=(
            RemoteFile(
                "onnx/model.onnx",
                "model.onnx",
                "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
                90405214,
            ),
            RemoteFile(
                "tokenizer.json",
                "tokenizer.json",
                "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
                466247,
            ),
            RemoteFile(
                "tokenizer_config.json",
                "tokenizer_config.json",
                "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
                350,
            ),
            RemoteFile(
                "sentence_bert_config.json",
                "sentence_bert_config.json",
                "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
                53,
            ),
        ),
    ),
    ModelAsset(
        local_directory=SETTINGS.ner.distilbert_local_directory,
        repository=SETTINGS.ner.distilbert_name,
        revision=SETTINGS.ner.distilbert_revision,
        files=(
            RemoteFile(
                "onnx/model.onnx",
                "model.onnx",
                "4440f9fc64cd28ac75d83a38d89716f25947799640cd0e5f1f9f6e57b9c14160",
                260926482,
            ),
            RemoteFile(
                "onnx/tokenizer.json",
                "tokenizer.json",
                "cb26b43c98e8266ae3e99c2a583cf8315d73b33a17e6b20b4df7ff1f22392d34",
                669021,
            ),
            RemoteFile(
                "onnx/tokenizer_config.json",
                "tokenizer_config.json",
                "4391b0abb71cd639e50a333c5c642d3c8659ba34099cb12a83dba2efc26f5451",
                1305,
            ),
            RemoteFile(
                "onnx/config.json",
                "config.json",
                "8f9f01d47f61087197f9fa85185d4a7a6248333c15af1b221aa5e8b9b76462b5",
                925,
            ),
        ),
    ),
)

TOTAL_DOWNLOAD_BYTES = sum(asset.size for asset in MODEL_ASSETS)


def human_size(count: int) -> str:
    """Bytes as the size a person would say out loud."""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f} GB"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.0f} MB"
    if count >= 1_000:
        return f"{count / 1_000:.0f} kB"
    return f"{count} B"


def _endpoint() -> str:
    """The host to fetch from, overridable so tests never touch the network."""
    return os.environ.get(_HUGGINGFACE_ENDPOINT_VARIABLE, _DEFAULT_ENDPOINT).rstrip("/")


def _url(asset: ModelAsset, remote: RemoteFile) -> str:
    return (
        f"{_endpoint()}/{asset.repository}/resolve/"
        f"{asset.revision}/{remote.remote_path}"
    )


def _umask() -> int:
    """The process umask, read the only way Python offers: by setting it."""
    current = os.umask(0)
    os.umask(current)
    return current


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_already_installed(destination: Path, remote: RemoteFile) -> bool:
    """Whether this exact file is on disk, so an interrupted install resumes.

    The digest is what makes that safe: a half-written file from a dropped
    connection has the right name and the wrong contents, and skipping on the
    name alone would leave it there forever.
    """
    return (
        destination.is_file()
        and destination.stat().st_size == remote.size
        and _digest(destination) == remote.sha256
    )


def _download(
    url: str,
    destination: Path,
    remote: RemoteFile,
    on_progress: Callable[[int], None] | None,
) -> None:
    """Fetch one file, verify it, then move it into place.

    Written to a temporary file in the destination directory and renamed only
    after the digest matches, so a failed install never leaves something that
    looks loadable. Same directory because a rename across filesystems is a
    copy, and the copy is the part that can be interrupted.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, staging_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".part"
    )
    staging = Path(staging_name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "restruct"})
        with urllib.request.urlopen(  # noqa: S310 - https, pinned by digest
            request, timeout=_NETWORK_TIMEOUT_SECONDS
        ) as response, os.fdopen(handle, "wb") as output:
            while True:
                block = response.read(_CHUNK_BYTES)
                if not block:
                    break
                output.write(block)
                if on_progress is not None:
                    on_progress(len(block))
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        staging.unlink(missing_ok=True)
        raise ModelDownloadFailed(url, str(error)) from error

    found = _digest(staging)
    if found != remote.sha256:
        staging.unlink(missing_ok=True)
        raise ModelDownloadFailed(
            url,
            f"checksum mismatch: expected {remote.sha256}, got {found}",
        )
    # mkstemp creates at 0600. These are read-only data files that a second
    # user on the machine, or a container running as somebody else, should be
    # able to read, so they get the mode a downloaded file would normally get.
    staging.chmod(0o666 & ~_umask())
    staging.replace(destination)


def missing_assets(models_directory: Path) -> tuple[ModelAsset, ...]:
    """The model directories that are not fully and correctly installed.

    Only names and sizes are checked here, not digests: this runs before every
    install to decide what to fetch, and hashing 350 MB to answer "is anything
    missing" would cost more than the question is worth. The digest still
    guards every byte that is written.
    """
    incomplete = []
    for asset in MODEL_ASSETS:
        directory = models_directory / asset.local_directory
        if any(
            not (directory / remote.local_name).is_file()
            or (directory / remote.local_name).stat().st_size != remote.size
            for remote in asset.files
        ):
            incomplete.append(asset)
    return tuple(incomplete)


def install_models(
    models_directory: Path,
    *,
    assets: Iterable[ModelAsset] | None = None,
    on_progress: Callable[[int], None] | None = None,
    on_file: Callable[[ModelAsset, RemoteFile, bool], None] | None = None,
) -> Path:
    """Download the model files into ``models_directory``, verified.

    Returns the directory, so a caller can name it in a message. Raises
    ``ModelDownloadFailed`` and leaves nothing half-written behind.
    """
    for asset in MODEL_ASSETS if assets is None else assets:
        directory = models_directory / asset.local_directory
        for remote in asset.files:
            destination = directory / remote.local_name
            present = _is_already_installed(destination, remote)
            if on_file is not None:
                on_file(asset, remote, present)
            if present:
                if on_progress is not None:
                    on_progress(remote.size)
                continue
            _download(_url(asset, remote), destination, remote, on_progress)
    return models_directory


def free_space(directory: Path) -> int:
    """Bytes available where the install would land.

    Walks up to the nearest directory that exists, because the target usually
    does not yet -- that is the whole reason the install is running.
    """
    probe = directory
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free
