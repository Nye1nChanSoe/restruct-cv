"""Guards on the one part of restruct that uses the network.

Nothing here talks to Hugging Face. ``RESTRUCT_MODELS_ENDPOINT`` points the
installer at a local server serving a tree laid out like the real repositories,
and the manifest is replaced with small files whose digests are computed from
what that server holds. What is being tested is the installer's behaviour --
what it verifies, what it refuses, what it leaves behind on failure -- not that
a remote host is up.

The real manifest is checked separately, and only for shape: a digest cannot be
confirmed without downloading 350 MB, but a typo in a path or a stale revision
is worth catching without one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import http.server
import threading
from pathlib import Path

import pytest

from restruct import cli, install
from restruct.errors import ModelDownloadFailed


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A local HTTP server rooted at a directory the test can write into."""
    root = tmp_path / "remote"
    root.mkdir()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *arguments, **keywords) -> None:
            super().__init__(*arguments, directory=str(root), **keywords)

        def log_message(self, *arguments) -> None:  # keep the test output quiet
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        install._HUGGINGFACE_ENDPOINT_VARIABLE,
        f"http://127.0.0.1:{server.server_address[1]}",
    )
    try:
        yield root
    finally:
        server.shutdown()
        server.server_close()


def _publish(root: Path, contents: dict[str, bytes]) -> tuple[install.ModelAsset, ...]:
    """Write files into the served tree and describe them as a manifest."""
    files = []
    for remote_path, payload in contents.items():
        path = root / "owner" / "model" / "resolve" / "abc123" / remote_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files.append(
            install.RemoteFile(
                remote_path,
                Path(remote_path).name,
                _digest(payload),
                len(payload),
            )
        )
    return (
        install.ModelAsset(
            local_directory="test-model",
            repository="owner/model",
            revision="abc123",
            files=tuple(files),
        ),
    )


def test_an_install_writes_the_files_the_loaders_open(
    served: Path, tmp_path: Path
) -> None:
    assets = _publish(served, {"onnx/model.onnx": b"weights", "tokenizer.json": b"{}"})
    destination = tmp_path / "models"

    install.install_models(destination, assets=assets)

    # The export is published under onnx/ and read beside the tokenizer, so the
    # remote path and the local name are deliberately not the same.
    assert (destination / "test-model" / "model.onnx").read_bytes() == b"weights"
    assert (destination / "test-model" / "tokenizer.json").read_bytes() == b"{}"


def test_a_file_that_arrives_wrong_is_not_left_on_disk(
    served: Path, tmp_path: Path
) -> None:
    """A wrong file that stays under the right name is worse than no file: the
    next run finds something loadable and fails somewhere else entirely."""
    assets = _publish(served, {"onnx/model.onnx": b"weights"})
    tampered = dataclasses.replace(
        assets[0],
        files=(install.RemoteFile("onnx/model.onnx", "model.onnx", "0" * 64, 7),),
    )
    destination = tmp_path / "models"

    with pytest.raises(ModelDownloadFailed) as raised:
        install.install_models(destination, assets=(tampered,))

    assert "checksum" in str(raised.value)
    assert not (destination / "test-model" / "model.onnx").exists()
    # Nor the staging file it was written to.
    assert list((destination / "test-model").glob("*.part")) == []


def test_a_file_that_never_arrives_is_reported_with_its_url(
    served: Path, tmp_path: Path
) -> None:
    assets = _publish(served, {"onnx/model.onnx": b"weights"})
    absent = dataclasses.replace(assets[0], revision="no-such-revision")

    with pytest.raises(ModelDownloadFailed) as raised:
        install.install_models(tmp_path / "models", assets=(absent,))

    assert "no-such-revision" in str(raised.value)


def test_a_second_install_re_fetches_nothing(served: Path, tmp_path: Path) -> None:
    """An interrupted install should resume, not restart: 350 MB is long enough
    that a dropped connection part way is a normal thing to happen."""
    assets = _publish(served, {"onnx/model.onnx": b"weights"})
    destination = tmp_path / "models"
    install.install_models(destination, assets=assets)

    fetched: list[str] = []
    install.install_models(
        destination,
        assets=assets,
        on_file=lambda asset, remote, present: fetched.append(remote.local_name)
        if not present
        else None,
    )
    assert fetched == []


def test_a_half_written_file_is_replaced_rather_than_kept(
    served: Path, tmp_path: Path
) -> None:
    """Skipping on the name alone would leave a truncated file there forever,
    which is why presence is decided by the digest and not by the filename."""
    assets = _publish(served, {"onnx/model.onnx": b"weights"})
    destination = tmp_path / "models" / "test-model"
    destination.mkdir(parents=True)
    (destination / "model.onnx").write_bytes(b"wei")

    install.install_models(tmp_path / "models", assets=assets)

    assert (destination / "model.onnx").read_bytes() == b"weights"


def test_missing_assets_names_only_what_is_incomplete(
    served: Path, tmp_path: Path
) -> None:
    assets = _publish(served, {"onnx/model.onnx": b"weights"})
    destination = tmp_path / "models"

    # A directory holding nothing of the real manifest is entirely missing.
    assert install.missing_assets(destination) == install.MODEL_ASSETS

    install.install_models(destination, assets=assets)
    assert all(
        asset.local_directory != "test-model"
        for asset in install.missing_assets(destination)
    )


# -- the real manifest -------------------------------------------------------


def test_the_manifest_names_the_revisions_the_package_uses() -> None:
    """Two sources of truth for the commit would let an install pull weights
    the rest of the package does not think it is running."""
    from restruct.configs import SETTINGS

    by_directory = {asset.local_directory: asset for asset in install.MODEL_ASSETS}
    embedding = by_directory[SETTINGS.model.local_directory]
    assert embedding.repository == SETTINGS.model.name
    assert embedding.revision == SETTINGS.model.revision

    ner = by_directory[SETTINGS.ner.distilbert_local_directory]
    assert ner.repository == SETTINGS.ner.distilbert_name
    assert ner.revision == SETTINGS.ner.distilbert_revision


def test_the_manifest_covers_every_file_a_loader_opens() -> None:
    """A model directory that installs without one of these downloads cleanly
    and then fails at load, reporting a problem two steps from its cause."""
    by_directory = {
        asset.local_directory: {remote.local_name for remote in asset.files}
        for asset in install.MODEL_ASSETS
    }
    assert by_directory["all-MiniLM-L6-v2"] == {
        "model.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
        "sentence_bert_config.json",
    }
    assert by_directory["distilbert-NER"] == {
        "model.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
    }


def test_the_manifest_installs_what_the_search_looks_for() -> None:
    """The install and the search have to agree on the directory names, or a
    successful install is followed by a missing-weights failure."""
    assert {asset.local_directory for asset in install.MODEL_ASSETS} == set(
        cli.MODEL_DIRECTORY_NAMES
    )
    assert all(
        cli.MODEL_WEIGHTS_FILE in {remote.local_name for remote in asset.files}
        for asset in install.MODEL_ASSETS
    )


def test_every_digest_is_a_sha256() -> None:
    for asset in install.MODEL_ASSETS:
        for remote in asset.files:
            assert len(remote.sha256) == 64
            assert set(remote.sha256) <= set("0123456789abcdef")


# -- where an install writes -------------------------------------------------


def test_an_install_goes_where_the_search_will_look(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(cli.MODELS_DIRECTORY_VARIABLE, raising=False)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\n")
    assert cli._install_directory(checkout, "") == checkout / "models"
    assert cli._install_directory(checkout, "") in cli._candidate_model_directories(
        checkout
    )


def test_an_installed_copy_installs_to_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not `./models`: a search may accept weights somebody put in the working
    directory, but writing 350 MB there because that is where the shell was is
    a different decision."""
    monkeypatch.delenv(cli.MODELS_DIRECTORY_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)
    installed = tmp_path / "site-packages-parent"
    installed.mkdir()
    assert cli._install_directory(installed, "") == Path.home() / ".restruct" / "models"


def test_an_explicit_target_settles_where_an_install_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cli.MODELS_DIRECTORY_VARIABLE, str(tmp_path / "environment"))
    assert cli._install_directory(tmp_path, str(tmp_path / "asked")) == (
        tmp_path / "asked"
    )
    assert cli._install_directory(tmp_path, "") == tmp_path / "environment"


# -- the prompt --------------------------------------------------------------


def test_a_non_interactive_run_is_never_offered_a_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A script, a container build or a pipe gets the exit code it has always
    got. Starting a 350 MB transfer nobody is watching is worse than failing
    with an instruction."""
    monkeypatch.delenv(cli.MODELS_DIRECTORY_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False, raising=False)

    called = False

    def _never(*arguments, **keywords):
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "_install_models", _never)
    with pytest.raises(cli.ModelAssetsMissing):
        cli._models(tmp_path / "not-a-checkout")
    assert not called


def test_the_missing_weights_message_names_the_command_that_fixes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(cli.MODELS_DIRECTORY_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(cli.ModelAssetsMissing) as raised:
        cli._load_models(tmp_path / "not-a-checkout")
    assert "--install-models" in str(raised.value)


def test_an_install_is_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--install-models` writes no result, so it must not fall through into
    the batch and start extracting a corpus."""
    installed: list[Path] = []
    monkeypatch.setattr(
        cli, "_install_models", lambda destination: installed.append(destination)
    )
    monkeypatch.setattr(
        cli, "_batch", lambda *arguments, **keywords: pytest.fail("ran the batch")
    )
    assert cli.main(["--install-models", str(tmp_path / "here")]) == cli.EXIT_OK
    assert installed == [tmp_path / "here"]


def test_a_download_failure_has_its_own_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distinct from missing weights: one is a setup step not yet taken, the
    other is worth retrying."""

    def _fail(destination: Path) -> None:
        raise ModelDownloadFailed("https://example.invalid/model.onnx", "no route")

    monkeypatch.setattr(cli, "_install_models", _fail)
    assert cli.main(["--install-models", str(tmp_path)]) == (
        cli.EXIT_MODEL_DOWNLOAD_FAILED
    )
