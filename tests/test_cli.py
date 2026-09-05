"""Guards on the command-line surface.

Two things are being protected. Exit codes are an API -- a script branching on
them breaks silently if one changes -- so each is pinned to its failure. And
``--stages`` selects *artifacts*, never whether a pass runs: the pipeline is a
chain where each pass feeds the next, so a flag that skipped one would quietly
produce a different resume rather than a less-documented one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from restruct import cli
from tests.helpers import SYNTHETIC_DIRECTORY, models_available

NATIVE_FIXTURE = SYNTHETIC_DIRECTORY / "1.pdf"

needs_models = pytest.mark.skipif(
    not models_available(),
    reason="local models/ weights are absent; see README for setup",
)


# -- stage parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("1-5", {1, 2, 3, 4, 5}),
        ("3", {3}),
        ("2,4,5", {2, 4, 5}),
        ("1-3,5", {1, 2, 3, 5}),
        ("5,1", {1, 5}),
        (" 2 , 4 ", {2, 4}),
        ("2-2", {2}),
    ],
)
def test_stage_specs_parse(spec: str, expected: set[int]) -> None:
    assert cli.parse_stages(spec) == expected


@pytest.mark.parametrize("spec", ["0", "6", "1-9", "3-1", "", "a", "1..3", "1-"])
def test_a_bad_stage_spec_is_rejected(spec: str) -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_stages(spec)


# -- what --ats and --stages mean ------------------------------------------


def selected(argv: list[str]) -> frozenset[int]:
    return cli._selected_stages(cli._parse_arguments(argv))


def test_no_flag_writes_no_artifacts() -> None:
    assert selected([str(NATIVE_FIXTURE), "-o", "out.json"]) == frozenset()


def test_debug_alone_means_stages_four_and_five() -> None:
    assert selected([str(NATIVE_FIXTURE), "-o", "out.json", "--ats"]) == {4, 5}


def test_stages_without_debug_enables_debug() -> None:
    """Documented in --help: asking for stages is asking for artifacts, and
    making someone pass two flags to say one thing is a papercut."""
    assert selected([str(NATIVE_FIXTURE), "-o", "out.json", "--stages", "1-3"]) == {
        1,
        2,
        3,
    }


def test_stages_overrides_the_debug_default() -> None:
    assert selected(
        [str(NATIVE_FIXTURE), "-o", "out.json", "--ats", "--stages", "2"]
    ) == {2}


def test_a_path_without_an_output_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli._parse_arguments([str(NATIVE_FIXTURE)])
    assert exit_info.value.code == 2


# -- exit codes --------------------------------------------------------------


def test_a_missing_input_reports_itself(tmp_path: Path) -> None:
    assert (
        cli.main([str(tmp_path / "absent.pdf"), "-o", str(tmp_path / "out.json")])
        == cli.EXIT_INPUT_NOT_FOUND
    )


def test_an_unreadable_format_reports_itself(tmp_path: Path) -> None:
    document = tmp_path / "resume.rtf"
    document.write_bytes(b"{\\rtf1 not a format v1 reads}")
    assert (
        cli.main([str(document), "-o", str(tmp_path / "out.json")])
        == cli.EXIT_UNSUPPORTED_FORMAT
    )


def test_a_corrupt_pdf_reports_itself(tmp_path: Path) -> None:
    """Right extension, wrong contents. Distinct from an unsupported format,
    because the remedy is different: one is a conversion, one is a re-export."""
    document = tmp_path / "broken.pdf"
    document.write_bytes(b"%PDF-1.7\nthis is not a pdf body")
    assert (
        cli.main([str(document), "-o", str(tmp_path / "out.json")])
        == cli.EXIT_INVALID_DOCUMENT
    )


def test_an_empty_file_with_a_pdf_extension_reports_itself(tmp_path: Path) -> None:
    document = tmp_path / "empty.pdf"
    document.write_bytes(b"")
    assert (
        cli.main([str(document), "-o", str(tmp_path / "out.json")])
        == cli.EXIT_INVALID_DOCUMENT
    )


def test_missing_model_weights_report_themselves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weights are local-only and never fetched at run time, so this is a
    setup problem with a specific fix rather than something to retry."""
    monkeypatch.setattr(
        cli,
        "_load_models",
        lambda root: (_ for _ in ()).throw(
            cli.ModelAssetsMissing(tmp_path / "models" / "all-MiniLM-L6-v2")
        ),
    )
    assert (
        cli.main([str(NATIVE_FIXTURE), "-o", str(tmp_path / "out.json")])
        == cli.EXIT_MODEL_ASSETS_MISSING
    )


# -- --reconstruct -----------------------------------------------------------


@needs_models
def test_reconstruct_draws_the_result_beside_it(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    assert cli.main([str(NATIVE_FIXTURE), "-o", str(output), "--reconstruct"]) == cli.EXIT_OK
    drawn = tmp_path / "out" / "reconstruction"
    assert (drawn / "reconstruction.pdf").is_file()
    assert (drawn / "page-1.png").is_file()


@needs_models
def test_without_the_flag_nothing_is_drawn(tmp_path: Path) -> None:
    """Quiet on success means quiet: a run that was not asked for a drawing
    leaves no drawing."""
    output = tmp_path / "out.json"
    assert cli.main([str(NATIVE_FIXTURE), "-o", str(output)]) == cli.EXIT_OK
    assert not (tmp_path / "out" / "reconstruction").exists()


def test_an_existing_result_is_drawn_without_loading_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drawing needs no models and no source document, so a result from last
    week can be looked at without re-running anything."""
    monkeypatch.setattr(
        cli,
        "_load_models",
        lambda root: pytest.fail("drawing a result must not read a model"),
    )
    resume_path = tmp_path / "resume.json"
    resume_path.write_text(
        json.dumps({"schema_version": "1.0", "header_profile": {"name": "Somchai"}}),
        encoding="utf-8",
    )
    assert cli.main([str(resume_path), "--reconstruct"]) == cli.EXIT_OK
    # Flat beside the JSON, named after it: a directory holding two files
    # is a directory to open, and the stem is what keeps two resumes drawn
    # into one place from overwriting each other.
    assert (tmp_path / "resume-reconstruction.pdf").is_file()
    assert (tmp_path / "resume-page-1.png").is_file()


def test_o_names_a_directory_for_a_standalone_drawing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run produces a page and not a result, so -o names where to put it,
    and inside a directory of its own the plain names need no qualifying."""
    monkeypatch.setattr(
        cli, "_load_models", lambda root: pytest.fail("drawing must not read a model")
    )
    resume_path = tmp_path / "resume.json"
    resume_path.write_text(
        json.dumps({"schema_version": "1.0", "header_profile": {"name": "Somchai"}}),
        encoding="utf-8",
    )
    drawn = tmp_path / "drawn"
    assert cli.main([str(resume_path), "--reconstruct", "-o", str(drawn)]) == cli.EXIT_OK
    assert (drawn / "reconstruction.pdf").is_file()
    assert (drawn / "page-1.png").is_file()


def test_a_result_that_is_not_there_reports_itself(tmp_path: Path) -> None:
    assert (
        cli.main([str(tmp_path / "absent.json"), "--reconstruct"])
        == cli.EXIT_INPUT_NOT_FOUND
    )


def test_a_json_path_without_the_flag_is_still_an_unsupported_format(
    tmp_path: Path,
) -> None:
    """--reconstruct is what makes a .json input meaningful; without it the
    file is a resume the extractor cannot read."""
    resume_path = tmp_path / "resume.json"
    resume_path.write_text("{}", encoding="utf-8")
    assert (
        cli.main([str(resume_path), "-o", str(tmp_path / "out.json")])
        == cli.EXIT_UNSUPPORTED_FORMAT
    )


# -- where the weights are looked for ----------------------------------------


def _weights(directory: Path) -> Path:
    """A directory shaped like a populated models/ directory."""
    for name in cli.MODEL_DIRECTORY_NAMES:
        (directory / name).mkdir(parents=True)
        (directory / name / cli.MODEL_WEIGHTS_FILE).write_text("")
    return directory


def test_a_directory_without_onnx_weights_is_not_a_models_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory left over from the torch era holds safetensors and no
    `model.onnx`. Accepting it because it is non-empty would let the run get
    as far as loading before failing, and would report the wrong problem."""
    monkeypatch.delenv(cli.MODELS_DIRECTORY_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)
    for name in cli.MODEL_DIRECTORY_NAMES:
        (tmp_path / "models" / name).mkdir(parents=True)
        (tmp_path / "models" / name / "model.safetensors").write_text("")
    with pytest.raises(cli.ModelAssetsMissing):
        cli._load_models(tmp_path / "not-a-checkout")


def test_the_checkout_is_only_a_candidate_when_it_is_a_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In an installed copy, `parents[2]` is site-packages/.. -- a directory
    nobody puts weights in. Offering it would send the reader to set up their
    environment in the wrong place."""
    monkeypatch.delenv(cli.MODELS_DIRECTORY_VARIABLE, raising=False)
    installed = tmp_path / "site-packages-parent"
    installed.mkdir()
    assert installed / "models" not in cli._candidate_model_directories(installed)

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\n")
    assert cli._candidate_model_directories(checkout)[0] == checkout / "models"


def test_an_explicit_directory_settles_the_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Someone who has said where the weights are should not have the tool
    quietly find a different set somewhere else."""
    monkeypatch.setenv(cli.MODELS_DIRECTORY_VARIABLE, str(tmp_path / "elsewhere"))
    assert cli._candidate_model_directories(tmp_path) == [tmp_path / "elsewhere"]


def test_the_working_directory_is_searched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installed-copy case: no checkout to fall back on, and weights kept
    beside the resumes being read."""
    monkeypatch.delenv(cli.MODELS_DIRECTORY_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)
    _weights(tmp_path / "models")
    assert cli._models_directory(tmp_path / "not-a-checkout") == tmp_path / "models"


def test_the_failure_names_every_place_that_was_looked_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader's next question is always "where does it want them?", and
    for an installed copy the answer is not obvious."""
    monkeypatch.delenv(cli.MODELS_DIRECTORY_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(cli.ModelAssetsMissing) as raised:
        cli._load_models(tmp_path / "not-a-checkout")
    message = str(raised.value)
    assert str(tmp_path / "models") in message
    assert str(Path.home() / ".restruct" / "models") in message


def test_every_error_has_its_own_exit_code() -> None:
    """A script branching on these breaks silently if two ever collide."""
    codes = [code for _, code in cli._EXIT_CODES]
    assert len(codes) == len(set(codes))
    assert cli.EXIT_OK not in codes
    assert cli.EXIT_UNEXPECTED not in codes
    assert 2 not in codes, "2 belongs to argparse's usage error"


# -- end to end --------------------------------------------------------------


@needs_models
def test_extracting_one_resume_is_quiet_and_writes_only_the_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "out.json"
    assert cli.main([str(NATIVE_FIXTURE), "-o", str(output)]) == cli.EXIT_OK

    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert [path.name for path in tmp_path.iterdir()] == ["out.json"]
    assert json.loads(output.read_text(encoding="utf-8"))["header_profile"]["name"]


@needs_models
@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (["--ats"], {"pass-4-sections"}),
        (["--stages", "1-3"], {"pass-1-physical", "pass-2-words", "pass-3-lines"}),
        (["--stages", "2"], {"pass-2-words"}),
    ],
)
def test_stage_selection_writes_exactly_those_overlays(
    tmp_path: Path,
    flags: list[str],
    expected: set[str],
) -> None:
    output = tmp_path / "out.json"
    assert cli.main([str(NATIVE_FIXTURE), "-o", str(output), *flags]) == cli.EXIT_OK
    debug = tmp_path / "out" / "debug"
    assert {
        child.name for child in debug.iterdir() if child.is_dir()
    } == expected


@needs_models
def test_selecting_stages_never_changes_the_result(
    tmp_path: Path,
) -> None:
    """The pipeline is a chain: every pass feeds the next, so --stages must
    select what is *written*, never what is *run*. If it gated execution this
    comparison would drift."""
    results = []
    for index, flags in enumerate(([], ["--ats"], ["--stages", "1-5"])):
        output = tmp_path / f"out{index}.json"
        assert cli.main([str(NATIVE_FIXTURE), "-o", str(output), *flags]) == cli.EXIT_OK
        results.append(output.read_text(encoding="utf-8"))
    assert results[0] == results[1] == results[2]


@needs_models
def test_stage_five_writes_the_evidence_track_and_the_overlay(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out.json"
    assert (
        cli.main([str(NATIVE_FIXTURE), "-o", str(output), "--stages", "5"])
        == cli.EXIT_OK
    )
    raw = tmp_path / "out" / "raw"
    assert (raw / "headerProfile.json").exists()
    assert (raw / "experience.json").exists()
    assert (tmp_path / "out" / "debug" / "page-1.png").exists()
    # Evidence lives here, never in the result.
    assert "bbox" in (raw / "experience.json").read_text(encoding="utf-8")
    assert "bbox" not in output.read_text(encoding="utf-8")


# -- -o naming a directory ---------------------------------------------------


@pytest.mark.parametrize("spec", [".", "./", "out/", "nested/deep/"])
def test_a_directory_output_takes_its_name_from_the_input(spec: str) -> None:
    """A trailing separator is how a caller says "directory" about one that
    does not exist yet, so the raw argument is what gets tested -- Path()
    normalises that separator away."""
    resolved = cli.resolve_output_path(spec, Path("resumes/priya-nair.pdf"))
    assert resolved.name == "priya-nair.json"
    assert resolved.parent == Path(spec)


def test_an_existing_directory_is_recognised_without_a_separator(
    tmp_path: Path,
) -> None:
    resolved = cli.resolve_output_path(str(tmp_path), Path("1.pdf"))
    assert resolved == tmp_path / "1.json"


@pytest.mark.parametrize("spec", ["out.json", "report", "a/b/result.json"])
def test_anything_else_is_taken_as_the_file_name(spec: str) -> None:
    """Guessing that a suffix-less path is a directory would make '-o report'
    create one nobody asked for."""
    assert cli.resolve_output_path(spec, Path("1.pdf")) == Path(spec)


def test_the_default_name_does_not_collide_between_resumes() -> None:
    """A fixed 'output.json' would have each extraction silently overwrite the
    last when several land in one directory."""
    names = {
        cli.resolve_output_path(".", Path(f"{stem}.pdf")).name
        for stem in ("1", "2", "priya-nair")
    }
    assert len(names) == 3


def test_a_suffixless_output_does_not_collide_with_its_artifact_directory() -> None:
    """'-o report' would otherwise put the result and the artifacts at the
    same path, and the write would fail on a directory."""
    assert cli._artifact_directory(Path("report")) == Path("report-artifacts")
    assert cli._artifact_directory(Path("report.json")) == Path("report")


@needs_models
def test_extracting_into_the_current_directory(tmp_path: Path) -> None:
    import shutil

    shutil.copy(NATIVE_FIXTURE, tmp_path / "priya.pdf")
    assert cli.main([str(tmp_path / "priya.pdf"), "-o", str(tmp_path)]) == cli.EXIT_OK
    assert (tmp_path / "priya.json").exists()


# -- version -----------------------------------------------------------------


def test_version_prints_the_installed_version_and_exits(capsys):
    """``--version`` answers from the distribution metadata, not a literal.

    A second copy of the number in the source is one that goes stale the first
    time ``pyproject.toml`` is bumped without it.
    """
    import restruct

    with pytest.raises(SystemExit) as exit_status:
        cli._parse_arguments(["--version"])

    assert exit_status.value.code == 0
    assert capsys.readouterr().out.strip() == f"restruct {restruct.__version__}"
