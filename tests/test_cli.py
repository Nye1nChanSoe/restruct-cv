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


# -- what --debug and --stages mean ------------------------------------------


def selected(argv: list[str]) -> frozenset[int]:
    return cli._selected_stages(cli._parse_arguments(argv))


def test_no_flag_writes_no_artifacts() -> None:
    assert selected([str(NATIVE_FIXTURE), "-o", "out.json"]) == frozenset()


def test_debug_alone_means_stages_four_and_five() -> None:
    assert selected([str(NATIVE_FIXTURE), "-o", "out.json", "--debug"]) == {4, 5}


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
        [str(NATIVE_FIXTURE), "-o", "out.json", "--debug", "--stages", "2"]
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
    document = tmp_path / "resume.docx"
    document.write_bytes(b"not a pdf")
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
        (["--debug"], {"pass-4-sections"}),
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
    for index, flags in enumerate(([], ["--debug"], ["--stages", "1-5"])):
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
