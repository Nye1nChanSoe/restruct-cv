"""The commands a contributor needs and a person extracting a resume does not.

Deliberately **not** a `[project.scripts]` entry. Everything here reads or
writes a directory that exists only in a checkout -- `resumes-synthetic/`,
`resumes-truths/`, `results/`, `examples/`, `tests/` -- so a console script
installed from PyPI would be a command that is broken for everyone who has it.
Keeping it a file in `tools/` is what makes "only in a checkout" a fact rather
than a convention.

    uv run tools/dev.py batch [--truths | --unsupported] [--stages 1-5]
    uv run tools/dev.py examples
    uv run tools/dev.py scorecard [--update-baseline]
    uv run tools/dev.py export-onnx [...]

The extraction itself is the same code the shipped CLI runs: this module
decides which corpus and where the artifacts go, and nothing else. A dev tool
that reached into the pipeline would be measuring something users never run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Run as a path (`uv run tools/dev.py`), sys.path[0] is tools/, so neither the
# package nor its sibling `tests` and `tools` modules are importable yet.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import restruct  # noqa: E402
from restruct import cli  # noqa: E402
from restruct.configs import SETTINGS  # noqa: E402
from restruct.stages import ALL_STAGES, raw_extraction_reader  # noqa: E402


def _corpus(arguments: argparse.Namespace) -> tuple[Path, Path]:
    """Which corpus a batch reads, and where its results go."""
    paths = SETTINGS.paths
    if arguments.truths:
        return (
            PROJECT_ROOT / paths.truths_input_directory,
            PROJECT_ROOT / paths.truths_results_directory,
        )
    if arguments.unsupported:
        # Those parses are untrustworthy by definition; the run exists so the
        # overlays that show why can be looked at.
        return (
            PROJECT_ROOT / paths.unsupported_input_directory,
            PROJECT_ROOT / paths.unsupported_results_directory,
        )
    return (
        PROJECT_ROOT / paths.input_directory,
        PROJECT_ROOT / paths.results_directory,
    )


def batch(arguments: argparse.Namespace) -> int:
    """Extract a whole corpus, writing every stage.

    Writing everything is the purpose rather than the default: this run is how
    the committed corpus is regenerated, and anything less would let a stale
    artifact survive it and make `git status examples/` read as clean.
    """
    from restruct.pipeline import extract_resume

    input_directory, output_root = _corpus(arguments)
    if not input_directory.is_dir():
        print(f"dev: no such corpus: {input_directory}", file=sys.stderr)
        return 1

    stages = arguments.stages if arguments.stages is not None else ALL_STAGES
    models = cli._models(PROJECT_ROOT)
    raw_debug_directory = PROJECT_ROOT / SETTINGS.debug.raw_extraction_directory
    ocr_debug_directory = PROJECT_ROOT / SETTINGS.debug.ocr_extraction_directory
    # The default corpus writes its pass-1 dumps to the shared debug
    # directories; a local corpus keeps them beside its own result.
    local = output_root != PROJECT_ROOT / SETTINGS.paths.results_directory

    sources = sorted(
        path
        for suffix in cli.SUPPORTED_SUFFIXES
        for path in input_directory.glob(f"*{suffix}")
    )
    for source in sources:
        resume_output = output_root / source.stem
        reader = raw_extraction_reader(source)
        extract_resume(
            source,
            resume_output,
            (
                resume_output / f"raw-{reader}.json"
                if local
                else raw_debug_directory / f"{source.stem}.raw-{reader}.json"
            ),
            (
                resume_output / "debug" / "ocr" / "raw-tesseract.json"
                if local
                else ocr_debug_directory / f"{source.stem}.ocr-tesseract.json"
            ),
            models[0],
            models[1],
            stages=stages,
        )
        if arguments.reconstruct:
            cli._reconstruct(
                resume_output / "resume.json", resume_output / "reconstruction"
            )
        print(f"extracted: {source.name}")
    return 0


def examples(arguments: argparse.Namespace) -> int:
    """Mirror the three worked examples out of results/ into examples/."""
    from tools.refresh_examples import refresh

    return refresh(PROJECT_ROOT)


def scorecard(arguments: argparse.Namespace) -> int:
    """Per-field precision, recall and F1 against the hand-written labels."""
    from tests.scorecard import main as run_scorecard

    return run_scorecard(["--update-baseline"] if arguments.update_baseline else [])


def export_onnx(arguments: argparse.Namespace) -> int:
    """Re-export models/*/model.onnx from the safetensors.

    Imported inside the function because it needs the `export` group -- torch,
    transformers, optimum -- and nothing else here does. `uv run --group export
    tools/dev.py export-onnx`.
    """
    from tools.export_onnx import main as run_export

    try:
        return run_export(arguments.rest)
    except ImportError as error:
        # torch, transformers and optimum are imported where they are used, so
        # this is what a run without the group looks like: a traceback naming
        # one of them, several frames from anything a reader could act on.
        print(
            f"dev: the export group is not installed ({error}). "
            "Run `uv run --group export tools/dev.py export-onnx`.",
            file=sys.stderr,
        )
        return 1


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dev",
        description="Contributor commands. Checkout only; see tools/dev.py.",
    )
    # Before the subparser, so `dev.py --version` answers instead of failing
    # on the required command.
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"restruct {restruct.__version__} (dev)",
        help="Print the version and exit.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("batch", help="extract a whole corpus into results/")
    corpus = run.add_mutually_exclusive_group()
    corpus.add_argument(
        "--truths",
        action="store_true",
        help="resumes-truths/ (local, gitignored) into results/0-truths/",
    )
    corpus.add_argument(
        "--unsupported",
        action="store_true",
        help="resumes-unsupported/ into results/1-unsupported/",
    )
    run.add_argument(
        "--stages",
        type=cli.parse_stages,
        metavar="SPEC",
        help=(
            "Which stages' artifacts to write: 1-5, 3, 2,4,5, 1-3,5. Selects "
            "artifacts only; every pass always runs, because each feeds the "
            "next. Defaults to all five."
        ),
    )
    run.add_argument(
        "--reconstruct",
        action="store_true",
        help="also draw each result back out as a readable page",
    )
    run.set_defaults(handler=batch)

    mirror = commands.add_parser(
        "examples", help="refresh examples/ from results/ (run batch first)"
    )
    mirror.set_defaults(handler=examples)

    score = commands.add_parser("scorecard", help="per-field precision/recall/F1")
    score.add_argument(
        "--update-baseline",
        action="store_true",
        help="re-freeze tests/baseline_scores.json at the current scores",
    )
    score.set_defaults(handler=scorecard)

    export = commands.add_parser(
        "export-onnx", help="re-export the weights (needs --group export)"
    )
    export.add_argument("rest", nargs=argparse.REMAINDER)
    export.set_defaults(handler=export_onnx)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        return arguments.handler(arguments)
    except cli.RestructError as error:
        # The same exit codes the shipped CLI reports, so a failure here reads
        # the same way it would for a user.
        print(f"dev: {error}", file=sys.stderr)
        for error_type, code in cli._EXIT_CODES:
            if isinstance(error, error_type):
                return code
        return cli.EXIT_UNEXPECTED


if __name__ == "__main__":
    raise SystemExit(main())
