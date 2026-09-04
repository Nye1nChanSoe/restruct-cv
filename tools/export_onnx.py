"""Export the two local models to ONNX, in place beside their weights.

Run time never touches torch: the package loads ``model.onnx`` through ONNX
Runtime and tokenizes with ``tokenizers``. This script is the other half of
that trade -- it is how the ``model.onnx`` files are produced, and it is the
only place in the repository that needs torch, transformers and optimum. They
live in the ``export`` dependency group and are not installed by a user::

    uv run --group export python tools/export_onnx.py
    uv run --group export python tools/export_onnx.py --precision int8

The output is written *into* the source model directory rather than beside it,
so a directory keeps its name and one directory answers both questions: which
weights it was exported from, and what run time reads. The safetensors are
left alone -- they are the input to the next export.

**fp32 is the default because int8 was measured and rejected.** Dynamic int8
quantization is four times smaller (23 MB and 66 MB, against 90 MB and 261 MB)
and it moves output. At fp32 the ONNX path reproduces the torch path exactly:
every golden snapshot is byte-identical and every scorecard field scores what
it scored before, which is what makes the swap an infrastructure change rather
than a behavioural one. Quantized, on the 2026-09-04 corpus:

- int8 both: macro F1 0.968 (from 0.966). `header.name` 0.94 -> 1.00 and
  `header.job_titles` 0.75 -> 0.77, but `experience.job_titles` and
  `experience.companies` both fall 0.98 -> 0.96 -- one company in
  `8.compound` is read as a job title. The change budget in CLAUDE.md does
  not allow a field to drop, so this does not ship.
- int8 MiniLM only, or int8 NER only: the same two experience fields drop,
  for 0.968 and 0.963 respectively. Neither half is the culprit alone.
- int8 MiniLM with `per_channel` (with and without `reduce_range`), which
  usually recovers quantization loss: the experience fields come back but
  `header.job_titles` collapses to 0.65, thirteen false positives.

The thresholds this package compares similarities against (0.55, with a 0.06
winner margin) sit close enough together that int8's noise crosses them. That
is the thing to revisit if the semantic tier is ever made less threshold-bound
-- and it is the thing to re-measure before bundling MiniLM in the wheel,
where 90 MB is the number that matters.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# task name per model directory; both are encoder-only forward passes.
EXPORTS = {
    "all-MiniLM-L6-v2": "feature-extraction",
    "distilbert-NER": "token-classification",
}


def _export(model_directory: Path, task: str, precision: str) -> Path:
    from optimum.exporters.onnx import main_export

    destination = model_directory / "model.onnx"
    with tempfile.TemporaryDirectory() as staging_name:
        staging = Path(staging_name)
        main_export(
            model_name_or_path=str(model_directory),
            output=staging,
            task=task,
            do_validation=False,
        )
        exported = staging / "model.onnx"
        if precision == "fp32":
            shutil.copy2(exported, destination)
            return destination

        from onnxruntime.quantization import QuantType, quantize_dynamic

        # Dynamic quantization: weights to int8 on disk, activations computed
        # in float at run time. No calibration set is needed, which matters
        # here because there is no corpus to calibrate against that is not
        # also the corpus the scorecard scores.
        quantize_dynamic(
            model_input=exported,
            model_output=destination,
            weight_type=QuantType.QInt8,
            extra_options={"MatMulConstBOnly": True},
        )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models-directory",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="directory holding the model directories (default: ./models)",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "int8"),
        default="fp32",
        help="fp32 is what ships; int8 is a quarter of the size and moves output",
    )
    parser.add_argument(
        "--only",
        choices=sorted(EXPORTS),
        help="export a single model directory",
    )
    arguments = parser.parse_args(argv)

    names = [arguments.only] if arguments.only else sorted(EXPORTS)
    for name in names:
        model_directory = arguments.models_directory / name
        if not model_directory.is_dir():
            print(f"missing model directory: {model_directory}", file=sys.stderr)
            return 1
        print(f"exporting {name} ({arguments.precision})...", flush=True)
        written = _export(model_directory, EXPORTS[name], arguments.precision)
        megabytes = written.stat().st_size / 1_000_000
        print(f"  {written} — {megabytes:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
