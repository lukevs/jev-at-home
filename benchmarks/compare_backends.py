"""Compare loaded runtimes on the same downloaded public evaluation cases."""

from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path
from statistics import median
from typing import Annotated

import typer

from jev_at_home.backends import load_evaluator
from jev_at_home.cli import DEFAULT_MODEL
from jev_at_home.schemas import Backend, TypeSafeWorkflow
from jev_at_home.typesafe_evals import (
    _download_typesafe_asset,
    _evaluate_suite,
    _load_evaluation_suite,
)


def compare_backends(
    output: Annotated[Path, typer.Option(help="Write raw measurements as JSON.")],
    repeats: Annotated[int, typer.Option(min=1)] = 2,
    batch_size: Annotated[int, typer.Option(min=1)] = 1,
    workflow: TypeSafeWorkflow | None = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    model: str = DEFAULT_MODEL,
) -> None:
    """Alternate backend order on each repetition, excluding loading/downloads."""

    workflows = [workflow] if workflow else list(TypeSafeWorkflow)
    suites = [
        _load_evaluation_suite(item, _download_typesafe_asset(item), limit)
        for item in workflows
    ]
    evaluators = {backend: load_evaluator(model, backend, "mps") for backend in Backend}
    for evaluator in evaluators.values():
        evaluator.evaluate(suites[0].batches[0].request, batch_size=batch_size)

    measurements = []
    for suite in suites:
        for repetition in range(repeats):
            order = list(Backend)
            if repetition % 2:
                order.reverse()
            for backend in order:
                # evaluate() materializes probabilities on the CPU before
                # returning, so the runner's timer includes GPU completion.
                result = _evaluate_suite(
                    evaluators[backend],
                    suite,
                    batch_size=batch_size,
                    temperature=1.0,
                )
                measurements.append(
                    {
                        "backend": backend.value,
                        "repetition": repetition + 1,
                        **result.model_dump(mode="json"),
                    }
                )
                typer.echo(
                    f"{suite.workflow.value} {backend.value} "
                    f"run={repetition + 1} "
                    f"seconds={result.inference_seconds:.3f} "
                    f"matches={result.correct_count}/{result.question_count}",
                )

    report = {
        "model": model,
        "platform": platform.platform(),
        "versions": {
            package: version(package)
            for package in ("torch", "transformers", "mlx", "mlx-lm")
        },
        "batch_size": batch_size,
        "repeats": repeats,
        "measurements": measurements,
        "median_suite_seconds": {
            backend.value: sum(
                median(
                    item["inference_seconds"]
                    for item in measurements
                    if item["backend"] == backend.value
                    and item["workflow"] == workflow.value
                )
                for workflow in workflows
            )
            for backend in Backend
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    typer.echo(json.dumps(report["median_suite_seconds"], indent=2))


if __name__ == "__main__":
    typer.run(compare_backends)
