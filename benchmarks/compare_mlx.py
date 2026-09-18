"""Ablate MLX scheduling/projection changes on identical weights and inputs."""

from __future__ import annotations

import json
import platform
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Annotated
from unittest.mock import patch

import mlx.core as mx
import typer
from mlx_lm.models import qwen3

from jev_at_home import mlx_evaluator
from jev_at_home.cli import DEFAULT_MODEL
from jev_at_home.evaluator import _compile_questions, _measure_shared_prefix
from jev_at_home.mlx_evaluator import MLXEvaluator
from jev_at_home.schemas import EvaluationRequest, TypeSafeWorkflow
from jev_at_home.typesafe_evals import (
    _download_typesafe_asset,
    _load_evaluation_suite,
    _read_answer_value,
)


def compare_mlx(
    output: Annotated[Path, typer.Option()],
    repeats: Annotated[int, typer.Option(min=1)] = 2,
    batch_sizes: str = "1,2,4,8",
    workflow: TypeSafeWorkflow | None = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    profile: bool = False,
    variants: str = "original,projection,batching,both",
    asset_dir: Path | None = None,
) -> None:
    """Warm each variant, reverse run order, and retain per-answer drift."""

    sizes = [int(value) for value in batch_sizes.split(",")]
    if not sizes or min(sizes) < 1:
        raise typer.BadParameter("batch sizes must be positive")
    workflows = [workflow] if workflow else list(TypeSafeWorkflow)
    assets = {
        item: (asset_dir / f"{item.value}-cases.js").read_text()
        if asset_dir
        else _download_typesafe_asset(item)
        for item in workflows
    }
    suites = [_load_evaluation_suite(item, assets[item], limit) for item in workflows]
    loaded = MLXEvaluator.load(DEFAULT_MODEL)
    implementations = {
        "original": _OriginalEvaluator,
        "projection": _ProjectionEvaluator,
        "batching": _BatchingEvaluator,
        "both": MLXEvaluator,
    }
    evaluators = {
        name: cls(
            model_name=DEFAULT_MODEL, tokenizer=loaded.tokenizer, model=loaded.model
        )
        for name, cls in implementations.items()
        if name in set(variants.split(",")) | {"original"}
    }
    report = {
        "model": DEFAULT_MODEL,
        "platform": platform.platform(),
        "versions": {name: version(name) for name in ("mlx", "mlx-lm", "transformers")},
        "repeats": repeats,
        "batch_sizes": sizes,
        "source_sha256": {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(mlx_evaluator.__file__))
        },
        "asset_sha256": {
            item.value: sha256(asset.encode()).hexdigest()
            for item, asset in assets.items()
        },
        "measurements": [],
        "prompt_analysis": [],
        "profiles": [],
    }
    for suite in suites:
        request = suite.batches[0].request
        configurations = [
            (name, size) for size in sizes for name in variants.split(",")
        ]
        if any(name not in evaluators for name, _ in configurations):
            raise typer.BadParameter("unknown benchmark variant")
        for name, size in configurations:
            typer.echo(f"Warm up {suite.workflow.value}: {name}, batch {size}")
            evaluators[name].evaluate(request, batch_size=size)
        reference = []
        reference_times = []
        for batch in suite.batches:
            started = perf_counter()
            reference.append(
                evaluators["original"].evaluate(batch.request, batch_size=1)
            )
            reference_times.append(perf_counter() - started)
        for batch in suite.batches:
            report["prompt_analysis"].append(
                {
                    "workflow": suite.workflow.value,
                    **_analyze_prompts(loaded, batch.request),
                }
            )
        for repetition in range(repeats):
            order = configurations if repetition % 2 == 0 else configurations[::-1]
            for name, size in order:
                elapsed = 0.0
                correct = changed = question_count = 0
                largest_drift = 0.0
                for batch, expected, reference_time in zip(
                    suite.batches, reference, reference_times, strict=True
                ):
                    if name == "original" and size == 1 and repetition == 0:
                        result = expected
                        elapsed += reference_time
                    else:
                        started = perf_counter()
                        result = evaluators[name].evaluate(
                            batch.request, batch_size=size
                        )
                        elapsed += perf_counter() - started
                    for key, answer in result.answers.items():
                        previous = expected.answers[key]
                        correct += (
                            _read_answer_value(answer) == batch.expected_answers[key]
                        )
                        changed += _read_answer_value(answer) != _read_answer_value(
                            previous
                        )
                        previous_probabilities = previous.model_dump(mode="json")[
                            "probabilities"
                        ]
                        probabilities = answer.model_dump(mode="json")["probabilities"]
                        largest_drift = max(
                            largest_drift,
                            *(
                                abs(probability - previous_probabilities[choice])
                                for choice, probability in probabilities.items()
                            ),
                        )
                        question_count += 1
                measurement = {
                    "workflow": suite.workflow.value,
                    "variant": name,
                    "batch_size": size,
                    "repetition": repetition + 1,
                    "seconds": elapsed,
                    "cases": suite.evaluated_case_count,
                    "questions": question_count,
                    "reference_matches": correct,
                    "changed_answers": changed,
                    "maximum_probability_drift": largest_drift,
                }
                report["measurements"].append(measurement)
                typer.echo(json.dumps(measurement), color=False)
                output.write_text(json.dumps(report, indent=2) + "\n")
        if profile:
            report["profiles"].append(
                {"workflow": suite.workflow.value, **_profile_request(loaded, request)}
            )
            output.write_text(json.dumps(report, indent=2) + "\n")


class _OriginalEvaluator(MLXEvaluator):
    """Control: original-order batches and full-vocabulary projection."""

    def _plan_batches(self, suffixes, batch_size):
        return [
            list(range(start, min(start + batch_size, len(suffixes))))
            for start in range(0, len(suffixes), batch_size)
        ]

    def _project_choices(self, hidden, label_token_ids):
        logits = (
            self.model.model.embed_tokens.as_linear(hidden)
            if self.model.args.tie_word_embeddings
            else self.model.lm_head(hidden)
        )
        return logits[:, mx.array(label_token_ids)]


class _ProjectionEvaluator(_OriginalEvaluator):
    _project_choices = MLXEvaluator._project_choices


class _BatchingEvaluator(MLXEvaluator):
    _project_choices = _OriginalEvaluator._project_choices


def _analyze_prompts(evaluator: MLXEvaluator, request: EvaluationRequest) -> dict:
    """Count padding and exact subgroup prefixes without running the model."""

    sequences = [q.token_ids for q in _compile_questions(evaluator.tokenizer, request)]
    prefix = _measure_shared_prefix(sequences)
    suffixes = [sequence[prefix:] for sequence in sequences]
    # Sorted adjacent LCPs count the additional trie edges that can be reused.
    ordered = sorted(sequences)
    additional = sum(
        max(0, _measure_shared_prefix([left, right]) - prefix)
        for left, right in zip(ordered, ordered[1:], strict=False)
    )
    return {
        "questions": len(sequences),
        "prefix_tokens": prefix,
        "suffix_tokens": sum(map(len, suffixes)),
        "additional_shared_tokens": additional,
        "padded_tokens": {
            str(size): {
                name: sum(
                    len(batch) * max(len(suffixes[index]) for index in batch)
                    for batch in cls._plan_batches(evaluator, suffixes, size)
                )
                for name, cls in (
                    ("original", _OriginalEvaluator),
                    ("bucketed", MLXEvaluator),
                )
            }
            for size in (1, 2, 4, 8)
        },
    }


def _profile_request(evaluator: MLXEvaluator, request: EvaluationRequest) -> dict:
    """Synchronize stage boundaries; diagnostic timings are NOT benchmark results."""

    durations = defaultdict(float)
    phase = "other"

    def measure(name, function):
        @wraps(function)
        def timed(*args, **kwargs):
            nonlocal phase
            previous_phase = phase
            if name in ("prefix", "suffix"):
                phase = name
            # Materialize inputs before timing lazy MLX operations.
            mx.eval([arg for arg in args if isinstance(arg, mx.array)])
            mx.synchronize()
            started = perf_counter()
            result = function(*args, **kwargs)
            if isinstance(result, mx.array):
                mx.eval(result)
            mx.synchronize()
            durations[f"{phase}.{name}"] += perf_counter() - started
            phase = previous_phase
            return result

        return timed

    with ExitStack() as stack:
        for owner, attribute, name in (
            (mlx_evaluator, "_compile_questions", "compile"),
            (MLXEvaluator, "_prefill_prefix", "prefix"),
            (MLXEvaluator, "_predict_suffixes", "suffix"),
            (MLXEvaluator, "_project_choices", "head"),
            (qwen3, "scaled_dot_product_attention", "attention"),
            (qwen3.MLP, "__call__", "mlp"),
        ):
            stack.enter_context(
                patch.object(owner, attribute, measure(name, getattr(owner, attribute)))
            )
        evaluator.evaluate(request, batch_size=1)
    typer.echo(json.dumps({"diagnostic_seconds": dict(durations)}))
    return {
        "note": "Synchronized diagnostics on first request only; nested times overlap.",
        "seconds": dict(durations),
    }


if __name__ == "__main__":
    typer.run(compare_mlx)
