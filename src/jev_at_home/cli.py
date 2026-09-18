"""Command-line interface for local batched typed evaluation."""

from __future__ import annotations

import sys
import time
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue, ValidationError
from rich import box
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from jev_at_home.evaluator import TransformersEvaluator
from jev_at_home.schemas import (
    Answer,
    ChoiceAnswer,
    Device,
    EvaluationRequest,
    EvaluationResult,
    QuestionSpec,
    ScoreAnswer,
    ScoreQuestionInput,
)

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
_CONSOLE = Console()

app = typer.Typer(
    no_args_is_help=True,
    help="Evaluate typed questions with one batched transformer forward pass.",
)


def run() -> None:
    app()


@app.command()
def judge(
    request_file: Annotated[
        Path | None,
        typer.Argument(
            help="JSON request file. Omit to read JSON from standard input.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    model: Annotated[
        str, typer.Option(help="Hugging Face causal language model to load.")
    ] = DEFAULT_MODEL,
    device: Annotated[
        Device | None, typer.Option(help="Inference device: cpu, mps, or cuda.")
    ] = None,
    temperature: Annotated[
        float,
        typer.Option(
            min=0.000001,
            help="Softmax temperature; this does not calibrate the probabilities.",
        ),
    ] = 1.0,
) -> None:
    """Judge all questions in REQUEST_FILE as a single model batch."""

    request = _load_request(request_file)
    evaluator = TransformersEvaluator.load(model, device)
    inference_started_at = time.perf_counter()
    result = evaluator.evaluate(request, temperature=temperature)
    inference_seconds = time.perf_counter() - inference_started_at
    _print_result(_CONSOLE, request, result, inference_seconds)


@app.command("example")
def show_example() -> None:
    """Print an example request that can be piped into `judge`."""

    typer.echo(
        EvaluationRequest.model_validate(
            {
                "state": "Help! My payouts have been failing for three days.",
                "questions": {
                    "department": {
                        "type": "enum",
                        "instructions": "Which team should handle this?",
                        "criteria": {
                            "billing": "Payments, invoicing, and refunds",
                            "technical": "Bugs, outages, and integrations",
                            "sales": "Pricing, upgrades, and new accounts",
                        },
                    },
                    "urgency": {
                        "type": "bool",
                        "instructions": "Does this request need urgent attention?",
                        "criteria": {
                            "true": "Time-sensitive or actively blocking",
                            "false": "No urgency is expressed",
                        },
                    },
                },
            }
        ).model_dump_json(indent=2)
    )


def _load_request(path: Path | None) -> EvaluationRequest:
    """Return the evaluation request read from a file or standard input."""

    if path is None:
        if sys.stdin.isatty():
            raise typer.BadParameter("provide REQUEST.json or pipe JSON on stdin")
        raw = sys.stdin.read()
    else:
        raw = path.read_text()
    try:
        return EvaluationRequest.model_validate_json(raw)
    except ValidationError as error:
        raise typer.BadParameter(str(error)) from error


def _print_result(
    console: Console,
    request: EvaluationRequest,
    result: EvaluationResult,
    inference_seconds: float,
) -> None:
    """Print the model and each question's answer distribution."""

    question_count = len(request.questions)
    question_label = "question" if question_count == 1 else "questions"
    console.print(Text.assemble(("Model: ", "bold"), result.model))
    console.print(
        Text.assemble(
            ("Inference: ", "bold"),
            _format_inference_duration(inference_seconds),
            (f" · {question_count} {question_label} · 1 batch", "dim"),
        )
    )
    console.print(_build_state_panel(request.state))

    for name, question in request.questions.items():
        answer = result.answers[name]
        summary = Text(question.instructions, style="bold")
        summary.append("\n")
        summary.append_text(_format_result_answer(answer))
        console.print(Panel(summary, title=Text(name, style="bold cyan"), expand=False))
        console.print(_build_probability_table(question, answer))


def _format_inference_duration(seconds: float) -> str:
    """Format an inference duration at a useful human scale."""

    if seconds < 1:
        return f"{seconds * 1_000:.1f} ms"
    return f"{seconds:.2f} s"


def _build_state_panel(state: JsonValue) -> Panel:
    """Build a panel containing the original shared state."""

    content = Text(state) if isinstance(state, str) else JSON.from_data(state)
    return Panel(content, title="State", title_align="left")


def _format_result_answer(answer: Answer) -> Text:
    """Format the primary value of a typed answer."""

    if isinstance(answer, ScoreAnswer):
        highest_level = max(answer.legend, default=0)
        return Text.assemble(
            ("Score: ", "dim"),
            (f"{answer.score:.2f} / {highest_level}", "bold green"),
        )
    return Text.assemble(
        ("Selected answer: ", "dim"),
        (_format_answer(answer.choice), "bold green"),
    )


def _build_probability_table(
    question: QuestionSpec, answer: Answer
) -> Table:
    """Build a probability table for one typed answer."""

    if isinstance(answer, ScoreAnswer):
        return _build_score_probability_table(answer)
    return _build_choice_probability_table(question, answer)


def _build_score_probability_table(answer: ScoreAnswer) -> Table:
    """Build a table of score levels, criteria, and probabilities."""

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold magenta")
    table.add_column("")
    table.add_column("Level")
    table.add_column("Criterion")
    table.add_column("Probability", justify="right")

    most_likely_level = max(answer.probabilities, key=answer.probabilities.__getitem__)
    for level, probability in answer.probabilities.items():
        table.add_row(
            "✓" if level == most_likely_level else "",
            str(level),
            answer.legend[level],
            f"{probability:.1%}",
        )

    return table


def _build_choice_probability_table(
    question: QuestionSpec, answer: ChoiceAnswer[Enum | bool]
) -> Table:
    """Build a table of choices, criteria, and probabilities."""

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold magenta")
    table.add_column("")
    table.add_column("Choice")
    table.add_column("Criterion")
    table.add_column("Probability", justify="right")

    for choice, probability in answer.probabilities.items():
        table.add_row(
            "✓" if choice == answer.choice else "",
            _format_answer(choice),
            _describe_choice(question, choice),
            f"{probability:.1%}",
        )

    return table


def _describe_choice(question: QuestionSpec, choice: Enum | bool) -> str:
    """Return the criterion description for a typed choice."""

    if isinstance(question, ScoreQuestionInput):
        raise TypeError("score questions do not have enum or bool choices")
    key = choice if isinstance(choice, bool) else str(choice.value)
    return question.criteria[key]


def _format_answer(choice: Enum | bool) -> str:
    """Format a typed answer for terminal output."""

    if isinstance(choice, bool):
        return str(choice).lower()
    return str(choice.value)
