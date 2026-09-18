"""Command-line interface for local batched enum evaluation."""

from __future__ import annotations

import sys
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

from jev_at_home.evaluator import TransformersChoiceEvaluator
from jev_at_home.schemas import (
    ChoiceAnswer,
    Device,
    EvaluationRequest,
    EvaluationResult,
    QuestionSpec,
)

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
_CONSOLE = Console()

app = typer.Typer(
    no_args_is_help=True,
    help="Evaluate enum questions with one batched transformer forward pass.",
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
    evaluator = TransformersChoiceEvaluator.load(model, device)
    result = evaluator.evaluate(request, temperature=temperature)
    _print_result(_CONSOLE, request, result)


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
    console: Console, request: EvaluationRequest, result: EvaluationResult
) -> None:
    """Print the model and each question's answer distribution."""

    console.print(Text.assemble(("Model: ", "bold"), result.model))
    console.print(_build_state_panel(request.state))

    for name, question in request.questions.items():
        answer = result.answers[name]
        summary = Text(question.instructions, style="bold")
        summary.append("\nSelected answer: ", style="dim")
        summary.append(_format_answer(answer.choice), style="bold green")
        console.print(Panel(summary, title=Text(name, style="bold cyan"), expand=False))
        console.print(_build_probability_table(question, answer))


def _build_state_panel(state: JsonValue) -> Panel:
    """Build a panel containing the original shared state."""

    content = Text(state) if isinstance(state, str) else JSON.from_data(state)
    return Panel(content, title="State", title_align="left")


def _build_probability_table(
    question: QuestionSpec, answer: ChoiceAnswer[Enum | bool]
) -> Table:
    """Build a table of choices, criteria, and probabilities for one answer."""

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

    key = choice if isinstance(choice, bool) else str(choice.value)
    return question.criteria[key]


def _format_answer(choice: Enum | bool) -> str:
    """Format a typed answer for terminal output."""

    if isinstance(choice, bool):
        return str(choice).lower()
    return str(choice.value)
