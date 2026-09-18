"""Command-line interface for local batched enum evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from jev_at_home.domain import EvaluationRequest
from jev_at_home.evaluator import TransformersChoiceEvaluator


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

app = typer.Typer(
    no_args_is_help=True,
    help="Evaluate enum questions with one batched transformer forward pass.",
)


def read_request(path: Path | None) -> EvaluationRequest:
    """Parse an evaluation request from a file, or from standard input."""

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
        str | None, typer.Option(help="Inference device: cpu, mps, or cuda.")
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

    request = read_request(request_file)
    evaluator = TransformersChoiceEvaluator.from_pretrained(model, device)
    result = evaluator.evaluate(request, temperature=temperature)
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def example() -> None:
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


def main() -> None:
    app()
