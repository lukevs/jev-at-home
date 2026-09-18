from __future__ import annotations

from enum import Enum
from io import StringIO

from rich.console import Console

from jev_at_home.cli import _print_result
from jev_at_home.schemas import ChoiceAnswer, EvaluationRequest, EvaluationResult


def test_print_result_includes_question_answer_and_probabilities() -> None:
    request = EvaluationRequest.model_validate(
        {
            "state": "My payment failed",
            "questions": {
                "department": {
                    "type": "enum",
                    "instructions": "Which team should handle this?",
                    "criteria": {
                        "billing": "Payments and refunds",
                        "technical": "Bugs and outages",
                    },
                }
            },
        }
    )
    department = Enum(
        "Department",
        {"BILLING": "billing", "TECHNICAL": "technical"},
    )
    result = EvaluationResult(
        model="test-model",
        answers={
            "department": ChoiceAnswer[department](
                choice=department.BILLING,
                probabilities={
                    department.BILLING: 0.75,
                    department.TECHNICAL: 0.25,
                },
            )
        },
    )
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=100)

    _print_result(console, request, result, 0.1234)

    rendered = output.getvalue()
    assert "test-model" in rendered
    assert "Inference: 123.4 ms · 1 question · 1 batch" in rendered
    assert "State" in rendered
    assert "My payment failed" in rendered
    assert "Which team should handle this?" in rendered
    assert "billing" in rendered
    assert "Payments and refunds" in rendered
    assert "75.0%" in rendered
