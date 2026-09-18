from __future__ import annotations

from enum import Enum
from io import StringIO

from rich.console import Console

from jev_at_home.cli import _print_result
from jev_at_home.schemas import (
    ChoiceAnswer,
    EvaluationRequest,
    EvaluationResult,
    ScoreAnswer,
)


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
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe is this?",
                    "criteria": ["Minor", "Moderate", "Severe"],
                },
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
            ),
            "severity": ScoreAnswer(
                score=1.25,
                legend={0: "Minor", 1: "Moderate", 2: "Severe"},
                probabilities={0: 0.10, 1: 0.55, 2: 0.35},
            ),
        },
    )
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=100)

    _print_result(console, request, result, 0.1234)

    rendered = output.getvalue()
    assert "test-model" in rendered
    assert "Inference: 123.4 ms · 2 questions · 1 batch" in rendered
    assert "State" in rendered
    assert "My payment failed" in rendered
    assert "Which team should handle this?" in rendered
    assert "billing" in rendered
    assert "Payments and refunds" in rendered
    assert "75.0%" in rendered
    assert "How severe is this?" in rendered
    assert "Score: 1.25 / 2" in rendered
    assert "Moderate" in rendered
    assert "55.0%" in rendered
