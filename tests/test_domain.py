from __future__ import annotations

import pytest
from pydantic import ValidationError

from jev_at_home.domain import BoolQuestion, EnumQuestion, EvaluationRequest


def test_question_requires_at_least_two_choices() -> None:
    with pytest.raises(ValidationError, match="between 2 and 26"):
        EnumQuestion(instructions="Pick one", criteria={"only": "The only choice"})


def test_bool_question_requires_true_and_false() -> None:
    with pytest.raises(ValidationError, match="both true and false"):
        BoolQuestion(instructions="Is it?", criteria={True: "Yes"})


def test_request_preserves_question_and_choice_order() -> None:
    request = EvaluationRequest.model_validate(
        {
            "state": {"message": "hello"},
            "questions": {
                "tone": {
                    "type": "enum",
                    "instructions": "What is the tone?",
                    "criteria": {"warm": "Friendly", "cold": "Unfriendly"},
                }
            },
        }
    )

    assert list(request.questions) == ["tone"]
    assert list(request.questions["tone"].criteria) == ["warm", "cold"]


def test_request_parses_json_boolean_criteria_as_booleans() -> None:
    request = EvaluationRequest.model_validate_json(
        """
        {
          "state": "The service is unavailable",
          "questions": {
            "is_outage": {
              "type": "bool",
              "instructions": "Is there an outage?",
              "criteria": {"true": "Unavailable", "false": "Available"}
            }
          }
        }
        """
    )

    question = request.questions["is_outage"]
    assert isinstance(question, BoolQuestion)
    assert set(question.criteria) == {True, False}
