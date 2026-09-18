from __future__ import annotations

import json
from enum import Enum

from jev_at_home.schemas import (
    BoolQuestionInput,
    ChoiceAnswer,
    EnumQuestionInput,
    EvaluationResult,
    ScoreAnswer,
    ScoreQuestionInput,
    TypeSafeWorkflow,
)
from jev_at_home.typesafe_evals import run_typesafe_evals


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes = []

    def evaluate(self, request, *, temperature, batch_size):
        self.calls += 1
        self.batch_sizes.append(batch_size)
        answers = {}
        for name, question in request.questions.items():
            match question:
                case BoolQuestionInput():
                    answers[name] = ChoiceAnswer[bool](
                        choice=True,
                        probabilities={True: 0.9, False: 0.1},
                    )
                case EnumQuestionInput():
                    choice_type = Enum("Choice", {"BLUE": "blue", "RED": "red"})
                    answers[name] = ChoiceAnswer[choice_type](
                        choice=choice_type.BLUE,
                        probabilities={choice_type.BLUE: 0.8, choice_type.RED: 0.2},
                    )
                case ScoreQuestionInput():
                    answers[name] = ScoreAnswer(
                        score=1.8,
                        legend={0: "Low", 1: "Medium", 2: "High"},
                        probabilities={0: 0.05, 1: 0.1, 2: 0.85},
                    )
        return EvaluationResult(model="fake", answers=answers)


def test_run_typesafe_evals_scores_published_reference_answers() -> None:
    asset = _build_viewer_asset()
    evaluator = FakeEvaluator()

    results = run_typesafe_evals(
        evaluator,
        [TypeSafeWorkflow.SECURITY_INCIDENTS],
        batch_size=2,
        download_asset=lambda workflow: asset,
    )

    result = results[0]
    assert evaluator.calls == 1
    assert evaluator.batch_sizes == [2]
    assert result.declared_case_count == 100
    assert result.published_case_count == 1
    assert result.evaluated_case_count == 1
    assert result.question_count == 3
    assert result.correct_count == 3
    assert result.agreement == 1.0


def _build_viewer_asset() -> str:
    payload = {
        "eval": {
            "id": "security_incidents",
            "n_cases": 100,
            "documents": [{"message": "This is clearly urgent."}],
            "questions": [
                {
                    "type": "noul",
                    "instructions": "Is this urgent?",
                    "criteria": {"true": "Urgent", "false": "Not urgent"},
                },
                {
                    "type": "choice",
                    "instructions": "Which color?",
                    "criteria": {"blue": "Blue", "red": "Red"},
                },
                {
                    "type": "score",
                    "instructions": "How strong?",
                    "criteria": ["Low", "Medium", "High"],
                },
            ],
            "cases": {
                "case-1": {
                    "models": {
                        "published-model": {
                            "nodes": [
                                {
                                    "node": "triage",
                                    "doc": 0,
                                    "questions": {
                                        "urgent": 0,
                                        "color": 1,
                                        "strength": 2,
                                    },
                                }
                            ]
                        }
                    },
                    "reference_answers": {
                        "triage": {
                            "urgent": {
                                "type": "noul",
                                "sets": [
                                    {
                                        "value": True,
                                        "probabilities": {
                                            "true": 0.8,
                                            "false": 0.2,
                                        },
                                    },
                                    {
                                        "value": True,
                                        "probabilities": {
                                            "true": 0.6,
                                            "false": 0.4,
                                        },
                                    },
                                ],
                            },
                            "color": {
                                "type": "choice",
                                "sets": [
                                    {
                                        "value": "blue",
                                        "probabilities": None,
                                    }
                                ],
                            },
                            "strength": {
                                "type": "score",
                                "sets": [
                                    {
                                        "value": "2",
                                        "probabilities": {
                                            "0": 0.1,
                                            "1": 0.2,
                                            "2": 0.7,
                                        },
                                    }
                                ],
                            },
                        }
                    },
                }
            },
        }
    }
    return f"__VIEWER_DATA__({json.dumps(payload)});"
