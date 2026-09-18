"""Typed request, question, and result schemas."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

Device = Literal["cpu", "mps", "cuda"]


class Question[T: Enum | bool](BaseModel):
    """One atomic judgment whose answer is a value of type T."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    instructions: str
    criteria: dict[T, str]


class EnumQuestionInput(BaseModel):
    """JSON representation of a question answered by an enum member."""

    model_config = ConfigDict(frozen=True)

    type: Literal["enum"] = "enum"
    instructions: str
    criteria: dict[str, str]


class BoolQuestionInput(BaseModel):
    """JSON representation of a question answered by a boolean."""

    model_config = ConfigDict(frozen=True)

    type: Literal["bool"] = "bool"
    instructions: str
    criteria: dict[bool, str]

    @field_validator("criteria")
    @classmethod
    def _validate_criteria(cls, value: dict[bool, str]) -> dict[bool, str]:
        if set(value) != {False, True}:
            raise ValueError("bool criteria must define both true and false")
        return value


type QuestionSpec = Annotated[
    EnumQuestionInput | BoolQuestionInput, Field(discriminator="type")
]


class EvaluationRequest(BaseModel):
    """Shared state and independent questions to evaluate against it."""

    model_config = ConfigDict(frozen=True)

    state: JsonValue
    questions: dict[str, QuestionSpec]


class ChoiceAnswer[T: Enum | bool](BaseModel):
    """A selected value of type T and the distribution over its alternatives."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    choice: T
    probabilities: dict[T, float]


class EvaluationResult(BaseModel):
    """Answers keyed by the corresponding question names."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model: str
    answers: dict[str, ChoiceAnswer[Enum | bool]]
