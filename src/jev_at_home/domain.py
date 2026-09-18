"""Data definitions for enum evaluation.

The module starts with the information the program represents, following the
How to Design Programs recipe.  These immutable values form the boundary
between JSON input, model inference, and JSON output.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class Question[T: str | bool](BaseModel):
    """One atomic judgment whose answer is a value of type T."""

    model_config = ConfigDict(frozen=True)

    instructions: str = Field(min_length=1)
    criteria: dict[T, str]

    @field_validator("instructions")
    @classmethod
    def instructions_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instructions must not be blank")
        return value

    @field_validator("criteria")
    @classmethod
    def descriptions_must_not_be_blank(cls, value: dict[T, str]) -> dict[T, str]:
        if any(not description.strip() for description in value.values()):
            raise ValueError("choice descriptions must not be blank")
        return value


class EnumQuestion(Question[str]):
    """A question answered by one member of a caller-defined string enum."""

    type: Literal["enum"] = "enum"

    @field_validator("criteria")
    @classmethod
    def criteria_must_define_an_enum(cls, value: dict[str, str]) -> dict[str, str]:
        if not 2 <= len(value) <= 26:
            raise ValueError("criteria must contain between 2 and 26 choices")
        if any(not name.strip() for name in value):
            raise ValueError("choice names must not be blank")
        return value


class BoolQuestion(Question[bool]):
    """A question answered by an actual boolean value."""

    type: Literal["bool"] = "bool"

    @field_validator("criteria")
    @classmethod
    def criteria_must_define_both_boole(cls, value: dict[bool, str]) -> dict[bool, str]:
        if set(value) != {False, True}:
            raise ValueError("bool criteria must define both true and false")
        return value


type QuestionSpec = Annotated[EnumQuestion | BoolQuestion, Field(discriminator="type")]


class EvaluationRequest(BaseModel):
    """Shared state and independent questions to evaluate against it."""

    model_config = ConfigDict(frozen=True)

    state: JsonValue
    questions: dict[str, QuestionSpec] = Field(min_length=1)

    @field_validator("questions")
    @classmethod
    def question_names_must_not_be_blank(
        cls, value: dict[str, QuestionSpec]
    ) -> dict[str, QuestionSpec]:
        if any(not name.strip() for name in value):
            raise ValueError("question names must not be blank")
        return value


class ChoiceAnswer[T: str | bool](BaseModel):
    """A selected value of type T and the distribution over its alternatives."""

    model_config = ConfigDict(frozen=True)

    choice: T
    probabilities: dict[T, float]


class EvaluationResult(BaseModel):
    """Answers keyed by the corresponding question names."""

    model_config = ConfigDict(frozen=True)

    model: str
    answers: dict[str, ChoiceAnswer[str | bool]]
