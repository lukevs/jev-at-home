"""Run the public TypeSafe workflow examples as question-level evals."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from jev_at_home.evaluator import TransformersEvaluator
from jev_at_home.schemas import (
    Answer,
    BoolQuestionInput,
    EnumQuestionInput,
    EvaluationRequest,
    ScoreAnswer,
    ScoreQuestionInput,
    TypeSafeEvalResult,
    TypeSafeWorkflow,
)

_ASSET_PREFIX = "__VIEWER_DATA__("
_ASSET_SUFFIX = ");"
_EVALS_BASE_URL = "https://evals.typesafe.ai"


class _ViewerQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    instructions: str
    criteria: dict[str, str] | list[str] | None = None


class _ViewerNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node: str
    doc: int | None = None
    questions: dict[str, int] = Field(default_factory=dict)


class _ViewerModelRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    nodes: list[_ViewerNode]


class _ViewerReferenceSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: JsonValue
    probabilities: dict[str, float] | None = None


class _ViewerReferenceAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    sets: list[_ViewerReferenceSet]


class _ViewerCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    models: dict[str, _ViewerModelRun]
    reference_answers: dict[str, dict[str, _ViewerReferenceAnswer]] = Field(
        default_factory=dict
    )


class _ViewerEval(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    n_cases: int
    cases: dict[str, _ViewerCase]
    documents: list[JsonValue]
    questions: list[_ViewerQuestion]


class _ViewerData(BaseModel):
    model_config = ConfigDict(frozen=True)

    eval: _ViewerEval


@dataclass(frozen=True)
class _EvaluationBatch:
    request: EvaluationRequest
    expected_answers: dict[str, bool | str]


@dataclass(frozen=True)
class _EvaluationSuite:
    workflow: TypeSafeWorkflow
    declared_case_count: int
    published_case_count: int
    evaluated_case_count: int
    batches: list[_EvaluationBatch]


def run_typesafe_evals(
    evaluator: TransformersEvaluator,
    workflows: list[TypeSafeWorkflow],
    *,
    case_limit: int | None = None,
    batch_size: int = 1,
    temperature: float = 1.0,
    download_asset: Callable[[TypeSafeWorkflow], str] | None = None,
) -> list[TypeSafeEvalResult]:
    """Run public workflow examples and score agreement with their references."""

    if batch_size < 1:
        raise ValueError("batch size must be at least one")

    download = download_asset or _download_typesafe_asset
    results = []
    for workflow in workflows:
        suite = _load_evaluation_suite(workflow, download(workflow), case_limit)
        results.append(
            _evaluate_suite(
                evaluator,
                suite,
                batch_size=batch_size,
                temperature=temperature,
            )
        )
    return results


def _load_evaluation_suite(
    workflow: TypeSafeWorkflow, asset: str, case_limit: int | None
) -> _EvaluationSuite:
    """Build executable batches from one TypeSafe viewer asset."""

    viewer_eval = _parse_viewer_asset(asset).eval
    if viewer_eval.id != workflow.value:
        raise ValueError(
            f"expected workflow {workflow.value!r}, received {viewer_eval.id!r}"
        )

    default_question_indexes = _index_default_questions(viewer_eval.cases)
    published_cases = list(viewer_eval.cases.items())
    selected_cases = published_cases[:case_limit] if case_limit else published_cases
    batches = [
        batch
        for _, case in selected_cases
        for batch in _build_case_batches(
            case,
            viewer_eval.documents,
            viewer_eval.questions,
            default_question_indexes,
        )
    ]
    return _EvaluationSuite(
        workflow=workflow,
        declared_case_count=viewer_eval.n_cases,
        published_case_count=len(published_cases),
        evaluated_case_count=len(selected_cases),
        batches=batches,
    )


def _evaluate_suite(
    evaluator: TransformersEvaluator,
    suite: _EvaluationSuite,
    *,
    batch_size: int,
    temperature: float,
) -> TypeSafeEvalResult:
    """Evaluate and score every question batch in a workflow suite."""

    correct_count = 0
    question_count = 0
    inference_seconds = 0.0

    for batch in suite.batches:
        inference_started_at = time.perf_counter()
        result = evaluator.evaluate(
            batch.request,
            temperature=temperature,
            batch_size=batch_size,
        )
        inference_seconds += time.perf_counter() - inference_started_at

        for question_id, expected_answer in batch.expected_answers.items():
            question_count += 1
            actual_answer = _read_answer_value(result.answers[question_id])
            correct_count += actual_answer == expected_answer

    return TypeSafeEvalResult(
        workflow=suite.workflow,
        declared_case_count=suite.declared_case_count,
        published_case_count=suite.published_case_count,
        evaluated_case_count=suite.evaluated_case_count,
        question_count=question_count,
        correct_count=correct_count,
        inference_seconds=inference_seconds,
    )

def _read_answer_value(answer: Answer) -> bool | str:
    """Return the discrete value used for reference-label agreement."""

    if isinstance(answer, ScoreAnswer):
        return str(max(answer.probabilities, key=answer.probabilities.__getitem__))
    if isinstance(answer.choice, Enum):
        return str(answer.choice.value)
    return answer.choice


def _parse_viewer_asset(asset: str) -> _ViewerData:
    """Parse the JavaScript wrapper around a TypeSafe viewer payload."""

    stripped_asset = asset.strip()
    if not stripped_asset.startswith(_ASSET_PREFIX) or not stripped_asset.endswith(
        _ASSET_SUFFIX
    ):
        raise ValueError("TypeSafe eval asset has an unexpected wrapper")
    payload = stripped_asset[len(_ASSET_PREFIX) : -len(_ASSET_SUFFIX)]
    return _ViewerData.model_validate_json(payload)


def _index_default_questions(cases: dict[str, _ViewerCase]) -> dict[str, int]:
    """Map question IDs for reference nodes no published model reached."""

    indexes = {}
    for case in cases.values():
        for model in case.models.values():
            for node in model.nodes:
                indexes.update(node.questions)
    return indexes


def _build_case_batches(
    case: _ViewerCase,
    documents: list[JsonValue],
    questions: list[_ViewerQuestion],
    default_question_indexes: dict[str, int],
) -> list[_EvaluationBatch]:
    """Build one shared-state batch for each referenced workflow node."""

    batches = []
    for node_name, reference_answers in case.reference_answers.items():
        document_index = _find_document_index(case, node_name)
        question_indexes = {
            **default_question_indexes,
            **_index_node_questions(case, node_name),
        }
        question_inputs = {
            question_id: _build_question_input(
                questions[question_indexes[question_id]]
            )
            for question_id in reference_answers
        }
        request = EvaluationRequest.model_validate(
            {"state": documents[document_index], "questions": question_inputs}
        )
        batches.append(
            _EvaluationBatch(
                request=request,
                expected_answers={
                    question_id: _select_reference_answer(reference_answer)
                    for question_id, reference_answer in reference_answers.items()
                },
            )
        )
    return batches


def _index_node_questions(case: _ViewerCase, node_name: str) -> dict[str, int]:
    """Map question IDs to the definitions displayed for one case node."""

    indexes = {}
    for model in case.models.values():
        for node in model.nodes:
            if node.node == node_name:
                indexes.update(node.questions)
    return indexes


def _find_document_index(case: _ViewerCase, node_name: str) -> int:
    """Find the state document read by a workflow node."""

    nodes = [node for model in case.models.values() for node in model.nodes]
    for node in nodes:
        if node.node == node_name and node.doc is not None:
            return node.doc
    for node in nodes:
        if node.doc is not None:
            return node.doc
    raise ValueError(f"no state document is published for node {node_name!r}")


def _build_question_input(
    question: _ViewerQuestion,
) -> BoolQuestionInput | EnumQuestionInput | ScoreQuestionInput:
    """Convert a TypeSafe viewer question to this project's input schema."""

    if question.type == "noul":
        criteria = question.criteria or {
            "true": "The condition holds",
            "false": "The condition does not hold",
        }
        if not isinstance(criteria, dict):
            raise ValueError("Noul criteria must be an object")
        return BoolQuestionInput(
            instructions=question.instructions,
            criteria=criteria,
        )
    if question.type == "choice":
        if not isinstance(question.criteria, dict):
            raise ValueError("Choice criteria must be an object")
        return EnumQuestionInput(
            instructions=question.instructions,
            criteria=question.criteria,
        )
    if question.type == "score":
        criteria = question.criteria
        if isinstance(criteria, dict):
            criteria = [description for _, description in criteria.items()]
        if not isinstance(criteria, list):
            raise ValueError("Score criteria must be a list")
        return ScoreQuestionInput(
            instructions=question.instructions,
            criteria=criteria,
        )
    raise ValueError(f"unsupported TypeSafe question type {question.type!r}")


def _select_reference_answer(answer: _ViewerReferenceAnswer) -> bool | str:
    """Select the top answer from the mean reference distribution."""

    distributions = [
        _build_reference_distribution(answer, item) for item in answer.sets
    ]
    option_names = dict.fromkeys(
        name for distribution in distributions for name in distribution
    )
    mean_probabilities = {
        name: sum(distribution.get(name, 0.0) for distribution in distributions)
        / len(distributions)
        for name in option_names
    }
    if answer.type == "noul":
        return mean_probabilities.get("true", 0.0) >= mean_probabilities.get(
            "false", 0.0
        )
    return max(mean_probabilities, key=mean_probabilities.__getitem__)


def _build_reference_distribution(
    answer: _ViewerReferenceAnswer, reference: _ViewerReferenceSet
) -> dict[str, float]:
    """Return a supplied distribution or a one-hot reference value."""

    if reference.probabilities:
        return reference.probabilities
    if answer.type == "noul":
        value = str(reference.value).lower() == "true"
        return {"true": float(value), "false": float(not value)}
    return {str(reference.value): 1.0}


def _download_typesafe_asset(workflow: TypeSafeWorkflow) -> str:
    """Download the public viewer asset for a TypeSafe workflow."""

    url = f"{_EVALS_BASE_URL}/{workflow.value}-cases.js"
    request = Request(url, headers={"User-Agent": "jev-at-home/0.1"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")
