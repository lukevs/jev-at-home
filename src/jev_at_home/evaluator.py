"""Turn next-token logits into batched enum judgments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jev_at_home.schemas import (
    BoolQuestionInput,
    ChoiceAnswer,
    Device,
    EvaluationRequest,
    EvaluationResult,
    Question,
    QuestionSpec,
)

_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class _ModelQuestion[T: Enum | bool]:
    """A prompt plus the token-level answer space used to score it."""

    name: str
    prompt: str
    choice_names: tuple[T, ...]
    label_token_ids: tuple[int, ...]


class TransformersChoiceEvaluator:
    """Evaluate every question in one batched causal-LM forward pass."""

    def __init__(
        self,
        *,
        model_name: str,
        tokenizer: Any,
        model: Any,
        device: Device,
    ) -> None:
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @classmethod
    def load(
        cls, model_name: str, device: Device | None = None
    ) -> TransformersChoiceEvaluator:
        """Load a Hugging Face causal LM and tokenizer for local evaluation."""

        selected_device = device or _select_default_device(torch)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(selected_device)
        model.eval()
        return cls(
            model_name=model_name,
            tokenizer=tokenizer,
            model=model,
            device=selected_device,
        )

    def evaluate(
        self, request: EvaluationRequest, *, temperature: float = 1.0
    ) -> EvaluationResult:
        """Return distributions for all questions using exactly one model call."""

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        model_questions = _compile_questions(self.tokenizer, request)
        next_token_logits = self._predict_next_token_logits(model_questions)
        probabilities = _calculate_choice_probabilities(
            model_questions,
            next_token_logits,
            temperature=temperature,
            device=self.device,
        )
        answers = _select_answers(model_questions, probabilities)
        return EvaluationResult(model=self.model_name, answers=answers)

    def _predict_next_token_logits(
        self, questions: list[_ModelQuestion[Enum | bool]]
    ) -> torch.Tensor:
        """Return each question's next-token logits from one model call."""

        model_inputs = self._tokenize_questions(questions)

        with torch.inference_mode():
            logits = self.model(**model_inputs).logits

        last_token_indices = model_inputs["attention_mask"].sum(dim=1) - 1
        row_indices = torch.arange(len(questions), device=self.device)
        return logits[row_indices, last_token_indices]

    def _tokenize_questions(
        self, questions: list[_ModelQuestion[Enum | bool]]
    ) -> dict[str, torch.Tensor]:
        """Return a padded tensor batch for the model questions."""

        inputs = self.tokenizer(
            [question.prompt for question in questions],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        return {name: tensor.to(self.device) for name, tensor in inputs.items()}


def _select_default_device(torch: Any) -> Device:
    """Return the fastest commonly available local inference device."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _compile_questions(
    tokenizer: Any, request: EvaluationRequest
) -> list[_ModelQuestion[Enum | bool]]:
    """Build prompts and label tokens for every requested question."""

    model_questions: list[_ModelQuestion[Enum | bool]] = []

    for name, question in request.questions.items():
        typed_question = _build_typed_question(name, question)
        prompt = tokenizer.apply_chat_template(
            _build_prompt_messages(
                request.state,
                typed_question.instructions,
                typed_question.criteria,
            ),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        choice_names = tuple(typed_question.criteria)
        labels = _LABELS[: len(choice_names)]

        model_questions.append(
            _ModelQuestion(
                name=name,
                prompt=prompt,
                choice_names=choice_names,
                label_token_ids=tuple(
                    _resolve_label_token_id(tokenizer, prompt, label)
                    for label in labels
                ),
            )
        )

    return model_questions


def _calculate_choice_probabilities(
    questions: list[_ModelQuestion[Enum | bool]],
    next_token_logits: torch.Tensor,
    *,
    temperature: float,
    device: Device,
) -> torch.Tensor:
    """Return each question's distribution over its declared choices."""

    candidate_ids, candidate_mask = _build_candidate_token_tensors(questions, device)
    candidate_logits = next_token_logits.gather(dim=1, index=candidate_ids)
    candidate_logits = candidate_logits.masked_fill(~candidate_mask, float("-inf"))
    return torch.softmax(candidate_logits.float() / temperature, dim=1).cpu()


def _select_answers(
    questions: list[_ModelQuestion[Enum | bool]], probabilities: torch.Tensor
) -> dict[str, ChoiceAnswer[Enum | bool]]:
    """Return typed answers for the model questions and their probabilities."""

    answers: dict[str, ChoiceAnswer[Enum | bool]] = {}

    for row, question in enumerate(questions):
        values = probabilities[row, : len(question.choice_names)].tolist()
        distribution = dict(zip(question.choice_names, values, strict=True))
        answers[question.name] = ChoiceAnswer[Enum | bool](
            choice=question.choice_names[
                max(range(len(values)), key=values.__getitem__)
            ],
            probabilities=distribution,
        )

    return answers


def _build_candidate_token_tensors(
    questions: list[_ModelQuestion[Enum | bool]], device: Device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded candidate-token IDs and their valid-entry mask."""

    max_choices = max(len(question.label_token_ids) for question in questions)
    candidate_ids = torch.zeros(
        (len(questions), max_choices), dtype=torch.long, device=device
    )
    candidate_mask = torch.zeros(
        (len(questions), max_choices), dtype=torch.bool, device=device
    )

    for row, question in enumerate(questions):
        width = len(question.label_token_ids)
        candidate_ids[row, :width] = torch.tensor(
            question.label_token_ids, dtype=torch.long, device=device
        )
        candidate_mask[row, :width] = True

    return candidate_ids, candidate_mask


def _build_typed_question(name: str, question: QuestionSpec) -> Question[Enum | bool]:
    """Return the typed question represented by a JSON question input."""

    if isinstance(question, BoolQuestionInput):
        return Question[bool](
            instructions=question.instructions,
            criteria=question.criteria,
        )

    enum_type = Enum(
        f"{name.title().replace('_', '') or 'Anonymous'}Choice",
        {
            f"OPTION_{index}": choice_name
            for index, choice_name in enumerate(question.criteria)
        },
    )

    criteria = {
        enum_type[f"OPTION_{index}"]: description
        for index, description in enumerate(question.criteria.values())
    }

    return Question[enum_type](
        instructions=question.instructions,
        criteria=criteria,
    )


def _build_prompt_messages[T: Enum | bool](
    state: Any, instructions: str, criteria: dict[T, str]
) -> list[dict[str, str]]:
    """Create a self-contained classification conversation for one question."""

    choices = "\n".join(
        f"{_LABELS[index]}. {_format_choice(name)}: {description}"
        for index, (name, description) in enumerate(criteria.items())
    )

    return [
        {
            "role": "system",
            "content": (
                "Classify the state using exactly one listed choice. "
                "Reply with only its single-letter label."
            ),
        },
        {
            "role": "user",
            "content": (
                f"State:\n{_format_state(state)}\n\n"
                f"Question:\n{instructions}\n\nChoices:\n{choices}"
            ),
        },
    ]


def _format_choice(choice: Enum | bool) -> str:
    """Return the external text for a typed choice."""

    if isinstance(choice, Enum):
        return str(choice.value)
    return json.dumps(choice)


def _format_state(state: Any) -> str:
    """Return a deterministic text representation of JSON state."""

    if isinstance(state, str):
        return state
    return json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False)


def _resolve_label_token_id(tokenizer: Any, prompt: str, label: str) -> int:
    """Return the token id for a one-token label following the prompt."""

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    continued_ids = tokenizer.encode(prompt + label, add_special_tokens=False)

    if (
        continued_ids[: len(prompt_ids)] != prompt_ids
        or len(continued_ids) != len(prompt_ids) + 1
    ):
        raise ValueError(
            f"label {label!r} is not exactly one continuation token for this tokenizer"
        )

    return continued_ids[-1]
