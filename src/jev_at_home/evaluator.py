"""Turn next-token logits into batched enum judgments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jev_at_home.domain import ChoiceAnswer, EvaluationRequest, EvaluationResult


LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class PreparedQuestion[T: str | bool]:
    """A prompt plus the token-level answer space used to score it."""

    name: str
    prompt: str
    choice_names: tuple[T, ...]
    label_token_ids: tuple[int, ...]


def render_state(state: Any) -> str:
    """Represent JSON state deterministically for inclusion in a prompt."""

    if isinstance(state, str):
        return state
    return json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False)


def question_messages[T: str | bool](
    state: Any, instructions: str, criteria: dict[T, str]
) -> list[dict[str, str]]:
    """Create a self-contained classification conversation for one question."""

    choices = "\n".join(
        f"{LABELS[index]}. {name}: {description}"
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
                f"State:\n{render_state(state)}\n\n"
                f"Question:\n{instructions}\n\nChoices:\n{choices}"
            ),
        },
    ]


def continuation_token_id(tokenizer: Any, prompt: str, label: str) -> int:
    """Return a label's token id, rejecting tokenizers that split the label."""

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    continued_ids = tokenizer.encode(prompt + label, add_special_tokens=False)
    if continued_ids[: len(prompt_ids)] != prompt_ids or len(continued_ids) != len(
        prompt_ids
    ) + 1:
        raise ValueError(
            f"label {label!r} is not exactly one continuation token for this tokenizer"
        )
    return continued_ids[-1]


def prepare_questions(
    tokenizer: Any, request: EvaluationRequest
) -> list[PreparedQuestion[str | bool]]:
    """Compile domain questions into prompts and one-token answer spaces."""

    prepared: list[PreparedQuestion[str | bool]] = []
    for name, question in request.questions.items():
        prompt = tokenizer.apply_chat_template(
            question_messages(request.state, question.instructions, question.criteria),
            tokenize=False,
            add_generation_prompt=True,
        )
        choice_names = tuple(question.criteria)
        labels = LABELS[: len(choice_names)]
        prepared.append(
            PreparedQuestion(
                name=name,
                prompt=prompt,
                choice_names=choice_names,
                label_token_ids=tuple(
                    continuation_token_id(tokenizer, prompt, label) for label in labels
                ),
            )
        )
    return prepared


def choose_device(torch: Any) -> str:
    """Choose the fastest commonly available local inference device."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TransformersChoiceEvaluator:
    """Evaluate every question in one batched causal-LM forward pass."""

    def __init__(
        self,
        *,
        model_name: str,
        tokenizer: Any,
        model: Any,
        device: str,
    ) -> None:
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @classmethod
    def from_pretrained(
        cls, model_name: str, device: str | None = None
    ) -> TransformersChoiceEvaluator:
        """Load a Hugging Face causal LM and tokenizer for local evaluation."""

        selected_device = device or choose_device(torch)
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

        prepared = prepare_questions(self.tokenizer, request)
        encoded = self.tokenizer(
            [question.prompt for question in prepared],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}

        with torch.inference_mode():
            # This is the only transformer invocation: every question is one row.
            logits = self.model(**encoded).logits

        last_token_indices = encoded["attention_mask"].sum(dim=1) - 1
        row_indices = torch.arange(len(prepared), device=self.device)
        next_token_logits = logits[row_indices, last_token_indices]

        max_choices = max(len(question.label_token_ids) for question in prepared)
        candidate_ids = torch.zeros(
            (len(prepared), max_choices), dtype=torch.long, device=self.device
        )
        candidate_mask = torch.zeros(
            (len(prepared), max_choices), dtype=torch.bool, device=self.device
        )
        for row, question in enumerate(prepared):
            width = len(question.label_token_ids)
            candidate_ids[row, :width] = torch.tensor(
                question.label_token_ids, dtype=torch.long, device=self.device
            )
            candidate_mask[row, :width] = True

        candidate_logits = next_token_logits.gather(dim=1, index=candidate_ids)
        candidate_logits = candidate_logits.masked_fill(~candidate_mask, float("-inf"))
        probabilities = torch.softmax(
            candidate_logits.float() / temperature, dim=1
        ).cpu()

        answers: dict[str, ChoiceAnswer[str | bool]] = {}
        for row, question in enumerate(prepared):
            values = probabilities[row, : len(question.choice_names)].tolist()
            distribution = dict(zip(question.choice_names, values, strict=True))
            answers[question.name] = ChoiceAnswer[str | bool](
                choice=question.choice_names[max(range(len(values)), key=values.__getitem__)],
                probabilities=distribution,
            )
        return EvaluationResult(model=self.model_name, answers=answers)
