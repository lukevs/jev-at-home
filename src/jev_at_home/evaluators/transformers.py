"""Turn next-token logits into batched typed judgments."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from pydantic import JsonValue
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from jev_at_home.schemas import (
    Answer,
    BoolQuestionInput,
    ChoiceAnswer,
    Device,
    EvaluationRequest,
    EvaluationResult,
    Question,
    QuestionSpec,
    ScoreAnswer,
    ScoreQuestion,
    ScoreQuestionInput,
)

_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_MINIMUM_SHARED_PREFIX_TOKENS = 32


@dataclass(frozen=True)
class _ModelQuestion:
    """A prompt plus the token-level answer space used to score it."""

    name: str
    prompt: str
    token_ids: list[int]
    choice_names: tuple[Enum | bool | int, ...]
    choice_descriptions: tuple[str, ...]
    label_token_ids: tuple[int, ...]
    returns_score: bool


@dataclass(frozen=True)
class _TokenBatch:
    """Left-padded model inputs and their true token positions."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor


class TransformersEvaluator:
    """Evaluate questions by sharing state prefill across batched suffixes."""

    def __init__(
        self,
        *,
        model_name: str,
        tokenizer: PreTrainedTokenizerBase,
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
    ) -> TransformersEvaluator:
        """Load a Hugging Face causal LM and tokenizer for local evaluation."""

        selected_device = device or _select_default_device(torch)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

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
        self,
        request: EvaluationRequest,
        *,
        temperature: float = 1.0,
        batch_size: int | None = None,
    ) -> EvaluationResult:
        """Return distributions while encoding a shared prompt prefix once."""

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch size must be at least one")

        model_questions = _compile_questions(self.tokenizer, request)
        next_token_logits = self._predict_next_token_logits(
            model_questions,
            batch_size or len(model_questions),
        )
        probabilities = _calculate_answer_probabilities(
            model_questions,
            next_token_logits,
            temperature=temperature,
            device=self.device,
        )
        answers = _select_answers(model_questions, probabilities)
        return EvaluationResult(model=self.model_name, answers=answers)

    def _predict_next_token_logits(
        self, questions: list[_ModelQuestion], batch_size: int
    ) -> torch.Tensor:
        """Return next-token logits with shared-prefix reuse when worthwhile."""

        token_sequences = [question.token_ids for question in questions]
        shared_prefix_length = _measure_shared_prefix(token_sequences)

        with torch.inference_mode():
            if shared_prefix_length >= _MINIMUM_SHARED_PREFIX_TOKENS:
                return self._predict_with_shared_prefix(
                    token_sequences,
                    shared_prefix_length,
                    batch_size,
                )
            return self._predict_full_prompts(token_sequences, batch_size)

    def _predict_with_shared_prefix(
        self,
        token_sequences: list[list[int]],
        shared_prefix_length: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Prefill shared tokens once and branch into bounded suffix batches."""

        prefix_tokens = token_sequences[0][:shared_prefix_length]
        prefix_batch = _build_token_batch(
            [prefix_tokens],
            start_position=0,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        prefix_output = self.model(
            input_ids=prefix_batch.input_ids,
            attention_mask=prefix_batch.attention_mask,
            position_ids=prefix_batch.position_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        prefix_cache = prefix_output.past_key_values
        del prefix_output

        logits = []
        suffixes = [tokens[shared_prefix_length:] for tokens in token_sequences]
        for suffix_batch in _group_token_sequences(suffixes, batch_size):
            logits.append(
                self._predict_suffixes(
                    suffix_batch,
                    prefix_cache,
                    shared_prefix_length,
                )
            )
        return torch.cat(logits)

    def _predict_suffixes(
        self,
        suffixes: list[list[int]],
        prefix_cache: Any,
        shared_prefix_length: int,
    ) -> torch.Tensor:
        """Return logits for unique suffixes branching from one cached prefix."""

        suffix_batch = _build_token_batch(
            suffixes,
            start_position=shared_prefix_length,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        prefix_attention = torch.ones(
            (len(suffixes), shared_prefix_length),
            dtype=torch.long,
            device=self.device,
        )
        branched_cache = copy.deepcopy(prefix_cache)
        branched_cache.batch_repeat_interleave(len(suffixes))
        output = self.model(
            input_ids=suffix_batch.input_ids,
            attention_mask=torch.cat(
                (prefix_attention, suffix_batch.attention_mask), dim=1
            ),
            position_ids=suffix_batch.position_ids,
            past_key_values=branched_cache,
            use_cache=False,
            logits_to_keep=1,
        )
        return output.logits[:, -1, :]

    def _predict_full_prompts(
        self, token_sequences: list[list[int]], batch_size: int
    ) -> torch.Tensor:
        """Return logits in bounded batches when prompts share no useful prefix."""

        logits = []
        for prompt_batch in _group_token_sequences(token_sequences, batch_size):
            model_inputs = _build_token_batch(
                prompt_batch,
                start_position=0,
                pad_token_id=self.tokenizer.pad_token_id,
                device=self.device,
            )
            output = self.model(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                position_ids=model_inputs.position_ids,
                logits_to_keep=1,
            )
            logits.append(output.logits[:, -1, :])
        return torch.cat(logits)


def _measure_shared_prefix(token_sequences: list[list[int]]) -> int:
    """Return shared token count while leaving every prompt a nonempty suffix."""

    if len(token_sequences) < 2:
        return 0
    shared_limit = min(map(len, token_sequences)) - 1
    first_sequence = token_sequences[0]
    for index in range(shared_limit):
        if any(
            tokens[index] != first_sequence[index]
            for tokens in token_sequences[1:]
        ):
            return index
    return shared_limit


def _group_token_sequences(
    token_sequences: list[list[int]], batch_size: int
) -> list[list[list[int]]]:
    """Group token sequences into bounded batches without changing their order."""

    return [
        token_sequences[start : start + batch_size]
        for start in range(0, len(token_sequences), batch_size)
    ]


def _build_token_batch(
    token_sequences: list[list[int]],
    *,
    start_position: int,
    pad_token_id: int,
    device: Device,
) -> _TokenBatch:
    """Build a left-padded batch with positions unaffected by its padding."""

    width = max(map(len, token_sequences))
    shape = (len(token_sequences), width)
    input_ids = torch.full(shape, pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros(shape, dtype=torch.long, device=device)
    position_ids = torch.zeros(shape, dtype=torch.long, device=device)

    for row, tokens in enumerate(token_sequences):
        token_count = len(tokens)
        input_ids[row, -token_count:] = torch.tensor(
            tokens, dtype=torch.long, device=device
        )
        attention_mask[row, -token_count:] = 1
        position_ids[row, -token_count:] = torch.arange(
            start_position,
            start_position + token_count,
            dtype=torch.long,
            device=device,
        )
    return _TokenBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )


def _select_default_device(torch: Any) -> Device:
    """Return the fastest commonly available local inference device."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _compile_questions(
    tokenizer: PreTrainedTokenizerBase, request: EvaluationRequest
) -> list[_ModelQuestion]:
    """Build prompts and label tokens for every requested question."""

    model_questions: list[_ModelQuestion] = []

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
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        model_questions.append(
            _ModelQuestion(
                name=name,
                prompt=prompt,
                token_ids=prompt_ids,
                choice_names=choice_names,
                choice_descriptions=tuple(typed_question.criteria.values()),
                label_token_ids=tuple(
                    _resolve_label_token_id(tokenizer, prompt, prompt_ids, label)
                    for label in labels
                ),
                returns_score=isinstance(question, ScoreQuestionInput),
            )
        )

    return model_questions


def _calculate_answer_probabilities(
    questions: list[_ModelQuestion],
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
    questions: list[_ModelQuestion], probabilities: torch.Tensor
) -> dict[str, Answer]:
    """Return typed answers for the model questions and their probabilities."""

    answers: dict[str, Answer] = {}

    for row, question in enumerate(questions):
        values = probabilities[row, : len(question.choice_names)].tolist()
        answers[question.name] = _build_answer(question, values)

    return answers


def _build_answer(question: _ModelQuestion, probabilities: list[float]) -> Answer:
    """Build the typed answer for one model question."""

    if question.returns_score:
        levels = tuple(int(choice) for choice in question.choice_names)
        distribution = dict(zip(levels, probabilities, strict=True))
        return ScoreAnswer(
            score=sum(
                level * probability for level, probability in distribution.items()
            ),
            legend=dict(
                zip(levels, question.choice_descriptions, strict=True)
            ),
            probabilities=distribution,
        )

    choices = tuple(
        choice for choice in question.choice_names if isinstance(choice, (Enum, bool))
    )
    distribution = dict(zip(choices, probabilities, strict=True))
    return ChoiceAnswer[Enum | bool](
        choice=max(distribution, key=distribution.__getitem__),
        probabilities=distribution,
    )


def _build_candidate_token_tensors(
    questions: list[_ModelQuestion], device: Device
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


def _build_typed_question(
    name: str, question: QuestionSpec
) -> Question[Enum | bool] | ScoreQuestion:
    """Return the typed question represented by a JSON question input."""

    if isinstance(question, BoolQuestionInput):
        return Question[bool](
            instructions=question.instructions,
            criteria=question.criteria,
        )

    if isinstance(question, ScoreQuestionInput):
        return ScoreQuestion(
            instructions=question.instructions,
            criteria=dict(enumerate(question.criteria)),
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


def _build_prompt_messages[T: Enum | bool | int](
    state: JsonValue, instructions: str, criteria: dict[T, str]
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


def _format_choice(choice: Enum | bool | int) -> str:
    """Return the external text for a typed choice."""

    if isinstance(choice, Enum):
        return str(choice.value)
    return json.dumps(choice)


def _format_state(state: JsonValue) -> str:
    """Return a deterministic text representation of JSON state."""

    if isinstance(state, str):
        return state
    return json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False)


def _resolve_label_token_id(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    prompt_ids: list[int],
    label: str,
) -> int:
    """Return the token id for a one-token label following the prompt."""

    continued_ids = tokenizer.encode(prompt + label, add_special_tokens=False)

    if (
        continued_ids[: len(prompt_ids)] != prompt_ids
        or len(continued_ids) != len(prompt_ids) + 1
    ):
        raise ValueError(
            f"label {label!r} is not exactly one continuation token for this tokenizer"
        )

    return continued_ids[-1]
