from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from jev_at_home.evaluators.transformers import (
    TransformersEvaluator,
    _build_token_batch,
    _compile_questions,
)
from jev_at_home.schemas import EvaluationRequest, ScoreAnswer


class FakeTokenizer:
    pad_token_id = 0
    padding_side = "left"

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        assert not tokenize
        assert add_generation_prompt
        assert not enable_thinking
        return "\n".join(message["content"] for message in messages) + "\nANSWER:"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(character) for character in text]


class FakeCache:
    def __init__(self, repeated_batches: list[int]) -> None:
        self.repeated_batches = repeated_batches

    def __deepcopy__(self, memo):
        return FakeCache(self.repeated_batches)

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.repeated_batches.append(repeats)


@dataclass
class FakeOutput:
    logits: torch.Tensor
    past_key_values: FakeCache | None = None


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0
        self.repeated_batches = []

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        position_ids,
        logits_to_keep,
        past_key_values=None,
        use_cache=None,
    ):
        self.calls += 1
        assert logits_to_keep == 1
        batch, _ = input_ids.shape
        logits = torch.zeros((batch, 1, 128))
        if use_cache:
            return FakeOutput(logits, FakeCache(self.repeated_batches))

        assert past_key_values is not None
        logits[0, :, ord("A")] = 1.0
        logits[0, :, ord("B")] = 3.0
        logits[1, :, ord("A")] = 4.0
        logits[1, :, ord("B")] = 2.0
        logits[1, :, ord("C")] = 1.0
        logits[2, :, ord("A")] = 5.0
        logits[2, :, ord("B")] = 1.0
        logits[3, :, ord("A")] = 1.0
        logits[3, :, ord("B")] = 2.0
        logits[3, :, ord("C")] = 4.0
        return FakeOutput(logits)


def test_all_questions_share_prefix_prefill_and_return_distributions() -> None:
    request = EvaluationRequest.model_validate(
        {
            "state": "A customer message",
            "questions": {
                "sentiment": {
                    "type": "enum",
                    "instructions": "What is the sentiment?",
                    "criteria": {
                        "positive": "Positive",
                        "negative": "Negative",
                    },
                },
                "priority": {
                    "type": "enum",
                    "instructions": "What is the priority?",
                    "criteria": {
                        "high": "Act now",
                        "medium": "Act soon",
                        "low": "Can wait",
                    },
                },
                "actionable": {
                    "type": "bool",
                    "instructions": "Does this need action?",
                    "criteria": {
                        "true": "Action is required",
                        "false": "No action is required",
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
    model = FakeModel()
    evaluator = TransformersEvaluator(
        model_name="fake", tokenizer=FakeTokenizer(), model=model, device="cpu"
    )

    result = evaluator.evaluate(request)

    assert model.calls == 2
    assert model.repeated_batches == [4]
    sentiment = result.answers["sentiment"].choice
    priority = result.answers["priority"].choice
    assert isinstance(sentiment, Enum)
    assert isinstance(priority, Enum)
    assert sentiment.value == "negative"
    assert priority.value == "high"
    assert result.answers["actionable"].choice is True
    assert set(result.answers["actionable"].probabilities) == {True, False}
    severity = result.answers["severity"]
    assert isinstance(severity, ScoreAnswer)
    assert severity.score == pytest.approx(
        sum(
            level * probability
            for level, probability in severity.probabilities.items()
        )
    )
    assert severity.legend == {0: "Minor", 1: "Moderate", 2: "Severe"}
    for answer in result.answers.values():
        assert sum(answer.probabilities.values()) == pytest.approx(1.0)


def test_shared_prefix_logits_match_full_qwen_prompts() -> None:
    request = EvaluationRequest.model_validate(
        {
            "state": "One shared state",
            "questions": {
                "first": {
                    "type": "bool",
                    "instructions": "Is the first condition true?",
                    "criteria": {"true": "Yes", "false": "No"},
                },
                "second": {
                    "type": "bool",
                    "instructions": "Is the second condition true?",
                    "criteria": {"true": "Yes", "false": "No"},
                },
            },
        }
    )
    tokenizer = FakeTokenizer()
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    ).eval()
    evaluator = TransformersEvaluator(
        model_name="tiny-qwen",
        tokenizer=tokenizer,
        model=model,
        device="cpu",
    )
    questions = _compile_questions(tokenizer, request)
    token_sequences = [
        tokenizer.encode(question.prompt, add_special_tokens=False)
        for question in questions
    ]
    full_batch = _build_token_batch(
        token_sequences,
        start_position=0,
        pad_token_id=tokenizer.pad_token_id,
        device="cpu",
    )

    with torch.inference_mode():
        expected = model(
            input_ids=full_batch.input_ids,
            attention_mask=full_batch.attention_mask,
            position_ids=full_batch.position_ids,
            logits_to_keep=1,
        ).logits[:, -1, :]
        actual = evaluator._predict_next_token_logits(questions, batch_size=2)

    torch.testing.assert_close(actual, expected)
