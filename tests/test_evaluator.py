from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest
import torch

from jev_at_home.evaluator import TransformersEvaluator
from jev_at_home.schemas import EvaluationRequest, ScoreAnswer


class FakeTokenizer:
    pad_token_id = 0
    padding_side = "right"

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

    def __call__(self, prompts, *, add_special_tokens, padding, return_tensors):
        assert not add_special_tokens
        assert padding
        assert return_tensors == "pt"
        rows = [self.encode(prompt, add_special_tokens=False) for prompt in prompts]
        width = max(map(len, rows))
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@dataclass
class FakeOutput:
    logits: torch.Tensor


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, input_ids, attention_mask):
        self.calls += 1
        batch, width = input_ids.shape
        logits = torch.zeros((batch, width, 128))
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


def test_all_questions_use_one_forward_pass_and_return_distributions() -> None:
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

    assert model.calls == 1
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
