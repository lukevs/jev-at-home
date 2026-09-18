from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from jev_at_home.evaluators.transformers import _ModelQuestion

mx = pytest.importorskip("mlx.core")
qwen = pytest.importorskip("mlx_lm.models.qwen3")
MLXEvaluator = import_module("jev_at_home.evaluators.mlx").MLXEvaluator


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
@pytest.mark.parametrize("prefix_length", [0, 40])
@pytest.mark.parametrize("tie_embeddings", [False, True])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_mlx_matches_independent_torch_prompts(
    batch_size: int, prefix_length: int, tie_embeddings: bool, device: str
) -> None:
    """Exercise cache reuse, padding, partial batches, and both output heads."""

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        tie_word_embeddings=tie_embeddings,
    )
    reference = Qwen3ForCausalLM(config).eval()
    model = qwen.Model(
        qwen.ModelArgs(
            model_type="qwen3",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=128,
            rms_norm_eps=1e-6,
            max_position_embeddings=32768,
            rope_theta=10000,
            tie_word_embeddings=tie_embeddings,
        )
    )
    weights = {
        name: mx.array(value.numpy()) for name, value in reference.state_dict().items()
    }
    model.load_weights(list(model.sanitize(weights).items()))
    evaluator = MLXEvaluator(
        model_name="tiny", tokenizer=SimpleNamespace(pad_token_id=0), model=model
    )
    prefix = list(range(1, prefix_length + 1))
    sequences = [
        prefix + suffix
        for suffix in (
            [41, 42, 43, 44],
            [51],
            [61, 62],
            [71, 72, 73],
            [81],
        )
    ]
    questions = [
        _ModelQuestion(
            name=str(index),
            prompt="",
            token_ids=sequence,
            choice_names=(True, False),
            choice_descriptions=("Yes", "No"),
            label_token_ids=(index + 2, 1) if index % 2 else (1, index + 2),
            returns_score=False,
        )
        for index, sequence in enumerate(sequences)
    ]
    with torch.inference_mode():
        expected = torch.cat(
            [
                reference(input_ids=torch.tensor([sequence]), logits_to_keep=1).logits[
                    :, -1, :
                ]
                for sequence in sequences
            ]
        ).numpy()
    with mx.stream(getattr(mx, device)):
        actual = np.array(evaluator._predict_choice_logits(questions, batch_size))
    expected = np.array(
        [
            row[list(question.label_token_ids)]
            for row, question in zip(expected, questions, strict=True)
        ]
    )
    # CPU verifies the equations strictly; Metal's float32 kernels round
    # differently (observed maximum error below 3e-4 for these same weights).
    tolerance = 1e-5 if device == "cpu" else 5e-4
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=1e-4)


def test_plan_batches_limits_padding_and_restores_original_indexes() -> None:
    evaluator = object.__new__(MLXEvaluator)
    suffixes = [[1] * length for length in (1, 1000, 10, 900, 3000, 800)]
    batches = evaluator._plan_batches(suffixes, batch_size=3)
    assert batches == [[4], [1], [5, 3], [0, 2]]
    assert sorted(index for batch in batches for index in batch) == list(range(6))
    for batch in batches:
        assert len(batch) <= 3
        assert (
            len(batch) == 1
            or len(batch) * max(len(suffixes[index]) for index in batch) <= 2048
        )


def test_project_choices_matches_full_vocabulary_in_bfloat16() -> None:
    nn = import_module("mlx.nn")
    head = nn.Linear(16, 128, bias=False)
    head.set_dtype(mx.bfloat16)
    evaluator = object.__new__(MLXEvaluator)
    evaluator.model = SimpleNamespace(
        args=SimpleNamespace(tie_word_embeddings=False), lm_head=head
    )
    hidden = mx.random.normal((3, 16)).astype(mx.bfloat16)
    tokens = [9, 1, 70]
    expected = head(hidden)[:, mx.array(tokens)].astype(mx.float32)
    actual = evaluator._project_choices(hidden, tokens).astype(mx.float32)
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=0.01, rtol=0.01
    )


@pytest.mark.parametrize(
    ("lengths", "batch_size", "expected"),
    [
        ([], 4, []),
        ([1, 2, 100], 2, [[2], [0, 1]]),
        ([1024, 1024], 8, [[0, 1]]),
        ([1024, 1025], 8, [[1], [0]]),
        ([3000, 3000], 8, [[1], [0]]),
        ([2, 1, 3], 1, [[0], [1], [2]]),
    ],
)
def test_plan_batches_handles_partial_and_oversized_batches(
    lengths: list[int], batch_size: int, expected: list[list[int]]
) -> None:
    evaluator = object.__new__(MLXEvaluator)
    assert (
        evaluator._plan_batches([[1] * length for length in lengths], batch_size)
        == expected
    )
