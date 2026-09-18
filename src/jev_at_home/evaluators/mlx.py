"""Evaluate Qwen3 with MLX's Metal kernels and reusable prefix caches."""

from __future__ import annotations

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.qwen3 import Model
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from jev_at_home.evaluators.transformers import (
    _MINIMUM_SHARED_PREFIX_TOKENS,
    _build_answer,
    _compile_questions,
    _group_token_sequences,
    _measure_shared_prefix,
    _ModelQuestion,
)
from jev_at_home.schemas import EvaluationRequest, EvaluationResult


class MLXEvaluator:
    """Run the same classifier prompts with unquantized Qwen3 weights on MLX."""

    def __init__(
        self,
        *,
        model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        model: Model,
    ) -> None:
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        # Explicitly match Qwen's configured query/key normalization epsilon.
        for layer in model.layers:
            layer.self_attn.q_norm.eps = model.args.rms_norm_eps
            layer.self_attn.k_norm.eps = model.args.rms_norm_eps

    @classmethod
    def load(cls, model_name: str) -> MLXEvaluator:
        """Load the existing Hugging Face weights without quantizing them."""

        if AutoConfig.from_pretrained(model_name).model_type != "qwen3":
            raise ValueError("the MLX backend currently supports Qwen3 models")
        model, _ = load(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(model_name=model_name, tokenizer=tokenizer, model=model)

    def evaluate(
        self,
        request: EvaluationRequest,
        *,
        temperature: float = 1.0,
        batch_size: int | None = None,
    ) -> EvaluationResult:
        """Score each question's last token and preserve the typed answers."""

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch size must be at least one")
        questions = _compile_questions(self.tokenizer, request)
        logits = self._predict_next_token_logits(
            questions, batch_size or len(questions)
        )
        answers = {}
        for row, question in enumerate(questions):
            candidates = logits[row, mx.array(question.label_token_ids)]
            probabilities = mx.softmax(candidates.astype(mx.float32) / temperature)
            answers[question.name] = _build_answer(question, probabilities.tolist())
        return EvaluationResult(model=self.model_name, answers=answers)

    def _predict_next_token_logits(
        self, questions: list[_ModelQuestion], batch_size: int
    ) -> mx.array:
        """Prefill once, then score bounded batches of right-padded suffixes."""

        sequences = [question.token_ids for question in questions]
        prefix_length = _measure_shared_prefix(sequences)
        if prefix_length < _MINIMUM_SHARED_PREFIX_TOKENS:
            prefix_length = 0
        prefix_cache = make_prompt_cache(self.model)
        if prefix_length:
            self.model.model(
                mx.array([sequences[0][:prefix_length]]), cache=prefix_cache
            )
            mx.eval([layer.state for layer in prefix_cache])

        caches = {1: prefix_cache}
        suffixes = [sequence[prefix_length:] for sequence in sequences]
        logits = []
        for suffix_batch in _group_token_sequences(suffixes, batch_size):
            width = len(suffix_batch)
            if width not in caches:
                caches[width] = _expand_prefix_cache(prefix_cache, width)
            logits.append(self._predict_suffixes(suffix_batch, caches[width]))
        return mx.concatenate(logits)

    def _predict_suffixes(
        self, suffixes: list[list[int]], cache: list[KVCache]
    ) -> mx.array:
        """Read each row's true endpoint, then rewind the reusable suffix slots."""

        lengths = [len(suffix) for suffix in suffixes]
        padded_length = max(lengths)
        tokens = mx.array(
            [
                suffix + [self.tokenizer.pad_token_id] * (padded_length - len(suffix))
                for suffix in suffixes
            ]
        )
        hidden = self.model.model(tokens, cache=cache)
        # Right padding cannot affect earlier tokens under causal attention.
        last_hidden = hidden[mx.arange(len(suffixes)), mx.array(lengths) - 1]
        if self.model.args.tie_word_embeddings:
            logits = self.model.model.embed_tokens.as_linear(last_hidden)
        else:
            logits = self.model.lm_head(last_hidden)
        # Finish lazy computation before rewinding or reusing the cache.
        mx.eval(logits)
        for layer in cache:
            layer.trim(padded_length)
        return logits


def _expand_prefix_cache(prefix: list[KVCache], width: int) -> list[KVCache]:
    """Copy the prefix once for each batch width used within this request."""

    cache = [KVCache() for _ in prefix]
    for source, target in zip(prefix, cache, strict=True):
        if not source.empty():
            target.state = tuple(
                mx.repeat(value, width, axis=0) for value in source.state
            )
    mx.eval([layer.state for layer in cache if not layer.empty()])
    return cache
