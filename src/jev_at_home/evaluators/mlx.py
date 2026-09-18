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
    _measure_shared_prefix,
    _ModelQuestion,
)
from jev_at_home.schemas import EvaluationRequest, EvaluationResult

_MAXIMUM_BATCH_TOKENS = 2048


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
        # MLX-LM 0.31 defaults these two norms to 1e-5, ignoring Qwen's config.
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
        logits = self._predict_choice_logits(questions, batch_size or len(questions))
        answers = {}
        for row, question in enumerate(questions):
            probabilities = mx.softmax(logits[row].astype(mx.float32) / temperature)
            answers[question.name] = _build_answer(question, probabilities.tolist())
        return EvaluationResult(model=self.model_name, answers=answers)

    def _predict_choice_logits(
        self, questions: list[_ModelQuestion], batch_size: int
    ) -> list[mx.array]:
        """Score length-bucketed suffixes and restore each question's choice order."""

        sequences = [question.token_ids for question in questions]
        prefix_length = _measure_shared_prefix(sequences)
        if prefix_length < _MINIMUM_SHARED_PREFIX_TOKENS:
            prefix_length = 0
        prefix_cache = self._prefill_prefix(sequences[0][:prefix_length])
        batch_cache = prefix_cache
        cached_width = 1
        suffixes = [sequence[prefix_length:] for sequence in sequences]
        label_token_ids = list(
            dict.fromkeys(token for q in questions for token in q.label_token_ids)
        )
        columns = {token: column for column, token in enumerate(label_token_ids)}
        indexes = []
        logits = []
        for batch_indexes in self._plan_batches(suffixes, batch_size):
            suffix_batch = [suffixes[index] for index in batch_indexes]
            width = len(suffix_batch)
            if width != cached_width:
                # Release the previous replica before allocating a new width.
                batch_cache = prefix_cache
                if width > 1:
                    batch_cache = _expand_prefix_cache(prefix_cache, width)
                cached_width = width
            logits.append(
                self._predict_suffixes(suffix_batch, batch_cache, label_token_ids)
            )
            indexes.extend(batch_indexes)
        rows = {index: row for row, index in enumerate(indexes)}
        predictions = mx.concatenate(logits)
        return [
            predictions[
                rows[index], mx.array([columns[token] for token in q.label_token_ids])
            ]
            for index, q in enumerate(questions)
        ]

    def _prefill_prefix(self, tokens: list[int]) -> list[KVCache]:
        """Encode the common prefix without computing unused vocabulary logits."""

        cache = make_prompt_cache(self.model)
        if tokens:
            self.model.model(mx.array([tokens]), cache=cache)
            mx.eval([layer.state for layer in cache])
        return cache

    def _plan_batches(
        self, suffixes: list[list[int]], batch_size: int
    ) -> list[list[int]]:
        """Group similar lengths within row/token caps; allow oversized singletons."""

        if batch_size == 1:
            return [[index] for index in range(len(suffixes))]
        indexes = sorted(range(len(suffixes)), key=lambda index: len(suffixes[index]))
        batches: list[list[int]] = []
        for index in indexes:
            if (
                not batches
                or len(batches[-1]) == batch_size
                or (
                    (len(batches[-1]) + 1) * len(suffixes[index])
                    > _MAXIMUM_BATCH_TOKENS
                )
            ):
                batches.append([])
            batches[-1].append(index)
        # Form short batches first to avoid padding the final partial batch
        # up to the longest suffix. Execute long batches first to grow caches once.
        return batches[::-1]

    def _predict_suffixes(
        self,
        suffixes: list[list[int]],
        cache: list[KVCache],
        label_token_ids: list[int],
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
        logits = self._project_choices(last_hidden, label_token_ids)
        # Finish lazy computation before rewinding or reusing the cache.
        mx.eval(logits)
        for layer in cache:
            layer.trim(padded_length)
        return logits

    def _project_choices(
        self, hidden: mx.array, label_token_ids: list[int]
    ) -> mx.array:
        """Project only requested vocabulary rows, preserving their token order."""

        head = (
            self.model.model.embed_tokens
            if self.model.args.tie_word_embeddings
            else self.model.lm_head
        )
        if hasattr(head, "scales"):
            raise ValueError("candidate projection requires unquantized model weights")
        weights = head.weight[mx.array(label_token_ids)]
        return hidden @ weights.T


def _expand_prefix_cache(prefix: list[KVCache], width: int) -> list[KVCache]:
    """Replicate the prefix when the active batch width changes."""

    cache = [KVCache() for _ in prefix]
    for source, target in zip(prefix, cache, strict=True):
        if not source.empty():
            target.state = tuple(
                mx.repeat(value, width, axis=0) for value in source.state
            )
    mx.eval([layer.state for layer in cache if not layer.empty()])
    return cache
