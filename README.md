# jev-at-home

A small experiment in Jev-shaped inference with an ordinary open-source causal
language model. It evaluates many independent enum, boolean, or score questions
with **one shared-prefix prefill and bounded suffix batches**, restricts each
row's next-token logits to single-token labels (`A`, `B`, ...), applies softmax,
and maps the distribution back to the caller's enum.

The repo is meant to show that much of the shape and latency advantage can 
be demonstrated by replacing autoregressive structured generation with 
batched classification over a closed answer space.

This is _not_ a reimplementation of TypeSafe's Jev model. TypeSafe publicly says
Jev uses a new architecture, parallel sampler, and Reinforcement Learning for
Calibrated Decisions, but has not published enough implementation detail to
reproduce those pieces.

## Example run

```console
$ uv run jev-at-home judge examples/support.json --device mps

Backend: transformers
Model: Qwen/Qwen3-4B-Instruct-2507
Inference: 194.6 ms · 3 questions · 1 batch
State: {"customer_tier": "business", "message": "Help! My payouts have been failing for three days."}

Which team should handle this?
✓ billing     100.0%
  technical     0.0%
  sales         0.0%

Does this request need urgent attention?
✓ true        100.0%
  false         0.0%

What is the customer's sentiment?
  positive      0.0%
  neutral       0.0%
✓ negative    100.0%
```

## How it works

```text
shared state ──> common prompt prefix ──> one KV-cache prefill
                                               │
N typed questions ──> unique prompt suffixes ──┤
                                               │
                                  bounded suffix batches
                                               │
                                final-token logits per row
                                               │
                                  select A/B/C/... logits
                                               │
                                     softmax + typed mapping
```

The probabilities are normalized **over the declared choices**. They are
useful scores, but they are not calibrated confidence estimates.

## Run it

The default model is the Apache-2.0
[`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
The first run downloads its roughly 8.1 GB of BF16 weights. Hugging Face caches
model snapshots under `~/.cache/huggingface/hub` for future runs.

```bash
uv sync
uv run jev-at-home judge examples/support.json
```

On Apple Silicon, an optional MLX backend runs the same Qwen3 weights with
Metal kernels and reuses the prefix cache between suffix batches:

```bash
uv run --extra mlx jev-at-home judge examples/support.json --backend mlx
uv run --extra mlx jev-at-home typesafe-eval --backend mlx --batch-size 1
```

This backend currently supports Qwen3 and keeps the original weight precision;
it does not quantize or change the prompts. The default Transformers backend
remains available on CPU, MPS, and CUDA. Both validate each answer label as an
exact single-token continuation. Prompt token IDs are computed once and reused
for that validation and inference.

MLX groups similarly sized suffixes within the requested batch size and a
2,048-padded-suffix-token budget. Longer individual questions run alone; they
are never truncated. It right-pads each batch and reads each question's actual
last token, so padding cannot affect earlier tokens under causal attention.
Answers and choices are restored to the caller's original order.

The backend retains the original prefix and only one active batch cache.
Suffix storage is reused after resetting its cursor; a new prefix replica is
allocated only when the batch width changes. Only the last hidden state per
question is projected, using just the requested label-token rows of the output
weights instead of the full vocabulary. Weights remain unquantized. The backend
also explicitly uses Qwen's configured query/key normalization epsilon.
Floating-point results can still differ between runtimes or batch shapes and
change close decisions.

Or use standard input:

```bash
uv run jev-at-home example | uv run jev-at-home judge
```

Select another instruct-tuned causal model or force a device with:

```bash
uv run jev-at-home judge examples/support.json \
  --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --device mps
```

## Run TypeSafe's public workflow examples

TypeSafe's eval viewer publishes five showcased cases for each of its four
workflows. Run all of those examples against the local model with:

```bash
uv run jev-at-home typesafe-eval --device mps
```

Run one workflow or use fewer cases while iterating:

```bash
uv run jev-at-home typesafe-eval \
  --workflow security_incidents \
  --limit 1 \
  --batch-size 4 \
  --device mps
```

The runner downloads the public viewer assets from
[`evals.typesafe.ai`](https://evals.typesafe.ai/), converts Noul, Choice, and
Score questions to this project's boolean, enum, and score schemas, and reports
question-level agreement with the published two-model consensus.

## Data

Input supports string enums, booleans, and ordered scores. Each question has an explicit
type discriminator:

```json
{
  "state": "The unstructured or JSON state to inspect",
  "questions": {
    "question_name": {
      "type": "enum",
      "instructions": "One atomic question",
      "criteria": {
        "enum_value_a": "What A means",
        "enum_value_b": "What B means"
      }
    }
  }
}
```

A boolean question maps its key directly to `true` or `false` in the result:

```json
{
  "type": "bool",
  "instructions": "Does this need urgent attention?",
  "criteria": {
    "true": "Time-sensitive or blocking",
    "false": "No urgency is expressed"
  }
}
```

A score question declares an ordered list of levels. Levels are numbered from
zero, and the returned score is the probability-weighted mean of those level
numbers. The result also includes the complete level distribution and legend.

```json
{
  "type": "score",
  "instructions": "How severe is the service impact?",
  "criteria": [
    "Minor: cosmetic or isolated impact",
    "Moderate: degraded service with a workaround",
    "Severe: a critical path is unavailable for many users"
  ]
}
```

### Qwen3-4B public example results

The Qwen results below were measured with `Qwen/Qwen3-4B-Instruct-2507`, the
optional MLX backend, and batch size one on an Apple M5 Max. Times are medians
of two runs per workflow. A case is one complete workflow example and can
contain many questions. Inference time includes prompt preparation and model
execution; it excludes loading and downloading eval assets.
These are the initial MLX-backend measurements, before the subsequent
candidate-projection and length-bucketing changes described below.

| Workflow | Public cases | Question matches | Reference agreement | Inference/case | Inference/question |
| --- | ---: | ---: | ---: | ---: | ---: |
| Security incidents | 5 | 30/48 | 62.5% | 2.73 s | 284.7 ms |
| Agent trace observability | 5 | 35/52 | 67.3% | 4.30 s | 413.7 ms |
| Invoice processing | 5 | 136/184 | 73.9% | 21.43 s | 582.4 ms |
| Customer service | 5 | 76/92 | 82.6% | 2.01 s | 109.3 ms |
| **Overall** | **20** | **277/376** | **73.7%** | **7.62 s** | **405.3 ms** |

These results measure 376 question instances inside the 20 public showcased
cases. A match means Qwen's top answer agrees with TypeSafe's separate published
reference consensus. TypeSafe does not publish the complete cases or
executable policy harness.

### Measured backend comparison

In the same process, using the same BF16 weights, prompts, batch size one, and
all 20 public cases, MLX was **1.38× faster overall (27% less inference time)**.
Each workflow ran twice per backend, in Transformers/MLX then MLX/Transformers
order. Both runtimes used the tokenization improvement described above.

| Backend | Reference agreement | Inference/case | Inference/question |
| --- | ---: | ---: | ---: |
| Transformers / MPS | 276/376 (73.4%) | 10.50 s | 558.7 ms |
| MLX | 277/376 (73.7%) | 7.62 s | 405.3 ms |

Individual decisions differ between runtimes; this is not bit-for-bit equality.
These are paired measurements from this run, not comparisons against historical
timings taken under different machine conditions. Batch one remained fastest in
the MLX invoice sweep of 1, 2, 4, and 8. The implementation uses
[MLX-LM](https://github.com/ml-explore/mlx-lm) and retains the original weights
without quantization. See the [raw measurements](benchmarks/m5-max-backends.json)
for workflow timings, reference counts, and exact dependency versions.

To reproduce a backend comparison on all 20 public cases, loading each model
once, downloading each asset once, and alternating execution order:

```bash
uv run --extra mlx python benchmarks/compare_backends.py \
  --repeats 2 --batch-size 1 --output /tmp/jev-backends.json
```

The JSON includes every timing and reference count, dependency versions, and
the sum of per-workflow median times. Add `--workflow invoice_processing --limit 1`
for a smaller experiment. The timer includes prompt preparation and waits for
probabilities to reach the CPU; model loading and warm-up are excluded.

### Check MLX optimizations independently

A subsequent batch-size-one validation covered all 20 cases / 376 questions:
the final MLX implementation matched the original-order/full-vocabulary MLX
control exactly on every answer and probability (277/376 reference matches).
This single-pass check was slower overall under variable background load; it
does not establish an additional latency improvement. Batch size one remains
the default. See the [validation measurements](benchmarks/m5-max-mlx-validation.json).
The invoice token analysis reduces padded suffix work at batch size eight from
212,890 to 99,687 tokens; this is a work reduction, not a measured speedup.

The MLX ablation benchmark compares original-order/full-vocabulary inference
against candidate-only projection, length/token-aware batching, or both. All
variants share one loaded model and identical prompts. It records reference
agreement, changed top answers, maximum probability drift against batch size
one, padding counts, source/asset hashes, and raw timings. Successive repetitions
reverse the execution order.

```bash
uv run --extra mlx python benchmarks/compare_mlx.py \
  --workflow invoice_processing --limit 1 \
  --batch-sizes 1,4 --variants original,both --repeats 2 \
  --output /tmp/jev-mlx-optimizations.json
```

Omit `--workflow` and `--limit` to cover all 20 public cases. Use
`--variants original,projection,batching,both` to isolate the changes.
`--asset-dir` accepts previously downloaded `<workflow>-cases.js` files for an
offline run. `--profile` separately instruments the first request in each
workflow, including compilation, prefix/suffix processing, attention, MLPs,
and output projection. These synchronized diagnostic timings overlap and
perturb execution; they are not used for the performance comparison.

## Sources

The key sources for the experiment are TypeSafe's [Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and [Jev documentation](https://docs.typesafe.ai/introduction)

## Test

```bash
uv run pytest
```

On Apple Silicon, include the MLX numerical-parity tests with:

```bash
uv run --extra mlx pytest
```
