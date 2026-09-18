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

The Qwen results below were measured with `Qwen/Qwen3-4B-Instruct-2507`, MPS,
and the default batch size of eight on an Apple M5 Max. A case is one complete
workflow example and can contain many questions. Inference time includes model
calls only; it excludes model loading and downloading the eval assets.

| Workflow | Public cases | Question matches | Reference agreement | Inference/case |
| --- | ---: | ---: | ---: | ---: |
| Security incidents | 5 | 30/48 | 62.5% | 3.17 s |
| Agent trace observability | 5 | 34/52 | 65.4% | 4.92 s |
| Invoice processing | 5 | 137/184 | 74.5% | 38.95 s |
| Customer service | 5 | 75/92 | 81.5% | 1.85 s |
| **Overall** | **20** | **276/376** | **73.4%** | **12.22 s** |

These results measure 376 question instances inside the 20 public showcased
cases. A match means Qwen's top answer agrees with TypeSafe's separate published
reference consensus. TypeSafe does not publish the complete cases or
executable policy harness.

## Sources

The key sources for the experiment are TypeSafe's
[Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and
[Jev documentation](https://docs.typesafe.ai/introduction)

## Test

```bash
uv run pytest
```
