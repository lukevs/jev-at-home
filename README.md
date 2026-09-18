# jev-at-home

A small experiment in Jev-shaped inference with an ordinary open-source causal
language model. It evaluates many independent enum or boolean questions in **one batched
transformer forward pass**, then restricts each row's next-token logits to
single-token labels (`A`, `B`, ...), applies softmax, and maps the distribution
back to the caller's enum.

This is not a reimplementation of TypeSafe's Jev model. TypeSafe publicly says
Jev uses a new architecture, parallel sampler, and Reinforcement Learning for
Calibrated Decisions, but has not published enough implementation detail to
reproduce those pieces. This repo isolates the simpler hypothesis: much of the
shape and latency advantage can be demonstrated by replacing autoregressive
structured generation with batched classification over a closed answer space.

## How it works

```text
shared state + N enum questions
              │
              ├─ one independent prompt per question
              │
              └─ padded batch ──> one LM forward pass
                                      │
                         next-token logits for each row
                                      │
                       select logits for A/B/C/... only
                                      │
                              softmax + enum mapping
```

The probabilities are normalized **only over the declared choices**. They are
useful scores, but they are not calibrated confidence estimates. Prompt wording,
label order, model choice, and temperature can all change them.

## Run it

The default model is the Apache-2.0
[`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).
The first run downloads its weights.

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

Input supports string enums and actual booleans. Each question has an explicit
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

## Design approach

The key sources for the experiment are TypeSafe's
[Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
[Jev documentation](https://docs.typesafe.ai/introduction), and the official
[*How to Design Programs* preface](https://felleisen.org/matthias/HtDP2e/part_preface.html).

## Test

```bash
uv run pytest
```
