# ThinkFuse: Trajectory-Aware Test-Time Fusion for Small Reasoning Models

Reference implementation of ThinkFuse, a **training-free, test-time** fusion
framework for small reasoning models. A single small model often stalls at a
few uncertain reasoning steps, while a second model would have taken a better
turn at exactly those points. ThinkFuse monitors the primary model's
segment-level uncertainty as it reasons, and only when the current segment is
unusually uncertain relative to the model's own recent history does it invoke
an auxiliary model for an alternative continuation. The two candidate segments
are then scored by the primary model's perplexity, and the more
trajectory-compatible one is appended. Fusion is restricted to the explicit
thinking phase, so once the model commits to an answer it continues alone.

## How it works

* **Segment-level EWCA uncertainty.** Every `T` tokens, ThinkFuse reads the
  top-k token log-probabilities of the primary model, converts each position to
  a normalized Shannon entropy, and aggregates them with Exponentially Weighted
  Causal Aggregation (EWCA), which emphasizes later tokens and propagates the
  accumulated uncertainty of preceding tokens.
* **Inter-segment adaptive threshold with soft budget.** The segment
  uncertainty is compared against an adaptive threshold built from an
  exponential moving average of the trajectory's mean and deviation. A soft
  budget term raises the threshold quadratically as the cumulative fusion ratio
  grows, discouraging excessive intervention.
* **Primary-PPL trajectory-compatibility selection.** When the threshold is
  crossed, the auxiliary model proposes an alternative segment. After
  cross-tokenizer prefix alignment and reasoning-tag normalization, the primary
  model's perplexity selects the more compatible candidate. No reward model and
  no agreement term are used.
* **Thinking-phase gating.** The uncertainty, threshold, and fusion logic run
  only while `</think>` has not been emitted. After the thinking phase ends, the
  primary model decodes the final answer on its own.

## What is in this repository

```
thinkfuse/
  uncertainty.py    per-token entropy / confidence-gap and EWCA segment aggregation
  threshold.py      EMA adaptive threshold with soft fusion budget
  tags.py           reasoning-tag normalization and thinking/answer phase detection
  alignment.py      byte-safe cross-tokenizer shared-prefix segment alignment
  scoring.py        primary-PPL trajectory-compatibility selection
  fusion.py         the ThinkFuse engine implementing the full segment loop
  model_client.py   backend-agnostic model-client interface (plus an offline mock)
example.py          runnable demo using the mock client (no GPU, no network)
```

## Installation

```bash
git clone https://github.com/js-lee-AI/ThinkFuse.git && cd ThinkFuse
pip install -r requirements.txt
python example.py
```

The core depends only on NumPy. Real backends (vLLM, transformers, PyTorch) are
optional and only needed when you wire ThinkFuse to actual models.

## Usage

```python
from thinkfuse import MockModelClient, ThinkFuse

primary = MockModelClient("qwen3-4b", script="<think> ... </think> answer")
auxiliary = MockModelClient("ministral-3b-reasoning")

engine = ThinkFuse(primary, auxiliary, preview_tokens=8)
result = engine.run("Question: ...\nAnswer:")

print(result.text)
print(result.total_segments, result.fused_segments, result.fusion_ratio)
```

To run against real models, implement `ModelClient` for your backend. The
engine needs token-level generation with top-k log-probabilities and a
perplexity call:

```python
class MyClient(ModelClient):
    name = "qwen3-4b"
    def encode(self, text): ...
    def decode(self, ids, skip_special=True): ...
    def batch_decode(self, id_lists): ...
    def generate(self, context_ids, n_tokens, top_k): ...   # -> (new_ids, eos, token_logprobs)
    def perplexity(self, context_ids, segment_text): ...     # -> float
```

A typical implementation wraps a vLLM or transformers server: `generate`
requests `n_tokens` with `logprobs=top_k`, and `perplexity` scores a candidate
continuation against the cached context.

## Default hyperparameters

| Parameter | Symbol | Value |
|---|---|---|
| Segment length | T | 8 |
| Top-k log-probs | r | 5 |
| EWCA position weight | alpha | 0.1 |
| EWCA accumulation | beta | 0.5 |
| EMA decay | eta | 0.05 |
| Deviation margin | tau | 1.0 |
| Soft-budget coefficient | lambda | 50 |
| Initial mean | mu_0 | 0.5 |
| Initial deviation | sigma_0 | 0.1 |
| Uncertainty signal | - | entropy |

## Results

Accuracy (%) on a fixed random subsample of 200 examples per benchmark under a
16K-token generation budget. ThinkFuse uses Qwen3-4B as the primary model and
improves over the standalone primary, while general test-time fusion baselines
(Cool-Fusion, AdaFuse) degrade it.

| Method | MATH-500 | GSM8K | AIME24 | GPQA | NQ-Open |
|---|---|---|---|---|---|
| Qwen3-4B (standalone) | 89.0 | 92.0 | 53.3 | 54.0 | 29.5 |
| ThinkFuse (Qwen3-4B x Ministral-3B-R) | 90.0 | 96.9 | 73.3 | 64.6 | 39.5 |
| ThinkFuse (Qwen3-4B x EXAONE-Deep-2.4B) | 91.0 | 99.0 | 73.3 | 55.1 | 38.5 |

See the paper for the full set of model pairs, baselines, significance tests,
and ablations.

## Citation

A BibTeX entry will be added once the arXiv version is available.

## License

The code in this repository is released under the [MIT License](LICENSE). The
paper itself is distributed under CC BY 4.0.
