# ADR-0007: ONNX as the inference format — one dynamic token axis, a pinned opset, attention that survives an empty window

- Status: accepted
- Date: 2026-09-10; amended 2026-09-13 (stand-in replaced by the real encoder), 2026-09-24 (the
  exporter exists)
- Full text before condensation: commit `5f14447`

## Context

Serving runs an inference artifact on CPU. The encoder did not exist yet, so export was a
constraint on a design still being made — cheapest to test early. The input shape was fixed: a
**set of tokens** (channel id, timestamp, value, gap to the channel's previous token), timeless
tokens for static features, a padding mask. The channel count is absorbed into the token count, so
the only varying axis is the number of tokens. A second symbolic axis in an exported graph would
mean a channel-by-time grid had leaked back in. A stand-in with this input shape was exported,
executed and compared against eager PyTorch.

## Decision

- **ONNX via `torch.onnx.export(dynamo=True)`.** Five named inputs — `features [batch, tokens, 2]`,
  `channel_ids`, `timestamps`, `timeless`, `padding_mask` (`[batch, tokens]`) — and one output,
  `pooled_embedding [batch, width]`. Named inputs keep integer and boolean types.
- **Exactly two symbolic axes, `batch` and `n_tokens`.** The suite asserts these are the only ones:
  the grid-leak detector, on every commit. Agreement with eager: max |diff| 3e-06 (fp32) over 41 to
  4,104 tokens, batch 1–3, padded and unpadded; permutation and padding invariance survive export.
- **Opset pinned to 20.** From opset 23 attention lowers to the fused `Attention` operator, which
  ONNX Runtime 1.29's CPU kernel rejects with a broadcast mask `[batch, 1, 1, tokens]` — the graph
  exports and loads, then fails on every run. So the suite executes the graph. Revisit when the
  runtime accepts a broadcast mask.
- **Attention is `scaled_dot_product_attention`.** For a window with no observed token, exported
  SDPA returns zeros while `nn.MultiheadAttention` and a manual `masked_fill(-inf)` return NaN.
  Both defences apply: SDPA, and a window must hold at least one observed token (tokeniser and API
  enforce it) — a silent all-zeros embedding is not an answer.
- `Dim(max=...)` constrains the exporter only; rejecting an oversized window is the caller's job.
- The output is not called `embedding`: it collides with a module name and the runtime rejects the
  graph.

## Consequences

- The encoder inherits four measured constraints: SDPA, one dynamic axis, branch-free handling of
  the timeless flag (a Python `if` does not survive tracing), non-empty windows.
- `onnx` and `onnxscript` join the `ml` extra; the architecture rules forbid them in the core.
  Wheels exist for every target platform.
- **A model on MPS cannot be exported** (`torch.export` fails). Checkpoints and export load onto the
  CPU; this reproduces the accelerator to 1e-4 (fp32).
- 2026-09-13: the stand-in was retired; `tests/ml/onnx_export` exports the real encoder (ADR-0017).
  On it, at the published tier, ONNX Runtime beat eager on x86 (216 ms vs 312 ms, 4.76M
  parameters) while the small model kept the old ordering. The revision threshold below stands.
- 2026-09-24: the exporter exists. `InferenceGraph` (`evaluation/adapters/onnx/`) exports not the
  bare encoder but the candidate a campaign keeps — encoder, pooling and the task's head — with two
  outputs, `pooled_embedding` and `prediction` in the task's unit; a session asked for one computes
  only that. Every constraint above held under all four transfer modes, over a vocabulary the task
  grew and with low-rank updates in place: opset 20, `batch` and `n_tokens` the only symbolic axes,
  agreement with eager to 1.0e-06 on the state and 1.9e-05 on the answer over a ceiling of 125, a
  fully padded window finite, opset 23 still failing at execution. The suite moved with the adapter
  to `tests/evaluation/adapters/onnx`. The latency ordering is unchanged on x86 (tier M: 114 ms
  through ONNX Runtime, 71.5 ms eager); the revision threshold stands. How the graph reaches Serving
  and the check it passes before it is kept: ADR-0040.

## Alternatives considered

- **TorchScript.** `torch.jit.script` rejects ordinary Python that `torch.export` accepts;
  `torch.jit.trace` works and matches eager exactly. On the stand-in it was faster:

  | Model | Machine | ONNX (opset 20) | TorchScript (traced) | eager |
  |---|---|---|---|---|
  | 0.02M | x86, Windows | 29.6 ms | 10.3 ms | 17.1 ms |
  | 0.02M | M1, macOS | 2.3 ms | 1.4 ms | 1.4 ms |
  | 3.19M | x86, Windows | 284.7 ms | 154.6 ms | 129.5 ms |
  | 3.19M | M1, macOS | 25.0 ms | 13.0 ms | 13.6 ms |

  One window of 512 tokens; x86 absolutes vary threefold between runs, the ordering did not
  (`tests/evaluation/adapters/onnx/report.py`, `docs/verification/onnx-export.md`). ONNX is chosen for
  portability — an artifact free of the training stack and its Python version — and for quantised
  variants, not speed. **Revisit** on the target container with tuned session options and
  quantisation: if ONNX Runtime is still slower at equal accuracy, CPU serving switches to
  TorchScript.
- **Padding to a fixed token count**: unnecessary once the dynamic axis worked; it would put a
  scale parameter in the graph.
- **Serving the eager model**: ties inference to the training stack.

Measured with onnxruntime 1.29.0, onnx 1.22.0, onnxscript 0.7.2, torch 2.14.0, on Windows x86_64
and macOS arm64; every finding reproduced on both.
