# ADR-0007: ONNX as the inference format — one dynamic token axis, a pinned opset, attention that survives an empty window

- Status: accepted
- Date: 2026-09-10

## Context

Serving hands out an inference artefact and runs it on CPU. The encoder that will fill that artefact
does not exist yet: its architecture — attention mechanism, time encoding, width — is decided later.
Exporting is therefore a constraint on a design still being made, and the cost of learning about an
export limitation grows with every day the design settles: once the architecture is frozen, the
options narrow to rewriting the model or dropping ONNX.

The shape of the input is already fixed, though, and it is unusual enough to be worth checking
early. The representation is a **set of tokens**, each carrying a channel identifier, a timestamp
and a value, plus the gap since the previous observation in that channel. Channels enter through an
embedding of their identifier, so a window of three channels and a window of thirty are the same
kind of input; the channel count is absorbed into the token count and is **not an axis of the
tensor**. Static features of a window arrive as extra tokens marked timeless, which skip time
encoding rather than being given an invented timestamp. Padding is carried by a mask.

That leaves exactly one structural axis that varies: the number of tokens. A second symbolic axis
appearing in an exported graph would mean a channel-by-time grid had leaked back into the
representation — the assumption about regular sampling the project refuses to make.

So: a stand-in with the target input shape and none of the target behaviour, exported, executed, and
compared against the model it came from. A negative result would have been worth as much as this
positive one.

## Decision

**ONNX, exported with `torch.onnx.export(dynamo=True)`.** The exporter traces through
`torch.export`, and the token axis is declared with `torch.export.Dim`. The graph takes five named
tensors — `features [batch, tokens, 2]`, `channel_ids`, `timestamps`, `timeless`, `padding_mask`,
the last four `[batch, tokens]` — and returns `pooled_embedding [batch, width]`. Five named inputs
rather than one packed float tensor: the integer identifier and the two boolean masks keep their
types, the graph documents itself, and the API that will deserialise a request has one obvious
target shape per field.

Two symbolic axes are declared, `batch` and `n_tokens`. Batch size is a necessity of inference, not
a property of the representation; the representation contributes the one axis it is entitled to. The
feature count stays a fixed integer in the graph, and the test suite asserts that these two names
are the *only* symbolic axes — that assertion is the grid-leak detector, running on every commit.

Measured against the eager model: agreement to `max |diff| = 3e-06` in float32 over token counts 41
to 4104, batches of 1 to 3, padded and unpadded, for all three attention implementations tried.
Permutation invariance and padding invariance survive the export, which matters because they are the
claims the encoder exists to make.

**Opset pinned to 20.** From opset 23 the exporter lowers scaled dot-product attention to the fused
`Attention` operator. The graph exports and loads, then fails on every execution: the CPU kernel in
ONNX Runtime 1.29 requires the mask's query axis to equal the query length and rejects a mask
broadcast as `[batch, 1, 1, tokens]`. Opset 20 keeps attention as explicit `MatMul` / `Softmax` /
`Where` and runs. Note what this implies about testing: a graph that exports and loads is not a
graph that runs, so the suite executes it.

Revisit when the runtime accepts a broadcast mask on `Attention`, or if the fused kernel becomes
worth expanding the mask to `[batch, 1, tokens, tokens]` — which costs memory quadratic in the
window and would have to earn it.

**Attention: `scaled_dot_product_attention`.** The three implementations tried differ only in how
the mask reaches the softmax, and they agree on everything except the one case that matters at the
edge. For a window with no observed token — attention over nothing, a softmax whose every input is
masked — the exported SDPA graph keeps a guard (`IsNaN` / `Where` nodes appear in it) and returns
zeros, while `nn.MultiheadAttention` and an explicit `masked_fill` with negative infinity produce
NaN, identically in PyTorch and in ONNX Runtime. The choice of attention module is therefore a
choice about boundary behaviour, not only about speed.

Both defences apply: SDPA is used, **and** a window is required to carry at least one observed
token — the tokeniser upholds it, and the API rejects a request that filtering has emptied. Relying
on the guard alone would leave a silent all-zeros embedding as the answer to a meaningless question.

**The declared bound is not a runtime guard.** `Dim(max=...)` constrains the exporter. A session
happily runs a window of 4104 tokens against a graph exported with a maximum of 4096, and returns
correct numbers. Rejecting an oversized window is the caller's job.

**Naming.** The output cannot be called `embedding`: the name collides with a value the channel
embedding contributes and the runtime rejects the graph as having a duplicate definition. Output
names must not collide with module names — a five-second failure that would have been a puzzling one
in stage 6.

## Consequences

- The suite in `tests/ml/onnx_export` runs in CI from this ticket on, against a stand-in, and is
  replaced by the real export test when the encoder exists. Three exports, about 14 seconds; the
  model suite as a whole stays well inside the budget for the unit and domain tests.
- The encoder inherits four constraints, each of them measured rather than assumed: SDPA, one
  dynamic axis, branch-free handling of the timeless flag (a Python `if` does not survive tracing),
  and the non-empty-window invariant.
- `onnx` and `onnxscript` join the `ml` extra — the dynamo exporter needs both. The core stays
  framework-free; `onnxscript` is added to the packages the architecture rules forbid there.
- Wheels exist for every target: `onnx` ships a `cp312-abi3` wheel that covers 3.12 through 3.14 on
  manylinux aarch64 and x86_64 and on macOS arm64, `onnxscript` is pure Python.
- **A model held on MPS cannot be exported.** `torch.export` fails on Apple silicon before it reaches
  ONNX at all, so a backbone trained on the development machine is moved to the CPU first. Moving it
  reproduces what the accelerator computed, to `1e-4` in float32 — the two differ only by the order
  they accumulate in. Checkpointing and export therefore load onto the CPU deliberately, rather than
  onto whichever device happens to be available.
- The stand-in is test code and stays test code. No exporter or inference adapter is written yet:
  its shape depends on the model, and the model does not exist.
- *2026-09-13:* the stand-in was retired with the encoder it stood in for (ADR-0017). The suite in
  `tests/ml/onnx_export` exports the real encoder from then on, with `scaled_dot_product_attention`
  as its only attention; the two losing variants left with it. The measurements above were made on
  the stand-in and stand as recorded. The first run on the encoder itself (the verification note,
  same date) reversed the latency ordering at the published tier on the x86 machine — ONNX Runtime
  216 ms against 312 ms eager for the 4.76M model — while the small model kept the old ordering.
  The revision threshold above is unchanged; it now has evidence on both sides to weigh.

## Alternatives considered

**TorchScript**, measured on the same stand-in rather than argued about. `torch.jit.script` refuses
it outright — it rejects ordinary Python that `torch.export` accepts, so choosing it would constrain
how the model may be written. `torch.jit.trace` succeeds and reproduces the eager output exactly.

Artefact size, which does not depend on the machine: 227 KiB against 122 KiB for the stand-in,
12.5 MiB against 12.3 MiB for a 3.19M-parameter model — ONNX carries the larger file, by a margin
that shrinks to nothing as the weights come to dominate.

Latency of one window of 512 tokens, on both machines this project develops on:

| Model | Machine | ONNX (opset 20) | TorchScript (traced) | eager |
|---|---|---|---|---|
| stand-in, 0.02M | x86 mobile, Windows | 29.6 ms | 10.3 ms | 17.1 ms |
| stand-in, 0.02M | M1, macOS | 2.3 ms | 1.4 ms | 1.4 ms |
| 3.19M, 256 wide, 6 blocks | x86 mobile, Windows | 284.7 ms | 154.6 ms | 129.5 ms |
| 3.19M, 256 wide, 6 blocks | M1, macOS | 25.0 ms | 13.0 ms | 13.6 ms |

Reproduce with `uv run scripts/onnx_export_report.py`, which prints this table for the machine it
runs on. Absolute latencies on the x86 laptop move by a factor of three between runs — an earlier
run of the same comparison gave 92.8 / 51.3 / 43.3 ms for the larger model — so only the ordering is
worth reading. The ordering held in every run, on both architectures.

**ONNX Runtime is the slowest of the three, at both sizes and on both machines.** Reporting it the
other way round would be easy and wrong, and the M1 result removes the obvious excuse: this is not
an artefact of one throttled x86 laptop. What the numbers still do not settle: the session options
are untouched defaults; the dynamic token axis denies the runtime the shape-specialised fusion it is
usually credited for; quantisation — a main reason to want ONNX — is not applied here; and neither
machine is the Linux container that serves. The decision rests on portability, on an artefact that
does not carry the training stack or its Python version, and on the quantised and reduced variants
the efficiency work needs — not on speed, which it currently loses.

Revision threshold, to be checked when the real backbone is exported: measured on the target
container, with tuned session options and the quantised variants in scope. If ONNX Runtime is still
slower at equal accuracy, the CPU serving path switches to TorchScript and ONNX keeps only its
portability role. The alternative stays viable — tracing works today, and the table above is what
it would be compared against.

**Padding every window to a fixed maximum token count**, with the mask carrying the difference. The
fallback if a dynamic axis had proved unexportable. It is not needed: the dynamic axis works, and
the fixed one would have put a scale parameter in the graph.

**Serving the eager model.** Ties inference to the training stack, its Python version and its
dependency tree, and gives up the quantised variants.

## Versions

Measured on two machines, with onnxruntime 1.29.0, onnx 1.22.0, onnxscript 0.7.2 and numpy 2.5.3 on
both: Windows x86_64 (Intel family 6 model 140) with Python 3.14.5 and torch 2.14.0+cpu, and macOS
arm64 (M1) with Python 3.14.7 and torch 2.14.0 — the architecture the containers and the ML worker
actually run on. Every finding above reproduced on both, including the opset 23 failure; the full
runs are in `docs/verification/onnx-export.md`.
