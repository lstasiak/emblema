# ONNX export of a set-shaped encoder

Purpose: confirm that an encoder with one dynamic axis — the token count — a padding mask, a channel
embedding and a timeless-token flag exports to ONNX and runs in ONNX Runtime, before the architecture
is frozen. The decision and the reasoning live in ADR-0007; this note records what was observed,
where, and with which versions. The suite in `tests/evaluation/adapters/onnx` runs the same checks
in CI on Linux x86_64; the legs below are the ones CI does not cover.

Method on any machine, two commands:

```sh
uv sync --all-extras
uv run pytest tests/evaluation/adapters/onnx           # the assertions
uv run python -m tests.evaluation.adapters.onnx.report  # the numbers, as a section for this file
```

The MPS leg is a test (`test_inference_graph_from_mps.py`), skipped where the accelerator is
absent, so on Apple silicon the suite covers it without any extra step.

## 2026-09-10 — Windows x86_64, development machine

Versions: Python 3.14.5, torch 2.14.0+cpu, onnxruntime 1.29.0, onnx 1.22.0, onnxscript 0.7.2,
numpy 2.5.3. CPU: Intel family 6 model 140 (mobile).

| Check | Result |
|-------|--------|
| export with `torch.onnx.export(dynamo=True)`, opset 20 | pass, 2.0–8.0 s per model |
| declared graph inputs | `features [batch, n_tokens, 2]`, `channel_ids` / `timestamps` / `timeless` / `padding_mask` `[batch, n_tokens]`, output `pooled_embedding [batch, width]` |
| symbolic axes in the whole graph | exactly `batch` and `n_tokens`; the feature count stays a fixed integer |
| PyTorch vs ONNX Runtime, tokens 41–4104, batch 1–3, padded and unpadded, three attention implementations | pass, `max abs diff = 3e-06` (float32) |
| permutation invariance and padding invariance after export | pass, deviation of the order of 2e-07 |
| window of nothing but padding | `scaled_dot_product_attention`: finite. `nn.MultiheadAttention` and explicit `masked_fill(-inf)`: NaN, in PyTorch and in ONNX Runtime alike |
| opset 23 | exports and loads, then **fails on every execution**: fused `Attention` kernel rejects a `[batch, 1, 1, tokens]` mask (`inconsistent q_sequence_length`) |
| window longer than the declared `Dim` maximum (4104 against 4096) | runs, output correct — the bound constrains the exporter, not the session |
| output named `embedding` | graph rejected at load: duplicate definition, collides with the channel embedding |
| `torch.jit.script` on the same module | fails (`Tensor cannot be used as a tuple` — it rejects ordinary Python the exporter accepts); `torch.jit.trace` succeeds, output identical to eager |
| artefact and latency, one window of 512 tokens | 3.19M-parameter model: onnx 12822 KiB / 284.7 ms, torchscript 12570 KiB / 154.6 ms, eager 129.5 ms. Absolute values swing threefold between runs on this laptop; the ordering does not. See ADR-0007 on what this does not settle |

Result: **pass with two limitations**, both recorded in ADR-0007 — the opset is pinned to 20, and an
empty window is only safe under SDPA.

## 2026-09-11 — macOS arm64, M1 development machine

The architecture the containers and the native ML worker run on. `uv sync --all-extras` installed
onnx and onnxscript without incident, and the whole suite passed, `test_export_from_mps.py` among it
— so a model that ran on MPS, moved back to the CPU, exports and reproduces what the accelerator
computed. Report as printed:

Versions: Python 3.14.7, torch 2.14.0, onnxruntime 1.29.0, onnx 1.22.0, onnxscript 0.7.2,
numpy 2.5.3. CPU: arm. MPS available: True.

| Check | Result |
|-------|--------|
| attention `sdpa`: opset 20, tokens 41-4104, batch 1-3, padded and unpadded | pass, `max abs diff = 3.3e-06` |
| attention `sdpa`: window of nothing but padding | finite |
| attention `module`: opset 20, tokens 41-4104, batch 1-3, padded and unpadded | pass, `max abs diff = 1.9e-06` |
| attention `module`: window of nothing but padding | **NaN** |
| attention `manual`: opset 20, tokens 41-4104, batch 1-3, padded and unpadded | pass, `max abs diff = 3.6e-06` |
| attention `manual`: window of nothing but padding | **NaN** |
| opset 23 (fused `Attention` node present: True) | exports; loads, then execution: `Fail`: Non-zero status code returned while running Attention node |
| export straight from a model held on MPS | `TorchExportError`: fails at step 1/3, before ONNX is reached |

| Model | ONNX (opset 20) | TorchScript (traced) | eager |
|---|---|---|---|
| stand-in, 0.02M params: artefact | 228 KiB | 123 KiB | — |
| stand-in: latency, one window of 512 tokens | 2.3 ms | 1.4 ms | 1.4 ms |
| 3.19M params: artefact | 12825 KiB | 12570 KiB | — |
| 3.19M params: latency, one window of 512 tokens | 25.0 ms | 13.0 ms | 13.6 ms |

Export took 1.3 s and 1.8 s, against 9.8 s and 17.6 s on the x86 laptop.

Result: **pass**, and two things settled that the x86 leg could not.

- **A model held on MPS does not export.** `torch.export` refuses it at its first stage, so the
  weights are moved to the CPU first — which the suite now asserts is safe. This is a constraint on
  checkpointing and on the real export, not on the encoder.
- **The latency ordering is not an x86 artefact.** ONNX Runtime is the slowest of the three here
  too, by roughly the same ratio, on hardware ten times faster in absolute terms. ADR-0007's
  revision threshold stands as written.

Everything else reproduced identically, including the opset 23 failure and the NaN split between
attention implementations. The largest deviation seen anywhere is `3.6e-06`, against a tolerance of
`1e-5`.

## 2026-09-13 — Windows x86_64, development machine — the encoder itself

The stand-in was retired with the encoder it stood in for (ADR-0017); from here on the suite and
this script export `SetEncoder` with `MaskedMeanPooling` on top, through the same harness. Attention
is `scaled_dot_product_attention` only, so the three-way comparison above is history. The tier-M
encoder is built over the 121 channels of the measured corpora, the vocabulary one backbone over the
mix would carry. Versions: Python 3.14.5, torch 2.14.0+cpu, onnxruntime 1.29.0, onnx 1.22.0,
onnxscript 0.7.2, numpy 2.5.3.

| Check | Result |
|-------|--------|
| opset 20, tokens 41–4104, batch 1–3, padded and unpadded | pass, `max abs diff = 1.0e-06` |
| window of nothing but padding | finite |
| opset 23 (fused `Attention` node present) | exports, loads, fails on execution — as before |
| `torch.jit.trace` | reproduces eager exactly |

| Model | ONNX (opset 20) | TorchScript (traced) | eager |
|---|---|---|---|
| test encoder, 0.02M, 32 wide, 2 blocks | 279 KiB; 9.6 ms | 134 KiB; 2.6 ms | 2.8 ms |
| tier M, 4.78M, 256 wide, 6 blocks | 19 149 KiB; 74.5 ms | 18 801 KiB; 51.6 ms | 49.2 ms |

Latency is one window of 512 tokens. Exports took 2.2 s and 4.4 s.

**ONNX Runtime is the slowest of the three at both sizes, and the earlier reading that said otherwise
did not reproduce here or on the M1.** An earlier run under this heading had the tier-M encoder at 216 ms through ONNX
against 312 ms eager, and concluded that the ordering reverses at the published tier. Three runs
later the same day, after this script was corrected, put ONNX at 74.5 / 68.4 / 70.0 ms against eager at
49.2 / 56.1 / 50.3 ms — the ordering of the small model, held throughout. The whole machine was some
four times faster in the later runs, which is the drift this note warns about in its first section
showing up in a conclusion rather than in a number. Nothing about the encoder changed between them
that could account for it: the vocabulary grew by 57 rows and the time encoding lost four
frequencies, 12 k parameters in a model of 4.78 M.

What survives is the reading the stand-in already gave: on this laptop, in this configuration,
exporting buys portability rather than speed. ADR-0007's revision threshold — the target container,
tuned session options, the quantised variants — is where that question is settled, and a single run
of a thermally unsteady laptop is not evidence against it either way.

`torch.jit.script` is no longer measured. It refused the encoder — first on the architecture value
the module carries, which TorchScript cannot convert, then on unpacking the projected heads — and
making it pass would mean annotating production code for a compiler that PyTorch deprecates on
Python 3.14 and that no path of this project uses. The traced artefact, which the table above
compares, is unaffected.

## 2026-09-13 — macOS arm64 (M1 Pro) — the encoder itself

Versions: Python 3.14.7, torch 2.14.0, onnxruntime 1.29.0, onnx 1.22.0, onnxscript 0.7.2,
numpy 2.5.3. MPS available.

| Check | Result |
|-------|--------|
| opset 20, tokens 41–4104, batch 1–3, padded and unpadded | pass, `max abs diff = 1.1e-06` |
| window of nothing but padding | finite |
| opset 23 (fused `Attention` node present) | exports, loads, fails on execution — as before |
| export straight from a model held on MPS | `TorchExportError` at step 1 of 3 — as before |

| Model | ONNX (opset 20) | TorchScript (traced) | eager |
|---|---|---|---|
| test encoder, 0.02M, 32 wide, 2 blocks | 281 KiB; 2.3 ms | 135 KiB; 1.4 ms | 1.4 ms |
| tier M, 4.78M, 256 wide, 6 blocks | 19 153 KiB; 23.3 ms | 18 802 KiB; 10.9 ms | 12.0 ms |

Latency is one window of 512 tokens, on the CPU: the session runs on `CPUExecutionProvider` and the
eager comparison is made on the same device, so this is a like-for-like reading and says nothing
about MPS.

**The two constraints of ADR-0007 that belong to this machine still hold on the encoder.** A model
held on MPS does not export — the exporter fails at tracing, so an accelerator-trained backbone must
be moved to the CPU before it becomes a graph, which is what the export adapter will do. Opset 23
still emits a fused `Attention` node whose kernel then refuses the mask at inference; opset 20 stays
pinned.

**The latency ordering is settled on two machines.** ONNX Runtime is the slowest of the three here
too, about twice eager at tier M (23.3 ms against 12.0 ms) and 1.6× at the small one. Both
architectures now agree, so the reading is not a property of one thermally unsteady laptop:
exporting buys portability, and the speed question is the one ADR-0007 defers to the target
container with tuned session options.

## 2026-09-24 — Windows x86_64, development machine — the fitted candidate

The export adapter exists (`evaluation/adapters/onnx/`, ADR-0040), and from here the suite and
this report export the candidate a campaign keeps rather than the bare encoder: the encoder under
a transfer mode, the pooling and the task's head, with the answer multiplied back into the task's
unit. The graph has two outputs, `pooled_embedding` and `prediction`. The candidates are the small
test encoder under each of the four modes, over a vocabulary three channels wider than the
backbone's — so every pretrained mode carries grown rows — with every trainable part nudged off
its starting value, so a low-rank update contributes to the comparison rather than exporting as
zero. The suite moved to `tests/evaluation/adapters/onnx`; run it and the report as before:

```sh
uv run pytest tests/evaluation/adapters/onnx
uv run python -m tests.evaluation.adapters.onnx.report
```

Versions: Python 3.14.5, torch 2.14.0+cpu, onnxruntime 1.29.0, onnx 1.22.0, onnxscript 0.7.2,
numpy 2.5.3. CPU: Intel family 6 model 140 (mobile).

| Check | Result |
|-------|--------|
| `from_scratch`: opset 20, tokens 41–4104, batch 1–3, padded and unpadded | pass, `max abs diff = 9.5e-07` on the state, `1.5e-05` on the answer over a ceiling of 125; 296 KiB, exported in 11.0 s; empty window finite |
| `frozen_probe`: same | pass, `1.0e-06` on the state, `3.8e-06` on the answer; 304 KiB, 4.4 s; finite |
| `lora`: same | pass, `8.9e-07` on the state, `1.9e-05` on the answer; 392 KiB, 6.6 s; finite |
| `full_fine_tuning`: same | pass, `8.9e-07` on the state, `7.6e-06` on the answer; 308 KiB, 6.2 s; finite |
| a window over the task's grown channels, every mode | pass, within the same tolerances (suite) |
| opset 23 (fused `Attention` node present) | exports, loads, fails on execution — as before |
| `torch.jit.trace` | reproduces eager exactly, on the answer |

| Model | ONNX (opset 20) | TorchScript (traced) | eager |
|---|---|---|---|
| test candidate, 0.02M, full fine-tuning | 308 KiB; 14.0 ms | 145 KiB; 5.7 ms | 5.5 ms |
| candidate at tier M, 4.78M, 256 wide, 6 blocks | 19 175 KiB; 114.1 ms | 18 812 KiB; 113.7 ms | 71.5 ms |

Latency is one window of 512 tokens. The first export of a process took 11.0 s and the rest 4 to
8 s; the exporter warms up once.

**Everything the encoder leg established holds on the fitted candidate, under every mode.** The
low-rank wrappers, the grown channel table read through a select, and the head with its scale
trace and run; the two axes stay the only symbolic ones; the deviation on the state is where it
was (`1.0e-06` against a tolerance of `1e-5`), and on the answer it is that deviation through a
linear head and a ceiling of 125 (`1.9e-05` against `1.25e-3`).

**The check a runtime makes before it keeps a graph passes with room.** The torch runtime runs
the graph on the validation windows it has just scored and records the largest difference from
its own answers; the runtime test over the published test corpus sees a deviation below `1e-4`
in the task's unit against a refusal threshold of `1e-3` of the ceiling (ADR-0040).

**The latency ordering is unchanged.** ONNX Runtime is slowest at both sizes on this machine, as
in every run above; at tier M the traced and the ONNX graph are level and eager is fastest. The
reading is the one ADR-0007 recorded and defers to the target container with tuned session
options. The arm64 leg, and the MPS leg the suite carries (`test_inference_graph_from_mps.py`,
a candidate held on the accelerator exported once moved to the host), are recorded here when
they run.
