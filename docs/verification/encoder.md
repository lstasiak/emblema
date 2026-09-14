# What full self-attention costs at the windows the corpora produce

Purpose: put a number on the decision to attend every token to every other rather than approximate
it. The encoder's cost grows with the square of a window's token count, and the window length is the
knob that bounds it; this note records what a training step costs per window at the default window
of every measured corpus, for the two compute tiers the project trains at, beside the arithmetic the
numbers should agree with. The decision and its reasoning live in ADR-0017; this note records what
was observed, where, and with which versions.

What must hold everywhere is a test, not a number here: permutation, indifference to padding,
gradient flow, the exact parameter count (`tests/pretraining`), and the round trip through ONNX
Runtime (`tests/ml/onnx_export`, now on the encoder itself). The verdict the report prints — whether
the attention buffers of a window fit the published tier's device — is arithmetic, and is held to
itself in `tests/scripts/test_encoder_budget_report.py`.

Method on any machine, two commands:

```sh
uv sync --all-extras
uv run pytest tests/pretraining tests/ml tests/scripts   # the assertions
uv run scripts/encoder_budget_report.py                  # the numbers
```

Window lengths come from `scripts/corpus_budget.toml` — the default window of every corpus whose
windows the data spike counted, as its mean tokens per window — and tier shapes from the same
file, through `scripts/budget_file.py`. A step is a forward and backward pass over a batch of
random tokens carrying 8192 tokens in all, so a batch of long windows is small and a batch of short
ones large; the numbers are reported per window.

"Attention buffers" is an upper bound on the memory the scores and probabilities occupy at the peak
of a step where the kernel materialises them: two square matrices per head and layer. Only the layer
inside its softmax really holds two at once — the layers behind it hold their saved probabilities
alone — so a deep encoder peaks at roughly half the figure, and the bound is charged deliberately,
because it is a verdict about fitting a device. A fused kernel materialises nothing and pays neither.

"Step memory" is what the step itself takes: the weights and the batch are already on the device
when the count starts, so what is measured grows with the batch and the window and is worth dividing
by the batch. CUDA reports a peak directly. MPS has no peak counter and an allocator that keeps
every block it has taken, so the figure is the growth of the driver's allocation across the steps
after emptying the cache — the high-water mark of that span, which is the same thing once the span
begins empty. The CPU reports nothing torch can read back.

## Runs

### 2026-09-13 — Windows x86_64, development machine

Versions: Python 3.14.5, torch 2.14.0+cpu. CPU: Intel family 6 model 140 (mobile), no accelerator.
The times are the CPU's and say nothing about a GPU; the arithmetic and the verdict do not depend on
the machine. Vocabulary: 121 channels over the measured corpora, so the parameter counts below are
those of one backbone over the whole mix.

| Tier | Width | Heads | Layers | Feed-forward | Parameters |
| --- | --- | --- | --- | --- | --- |
| S | 192 | 3 | 4 | 768 | 1.81M |
| M | 256 | 4 | 6 | 1024 | 4.78M |

Attention buffers per window at the peak of a step, scores materialised:

| Tier | Window | Tokens | fp32 | fp16 |
| --- | --- | --- | --- | --- |
| S | physionet2012 h48 | 434 | 17 MiB | 9 MiB |
| S | skab w100s10 | 800 | 59 MiB | 29 MiB |
| S | esa_ad h1s1 | 839 | 64 MiB | 32 MiB |
| S | cmapss w50s5 | 1050 | 101 MiB | 50 MiB |
| S | smd w100s20 | 3800 | 1,322 MiB | 661 MiB |
| M | physionet2012 h48 | 434 | 34 MiB | 17 MiB |
| M | skab w100s10 | 800 | 117 MiB | 59 MiB |
| M | esa_ad h1s1 | 839 | 129 MiB | 64 MiB |
| M | cmapss w50s5 | 1050 | 202 MiB | 101 MiB |
| M | smd w100s20 | 3800 | 2,644 MiB | 1,322 MiB |

Training step, measured:

| Tier | Window | Tokens | Batch | s / window | Step memory / window |
| --- | --- | --- | --- | --- | --- |
| S | physionet2012 h48 | 434 | 18 | 42 ms | n/a |
| S | skab w100s10 | 800 | 10 | 116 ms | n/a |
| S | esa_ad h1s1 | 839 | 9 | 127 ms | n/a |
| S | cmapss w50s5 | 1050 | 7 | 224 ms | n/a |
| S | smd w100s20 | 3800 | 2 | 1,612 ms | n/a |
| M | physionet2012 h48 | 434 | 18 | 118 ms | n/a |
| M | skab w100s10 | 800 | 10 | 252 ms | n/a |
| M | esa_ad h1s1 | 839 | 9 | 284 ms | n/a |
| M | cmapss w50s5 | 1050 | 7 | 381 ms | n/a |
| M | smd w100s20 | 3800 | 2 | 2,687 ms | n/a |

Three runs of the script the same day agreed to within 2 % at tier M at every window length, and
to within 25 % at tier S, whose steps are small enough for fixed overheads to show. The ratios
below are therefore tier M's.

Verdict, for the published tier at a batch of 8 in fp16, against 11 GiB of a 15 GiB device (4 GiB
of headroom for weights, optimiser state, dense activations and the runtime; a 16 GB T4 reports
15360 MiB to the process):

- physionet2012 h48, 434 tokens: 0.1 GiB of attention buffers — fits.
- skab w100s10, 800 tokens: 0.5 GiB — fits.
- esa_ad h1s1, 839 tokens: 0.5 GiB — fits.
- cmapss w50s5, 1050 tokens: 0.8 GiB — fits.
- smd w100s20, 3800 tokens: 10.3 GiB — fits, with the scores materialised; the one window where
  the quadratic term is the bill. The margin is 0.67 GiB against a bound that charges twice what a
  six-layer encoder really peaks at, so the true margin is several gigabytes wider — but it rests
  on the 4 GiB assumed for everything else, and that assumption is what to re-check on the device.
  The other windows do not depend on it.

What the numbers say beyond the verdict:

- **The quadratic term is a minority below about a thousand tokens, but not nothing.** From 434 to
  1050 tokens — 2.4× the tokens — the time per window grows 3.2× at tier M. Linear would be 2.4×,
  because a step carries the same 8192 tokens whatever the window length, so a third of that growth
  is already attention.
- **At 3800 tokens the quadratic term dominates.** Going from 1050 to 3800 tokens (3.6×) costs 7.1×
  at tier M — past the linear factor, short of the 13× the scores alone would cost. That window
  belongs to SMD, and it is the first candidate for the window-length knob if the published device
  is tight; every other default window has headroom of an order of magnitude.
- **The exact parameter count of the published tier's reference shape is 4.78M over the mix** —
  under the 5–30M band aimed at, as the coarse 12·d²·L estimate (4.72M) already was. The
  shape of tier M is for the corpus-saturation measurement to settle; the count is now exact and
  available without torch.

### 2026-09-13 — macOS arm64 (M1 Pro), MPS

Versions: Python 3.14.7, torch 2.14.0, device `mps`. The shapes, the attention arithmetic and the
verdict are the same file's and do not repeat here; what this machine adds is a step on an
accelerator and the memory column, which only a device that reports its own allocation can fill.

| Tier | Window | Tokens | Batch | s / window | Step memory / window |
| --- | --- | --- | --- | --- | --- |
| S | physionet2012 h48 | 434 | 18 | 5 ms | 60 MiB |
| S | skab w100s10 | 800 | 10 | 10 ms | 106 MiB |
| S | esa_ad h1s1 | 839 | 9 | 11 ms | 118 MiB |
| S | cmapss w50s5 | 1050 | 7 | 15 ms | 152 MiB |
| S | smd w100s20 | 3800 | 2 | 123 ms | 1,556 MiB |
| M | physionet2012 h48 | 434 | 18 | 9 ms | 117 MiB |
| M | skab w100s10 | 800 | 10 | 21 ms | 216 MiB |
| M | esa_ad h1s1 | 839 | 9 | 23 ms | 233 MiB |
| M | cmapss w50s5 | 1050 | 7 | 31 ms | 299 MiB |
| M | smd w100s20 | 3800 | 2 | 245 ms | 2,581 MiB |

- **The memory figure tracks the window, which is the whole point of measuring it this way.** From
  434 to 3800 tokens it grows 26× at tier S and 22× at tier M. Read straight off the MPS driver it
  would not have: that counter is a process-wide high-water mark an allocator never gives back, so
  every row would have carried the rows before it and the first would have carried the Metal
  context on top.
- **Measured memory sits between the true peak and the bound, where it belongs.** At tier M and
  3800 tokens the bound charges 2,644 MiB of attention; the real peak is nearer 1,542 MiB, since
  only the layer inside its softmax holds two matrices and the five behind it hold one. Measured
  2,581 MiB leaves about 1.0 GiB for the dense activations and gradients — which is the shape of
  the decomposition at every other row too: subtract the true attention peak and what is left grows
  with the token count, as a per-window figure at a fixed 8192 tokens per step should.
- **The cost curve is the x86 machine's.** At tier M, 434 → 1050 tokens (2.4× the tokens) costs
  3.4× per window, and 1050 → 3800 (3.6×) costs 7.9× — against 3.2× and 7.1× on the CPU. The same
  reading: attention is a minority below a thousand tokens and the bill above three thousand.
- **Two scripts agree on the same step.** `loader_throughput_report.py` times tier M at 1050 tokens
  independently and reports 31 ms per window, to the millisecond what the table above says.
- The step is some 11× faster than the x86 CPU at tier M (245 ms against 2,687 ms for the longest
  window), in the band ADR-0008 assumes between this machine and no accelerator.

The verdict is unchanged and does not depend on the machine: the arithmetic is the same, and SMD's
3800-token window remains the only one where the quadratic term is the bill.
