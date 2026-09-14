# Batching windows for training

Purpose: confirm that putting a batch in front of the model costs a negligible fraction of a
training step, and record what windows cost to hold in memory — the number that decides the format
of the published corpus. The decisions live in ADR-0013; this note records what was
observed, where, and with which versions.

The properties that must hold everywhere are tests, not numbers here: the order a seed and an epoch
give, the shapes a batch of unequal windows has, and the padding that follows each of them
(`tests/shared/adapters/loaders`, `tests/shared/adapters/tensors`, `tests/shared/kernel`). The ratio
between a step and the batch it consumes is also a test, skipped where the accelerator is absent
(`tests/ml/test_loader_keeps_up_on_mps.py`), so on Apple silicon it needs no extra step; the corpus
that test feeds is built by the report script and checked on every machine
(`tests/scripts/test_loader_throughput_report.py`).

Method on any machine, two commands:

```sh
uv sync --all-extras
uv run pytest tests/shared tests/scripts tests/ml   # the assertions
uv run scripts/loader_throughput_report.py          # the numbers
```

The script measures the real windows of C-MAPSS FD001 where the raw files are present, and windows
of the same shape generated in memory where they are not; which one it used is in its output.
"Feeding the model" is collating a batch plus moving it to the device, because a batch built in
host memory has to cross before the step can start.

## Runs

### 2026-09-12 — Windows x86_64, development machine

Versions: Python 3.14.5, torch 2.14.0+cpu, numpy 2.5.3. CPU: Intel family 6 model 140 (mobile), no
accelerator, so the step times below are the CPU's and the ratio is not the one the ticket is about
— the MPS leg is.

|  |  |
| --- | --- |
| Machine | Windows-11-10.0.26200-SP0, Intel64 Family 6 Model 140 Stepping 1, GenuineIntel |
| Python | 3.14.5 |
| torch | 2.14.0+cpu |
| numpy | 2.5.3 |
| Device | cpu |
| Batch | 8 |

#### Windows

| Source | Windows | Tokens per window | Held in memory per window |
| --- | --- | --- | --- |
| C-MAPSS FD001, default window | 512 | 1050 | 93.4 KiB |

#### Feeding the model

| Workers | Collating, s / batch | To cpu, s / batch | Windows / s | Tokens / s |
| --- | --- | --- | --- | --- |
| 0 | 2.20 ms | 0.02 ms | 3,604 | 3,784,721 |
| 2 | 2.93 ms | 0.02 ms | 2,716 | 2,852,162 |

#### Ordering an epoch

| Windows in the corpus | s / epoch |
| --- | --- |
| 512 | 2.6 ms |
| 25,395 | 93.3 ms |
| 250,000 | 1079.1 ms |

#### Training step, stand-in encoder

| Tier | Width | Layers | s / step | Windows / s |
| --- | --- | --- | --- | --- |
| S | 192 | 4 | 2552 ms | 3 |
| M | 256 | 6 | 5634 ms | 1 |

#### Verdict

- Tier S, 0 worker(s): a step costs 1149.8x feeding it — off the critical path at a margin of 2x.
- Tier S, 2 worker(s): a step costs 866.5x feeding it — off the critical path at a margin of 2x.
- Tier M, 0 worker(s): a step costs 2538.6x feeding it — off the critical path at a margin of 2x.
- Tier M, 2 worker(s): a step costs 1913.1x feeding it — off the critical path at a margin of 2x.

Reading:

- **Feeding is free relative to a step.** Three orders of magnitude on this CPU. The absolute
  per-batch figure swings by a factor of three between runs on this laptop, as the export
  measurements did; the order of magnitude does not.
- **Worker processes cost a little here and buy nothing.** A third more per batch than collating in
  the training process, because Windows starts them by spawning and each one is handed the whole
  list of windows once. That handover is paid at startup rather than per epoch — the workers are
  persistent — which is the difference between this row and the earlier measurement that had them
  restarting each epoch and looked catastrophic. macOS forks and pays less again; either way, the
  row that matters is zero workers, which is what a run uses until the corpus is a memory map.
- **A window costs 93.4 KiB held as Python objects.** Full C-MAPSS is 25 395 windows of this shape:
  2.3 GB before a single tensor exists. This is the number that decided the published format.
- **Ordering an epoch by digest is cheap at this project's scale** and stops being cheap somewhere
  past a million windows: 93 ms at 25 000, 1.1 s at 250 000.

### 2026-09-12 — macOS arm64 (M1), MPS

The leg the ticket's definition of done is about. Versions: Python 3.14.7, torch 2.14.0, numpy
2.5.3.

|  |  |
| --- | --- |
| Machine | macOS-26.6.2-arm64-arm-64bit-Mach-O, arm |
| Python | 3.14.7 |
| torch | 2.14.0 |
| numpy | 2.5.3 |
| Device | mps |
| Batch | 8 |

#### Windows

| Source | Windows | Tokens per window | Held in memory per window |
| --- | --- | --- | --- |
| C-MAPSS FD001, default window | 512 | 1050 | 93.4 KiB |

#### Feeding the model

| Workers | Collating, s / batch | To mps, s / batch | Windows / s | Tokens / s |
| --- | --- | --- | --- | --- |
| 0 | 0.74 ms | 1.48 ms | 3,619 | 3,799,958 |
| 2 | 0.67 ms | 1.48 ms | 3,727 | 3,912,856 |

#### Ordering an epoch

| Windows in the corpus | s / epoch |
| --- | --- |
| 512 | 0.5 ms |
| 25,395 | 28.5 ms |
| 250,000 | 301.3 ms |

#### Training step, stand-in encoder

| Tier | Width | Layers | s / step | Windows / s |
| --- | --- | --- | --- | --- |
| S | 192 | 4 | 112 ms | 71 |
| M | 256 | 6 | 221 ms | 36 |

#### Verdict

- Tier S, 0 worker(s): a step costs 50.7x feeding it — off the critical path at a margin of 2x.
- Tier S, 2 worker(s): a step costs 52.2x feeding it — off the critical path at a margin of 2x.
- Tier M, 0 worker(s): a step costs 100.0x feeding it — off the critical path at a margin of 2x.
- Tier M, 2 worker(s): a step costs 102.9x feeding it — off the critical path at a margin of 2x.

Reading — **the ticket's definition of done is met**: at tier M a step costs a hundred times what
feeding it does, and at tier S, the smaller model this machine will actually pretrain, fifty times.

- **Collating is three times faster here than on the x86 laptop** (0.74 ms against 2.20 ms), and
  worker processes neither help nor hurt: macOS forks, so the handover costs almost nothing, and
  there is nothing for the workers to overlap with anyway.
- **Moving the batch to the accelerator costs twice what building it does** — 1.48 ms against
  0.74 ms — and is the larger half of feeding on this machine. Unified memory makes the crossing
  cheap, not free; leaving it out of the comparison would have flattered the loader by a factor of
  three.
- **The step is 25 to 50 times faster than on the x86 CPU** (221 ms against 5634 ms at tier M),
  which is the ratio the compute budget of ADR-0008 assumes between this machine and no
  accelerator.
- **Ordering an epoch is three times cheaper here**: 28.5 ms for a full C-MAPSS, 301 ms for a
  corpus ten times larger.

### 2026-09-13 — macOS arm64 (M1), MPS — the encoder itself

The stand-in was retired once the encoder existed; the script now times `SetEncoder` at each tier's shape, over
the 121 channels of the measured corpora. Only the step and the verdict change — collation,
ordering and the cost of holding a window are the loader's and were not touched.

#### Training step, real encoder

| Tier | Width | Layers | s / step | Windows / s |
| --- | --- | --- | --- | --- |
| S | 192 | 4 | 120 ms | 67 |
| M | 256 | 6 | 247 ms | 32 |

#### Verdict

- Tier S, 0 worker(s): a step costs 46.7x feeding it — off the critical path at a margin of 2x.
- Tier S, 2 worker(s): a step costs 48.3x feeding it — off the critical path at a margin of 2x.
- Tier M, 0 worker(s): a step costs 96.2x feeding it — off the critical path at a margin of 2x.
- Tier M, 2 worker(s): a step costs 99.4x feeding it — off the critical path at a margin of 2x.

**The stand-in was well dimensioned.** The real encoder costs 7 % more per step at tier S and 12 %
more at tier M (120 ms against 112 ms, 247 ms against 221 ms), so every reading of the section above
survives the substitution and the ticket's definition of done is met on the model the project will
actually train: at tier M a step costs a hundred times what feeding it does, at tier S fifty.

The step at tier M matches `encoder_budget_report.py` to the millisecond: 247 ms for a batch of 8
at 1050 tokens here is 31 ms per window there. Two scripts, written apart, timing the same thing.

