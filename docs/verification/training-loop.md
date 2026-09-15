# The training loop: a run that can be stopped

Purpose: show that a run interrupted inside an epoch and picked up from its checkpoint is the run
that was not interrupted, and record what that costs — the size of a checkpoint, the time an epoch
takes, and which precisions a machine will run. The decisions are in ADR-0021 (the runtime and the
resume), ADR-0022 (the experiment file) and ADR-0023 (tracking); this note records what was
observed, where, and with which versions.

The properties that must hold everywhere are tests, not numbers here: that a resumed run ends in
the weights the uninterrupted one ended in (`tests/ml/test_resume_matches_uninterrupted.py`), that
every runtime reports one outcome per epoch and refuses a checkpoint from another run
(`tests/pretraining/ports/test_training_runtime_contract.py`), that a checkpoint round-trips on the
host and that a device is refused a precision it cannot run
(`tests/pretraining/adapters/training`). What this note adds is the arithmetic of a real machine:
equality is claimed bit for bit on the host, and an accelerator has to say for itself.

Method on any machine, two commands:

```sh
uv sync --all-extras
uv run pytest tests/pretraining tests/ml   # the assertions
uv run scripts/training_loop_report.py     # the numbers
```

The script publishes a control corpus through the real use cases, trains it three epochs, then
trains it again while stopping after the second epoch's checkpoint — which falls inside the epoch,
not on its boundary — and resumes from that checkpoint. It compares the losses of both runs epoch
by epoch, the largest difference between their weights, and whether the two ended in the same
artifact: the store is content-addressed, so one reference for two runs is the strongest form the
claim has.

## Runs

### 2026-09-15 — Windows x86_64, development machine

The host leg, where equality is exact: the same arithmetic in the same order, so the resumed run's
weights are the uninterrupted run's byte for byte and the content-addressed store gives both the
same reference. The accelerator leg is what the M1 has to answer, and it is where half precision
can be measured at all — this device refuses it (`fp32, bf16` below), which is the policy working
rather than a limitation of the run.

Command: `uv run scripts/training_loop_report.py`.

|  |  |
| --- | --- |
| Machine | Windows-11-10.0.26200-SP0, Intel64 Family 6 Model 140 Stepping 1, GenuineIntel |
| Python | 3.14.5 |
| torch | 2.14.0+cpu |
| Device | cpu |
| Precisions this device runs | fp32, bf16 |
| Experiment | `experiments/control-a-s.toml` |
| Corpus | control-a cut to 8 units: 248 training and 87 validation windows |
| Run | 3 epochs, batch 32, fp32, a checkpoint every 3 steps |
| Checkpoint | 26.88 MB, read in 29 ms |
| Interruption | after the second epoch's checkpoint, which falls inside the epoch |
| Largest weight difference | 0.000e+00 |
| Same artifact | yes |

#### Runs

| Run | Epochs | Last training | Last validation | Wall seconds | Weights |
| --- | --- | --- | --- | --- | --- |
| uninterrupted | 3 | 0.673601 | 0.439631 | 33.0 | e1a7c0eb0526 |
| resumed | 3 | 0.673601 | 0.439631 | 35.8 | e1a7c0eb0526 |

The wall seconds are not a comparison of cost: the resumed row covers the two epochs that were
thrown away, the re-entered epoch and the one after it, while re-entering an epoch skips the
micro-batches already consumed rather than training them again. What is comparable is everything
else in the row. They are also the softest number here — an earlier run of this same script, taken
while the test suite had the CPU, read 79 s and 23 s an epoch, a bit over twice these. The losses
and the weights did not move between the two runs, which is the point: what this note claims is
exact, and what it merely observes is the machine's mood.

#### Validation loss per epoch

| Epoch | Uninterrupted | Resumed | Seconds |
| --- | --- | --- | --- |
| 1 | 0.651497 | 0.651497 | 8.4 |
| 2 | 0.478727 | 0.478727 | 10.6 |
| 3 | 0.439631 | 0.439631 | 10.3 |

#### Drawing from the store

The other half of the ticket, checked the same day on the same machine: a two-epoch run of
`control-a-s` was trained, stored and drawn, and then drawn again from the stored directory alone
with `--figures-only`. All three figures came out byte for byte identical, which is what says the
stored run holds everything a figure needs — the curve, the baseline levels and the drawn windows
token by token — and that a change of style costs a redraw rather than a run.

#### What the numbers say

- **A resumed run is the run it resumed.** Equal to six decimal places on every epoch's loss, zero
  difference on every weight, one artifact reference for both runs.
- **A checkpoint of this run is 26.9 MB**, which is everything three times over: the tier-S encoder
  (1.79 M parameters), the decoder that trains it, and Adam's two moments of both, in single
  precision. The published tier's encoder is 2.67 times larger, so the interval a run checkpoints
  at is a real choice about the remote bucket's 10 GB, not a free one.
- **An epoch of 248 windows costs 8–11 s on this idle CPU**; the mid-epoch checkpoint and the
  resume cost nothing measurable beside it.
- **This device runs fp32 and bf16 and refuses fp16**, by the table in `TorchPrecision`. Half
  precision is a CUDA and MPS matter and is measured where it exists.

Open, for the M1 leg: whether a resumed run on MPS matches bit for bit or only to a tolerance —
the accelerator reorders reductions, so the honest expectation there is closeness, not equality,
and the number that closeness comes out at is what the note is missing.
