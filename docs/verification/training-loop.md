# The training loop: a run that can be stopped

Purpose: show that a run interrupted inside an epoch and picked up from its checkpoint is the run
that was not interrupted, and record what that costs — the size of a checkpoint, the time an epoch
takes, and which precisions a machine will run. The decisions are in ADR-0021 (the runtime and the
resume), ADR-0022 (the experiment file) and ADR-0023 (tracking); this note records what was
observed, where, and with which versions.

The properties that must hold everywhere are tests, not numbers here: that a resumed run ends in
the weights the uninterrupted one ended in, exactly on the host and within a measured tolerance on
MPS (`tests/ml/test_resume_matches_uninterrupted.py`), that every runtime reports one outcome per
epoch and refuses a checkpoint from another run
(`tests/pretraining/ports/test_training_runtime_contract.py`), that a checkpoint round-trips on the
host, that a device is refused a precision it cannot run, and that a run whose loss or gradients
stop being finite stops before it steps or writes anything (`tests/pretraining/adapters/training`).
What this note adds is the arithmetic of a real machine: equality is claimed bit for bit on the
host, and an accelerator has to say for itself.

Method on any machine:

```sh
uv sync --all-extras
uv run pytest tests/pretraining tests/ml                                    # the assertions
uv run scripts/training_loop_report.py                                      # the numbers
uv run scripts/training_loop_report.py --device mps --repeats 6             # an accelerator
uv run scripts/training_loop_report.py --device mps --precision fp16 --repeats 20
```

The script publishes a control corpus through the real use cases and trains it three epochs, then
trains it again while stopping after the second epoch and resuming from the last checkpoint written
inside it — not on its boundary. It does both in turn, as many times as `--repeats` asks, and
compares every pair of runs: the largest difference between their weights, the largest between
their losses, and whether they ended in the same artifact. The store is content-addressed, so one
reference for two runs is the strongest form the claim has; where a device does not repeat its own
arithmetic, the claim is that pairs with a resume in them lie no further apart than pairs without.

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

### 2026-09-15 — macOS arm64 (M1 Pro), MPS

The first reading on the accelerator, with the script as it then was: one uninterrupted run against
one resumed run. On the host that is enough, because the arithmetic repeats; here it is not, and the
table cannot say whether its weight difference comes from the resume or from the device. The next
section answers that.

Command: `uv run scripts/training_loop_report.py --device mps`.

|  |  |
| --- | --- |
| Machine | macOS-26.6.2-arm64-arm-64bit-Mach-O, arm |
| Python | 3.14.7 |
| torch | 2.14.0 |
| Device | mps |
| Precisions this device runs | fp32, fp16 |
| Experiment | `experiments/control-a-s.toml` |
| Corpus | control-a cut to 8 units: 248 training and 87 validation windows |
| Run | 3 epochs, batch 32, fp32, a checkpoint every 3 steps |
| Checkpoint | 26.89 MB, read in 12 ms |
| Interruption | after the second epoch's checkpoint, which falls inside the epoch |
| Largest weight difference | 5.572e-05 |
| Same artifact | no |

#### Runs

| Run | Epochs | Last training | Last validation | Seconds | Weights |
| --- | --- | --- | --- | --- | --- |
| uninterrupted | 3 | 0.673601 | 0.439631 | 4.8 | f47583ca067d |
| resumed | 3 | 0.673601 | 0.439631 | 3.9 | e2dcfa8d02e3 |

#### Validation loss per epoch

| Epoch | Uninterrupted | Resumed | Seconds |
| --- | --- | --- | --- |
| 1 | 0.651497 | 0.651497 | 1.8 |
| 2 | 0.478727 | 0.478727 | 1.2 |
| 3 | 0.439631 | 0.439631 | 1.2 |

The same command run again minutes later printed the same losses, a largest weight difference of
3.915e-05 and uninterrupted weights `d6a869db2877` rather than `f47583ca067d`: two processes, one
uninterrupted run, two sets of weights.

#### Tracked through the report

The same evening the report ran through the training runtime and logged to the local stack's
MLflow server:

```sh
uv run scripts/masked_reconstruction_report.py --experiment control-a-s --epochs 2 --units 8 --track http://127.0.0.1:5000
```

A smoke run: two epochs scored on two validation units say nothing about the objective, and the
report says so itself — every kind `beaten`, a warning on the evidence for each, the assessment
`recalibrate`. Its figures are not kept. What it checks is the path from the use case to the
tracker and the store:

- The tracked run — experiment `control-a-s`, status `FINISHED` — holds the experiment's parameters
  with the epochs override recorded as 2, the corpus and the tier as tags, and the backbone's
  reference as a tag. Its metrics per epoch (training and validation loss, hidden share, seconds)
  equal the stored run's `epochs.csv` to every digit: validation 0.6514974132679904, then
  0.5121960828391635.
- No checkpoint tag: sixteen steps never reach the experiment's interval of 200, so no checkpoint
  was written, and none is claimed.
- Its first epoch is the first epoch above — validation 0.6515, training 1.1237 — through the use
  case and a store on disk rather than the runtime and a store in memory. The second epoch differs
  from the one above, as it should: two epochs decay the rate over one epoch, three over two.

### 2026-09-16 — macOS arm64 (M1 Pro), MPS, repeated

The script now repeats both runs and compares every pair. Two things it printed before changed with
that. The resumed column of the second epoch is the resumed run's own validation, after it re-entered
the epoch; until now it was the interrupted run's, which equals the uninterrupted run by
construction, so on the host leg above only the third row tested the resume. And a checkpoint's cost
is split into fetching it — the store hashes what it returns — and reading it back.

Commands: `uv run scripts/training_loop_report.py --device mps --repeats 6`, then the same with
`--precision fp16 --repeats 20`. The same checkout, with `--device cpu --repeats 2`, gave zero in
every pair and one artifact for all four runs, as on the host leg above.

|  |  |
| --- | --- |
| Machine | macOS-26.6.2-arm64-arm-64bit-Mach-O, arm |
| Python | 3.14.7 |
| torch | 2.14.0 |
| Device | mps |
| Precisions this device runs | fp32, fp16 |
| Experiment | `experiments/control-a-s.toml` |
| Corpus | control-a cut to 8 units: 248 training and 87 validation windows |
| Run | 3 epochs, batch 32, a checkpoint every 3 steps; fp32, then fp16 |
| Repeats | 6 uninterrupted and 6 resumed at fp32, 20 and 20 at fp16, in turn |
| Checkpoint | 26.89 MB, fetched in 11–12 ms, read in 13 ms |
| Interruption | after the second epoch, resumed from its last checkpoint, which falls inside it |
| Same artifact | no two runs, at either precision |

#### Largest difference per pair of runs, fp32

| Pair | Pairs | Weights, smallest | Weights, median | Weights, largest | Losses, largest |
| --- | --- | --- | --- | --- | --- |
| uninterrupted, uninterrupted | 15 | 2.921e-05 | 3.681e-05 | 4.745e-05 | 3.299e-08 |
| uninterrupted, resumed | 36 | 2.778e-05 | 3.468e-05 | 4.905e-05 | 2.474e-08 |
| resumed, resumed | 15 | 2.792e-05 | 3.462e-05 | 5.432e-05 | 2.071e-08 |

All twelve runs printed 0.673601 and 0.439631 as their last losses, and every epoch's validation
loss lay within 2.1e-08 across them. The losses compared are every epoch's validation loss and the
last epoch's training loss: the epoch a resumed run re-enters is trained only in part by it, so its
training loss covers the batches after the checkpoint and is not the epoch's.

#### Largest difference per pair of runs, fp16

| Pair | Pairs | Weights, smallest | Weights, median | Weights, largest | Losses, largest |
| --- | --- | --- | --- | --- | --- |
| uninterrupted, uninterrupted | 190 | 5.594e-03 | 7.364e-03 | 1.040e-02 | 1.299e-04 |
| uninterrupted, resumed | 400 | 5.419e-03 | 7.373e-03 | 1.156e-02 | 1.299e-04 |
| resumed, resumed | 190 | 5.450e-03 | 7.411e-03 | 1.198e-02 | 1.299e-04 |

The first two epochs' validation losses were equal in all forty runs (0.651659 and 0.478428). The
last one came out at one of two values, 0.439469 or 0.439599: seven of the twenty uninterrupted
runs and four of the twenty resumed runs took the second, so a resumed run that lands there has not
failed its resume. That 1.3e-04 is the whole of the loss difference in every kind of pair.

#### A test on the device

`tests/ml/test_resume_matches_uninterrupted.py` resumes on MPS as well, held to tolerances measured
on its own configuration (eight training windows in batches of two, three epochs, a checkpoint every
five steps). Faults were injected by altering a checkpoint as it is read, before it is put back:

| Resume | Largest loss difference | Largest weight difference |
| --- | --- | --- |
| correct: ten uninterrupted and ten resumed runs, each against one more | 1.4e-07 | 1.1e-04 |
| optimiser state not restored | 5.8e-02 | 4.9e-03 |
| masks drawn from the seed again | 1.3e-02 | 3.9e-03 |
| the last step before the checkpoint trained twice | 2.0e-02 | 5.3e-03 |

The test holds losses to 1e-5 and weights to 1e-3. With each of the three faults injected it failed;
unchanged, it passed eight runs out of eight. A fault it cannot see is one that changes nothing
here: without dropout the host's generator is not drawn from after initialisation, and leaving its
state out of the checkpoint moved no number beyond the correct runs' spread.

#### What the numbers say

- **On MPS a resumed run is the uninterrupted run to the tolerance the device keeps with itself**, at
  both precisions it runs: pairs with a resume lie as far apart as pairs without — medians 3.47e-05
  against 3.68e-05 at fp32, 7.37e-03 against 7.36e-03 at fp16. That is the claim this device can
  support; the bit-for-bit claim, and the test that makes it, stay on the host.
- **Half precision repeats itself two hundred times more loosely**, in weights, and it trains to where
  single precision does: its last validation loss lies within 2e-04 of fp32's 0.439631, and every
  loss it reported was finite. On this backend autocast computes the squared error in single
  precision, so the summed error ADR-0021 watches under half precision does not pass through a
  half-precision tensor; this run cannot say more about large batches than that.
- **Equality of references is a host property.** No two runs on MPS shared a reference, identically
  configured as they were, so runs on this device are compared by their weights and losses.
- **The spread belongs to the device, and its cause is not isolated here.** The masks are drawn on
  the host and this experiment's dropout is zero, so no generator on the device is drawn from. The
  report's full runs on this machine do not repeat either (`masked-reconstruction.md`).
- **A tracked resumed run is not the uninterrupted run's curve at the epoch it re-entered.** That
  epoch's training loss is the mean over the batches after the checkpoint; its validation loss, and
  every other epoch, are the run's.
- **A checkpoint is 26.89 MB on MPS**, against 26.88 MB for the same configuration on this machine's
  CPU; the difference is the accelerator's own generator state, which the host does not keep.
  Fetching it costs 11–12 ms, most of it the store's checksum, and reading it back 13 ms.
- **An epoch of 248 windows costs 1.2–1.5 s on MPS** after a first one of 1.7–1.8 s in each process,
  against 1.7–1.8 s on this machine's CPU and 8–11 s on the development machine's.
- **This device runs fp32 and fp16 and refuses bf16**, by the table in `TorchPrecision`.
