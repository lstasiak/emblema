# ADR-0029: Pretraining over a mixture of corpora — one vocabulary chained through the publications, optimiser steps of one corpus each weighed by their count, the best epoch kept by the mean relative validation, and a run that says where it can be picked up from

- Status: accepted (2026-09-19, on the run below)
- Date: 2026-09-18
- Full text before condensation: commit `5f14447`

## Context

The reference shape is admitted on the value of the mix alone (ADR-0008). The first full
pretraining runs over four corpora:

| Corpus | Window | Training windows | Tokens / window | Share of tokens | Share of windows |
| --- | --- | --- | --- | --- | --- |
| C-MAPSS | 50 / 5 cycles | 20,237 | 1,050 | 11 % | 14 % |
| SKAB | 100 / 10 s | 3,896 | 800 | 2 % | 3 % |
| SMD | 50 / 10 min | 58,411 | 1,900 | 60 % | 41 % |
| ESA-AD | 1 / 1 h | 61,202 | 839 mean | 27 % | 43 % |

C-MAPSS is learnt from a quarter of its engines; the others are data-limited; SMD and SKAB are
best after one to three passes; ESA-AD's held-out months err 5× a classical corpus's under the
trivial predictor (ADR-0028); window lengths differ 2.4×. An epoch prices at ~40 minutes on a T4;
sessions end at 12 hours, with no tracking server.

## Decision

- **`TrainingMixture`** is a domain value: corpora in chain order; a one-corpus run is a mixture of
  one and digests exactly as before. The backbone's input becomes a tuple; order and result list
  manifests and shapes in that order. The reader port reads one manifest; use cases compose reads.
- **The vocabulary is chained through publications**: each corpus is published continuing the
  previous manifest's vocabulary (C-MAPSS 1–21, SKAB 22–29, SMD 30–67, ESA-AD 68–84). The invariant:
  each corpus's channel names are a prefix of the next's; otherwise refused. Leaving a corpus out
  is still a chain.
- **Every optimiser step uses micro-batches of one corpus**, steps interleaved by seed, an epoch =
  every window once. Micro-batch 16 with 2 accumulated (effective 32). Remainders close the epoch,
  so at most two of ~4,500 steps mix corpora. **Under Adam a corpus weighs its share of steps** —
  ESA-AD 43 %, SMD 41 %, C-MAPSS 14 %, SKAB 3 % — stated, not a parameter yet.
- **Fixed epochs, best epoch kept by mean relative validation** (ADR-0028's default). Best weights
  are written when an epoch improves. No early stopping: a patience rule would end each run at a
  different learning rate. Eight epochs.
- **Huber loss, knee at one deviation, dropout zero** — as for the single-corpus backbones, so the
  comparison does not read regularisation into every cell.
- **Each manifest holds its corpus's window**; the experiment lists corpora in chain order.
- **The run logs where it stands** via standard-library logging: progress every N steps (loss,
  seconds per step, time left), every checkpoint reference, per-corpus loss per epoch. The platform
  keeps the log, so a dropped run's checkpoint is read off it. Checkpoint every 2,000 steps
 .

## Consequences

- The table of backbone inputs gains a position; handoff documents move a version; existing
  backbones decode unchanged.
- Every run keeps its best epoch from here on, reports included.
- Mixed-run publications continue the C-MAPSS manifest, so single and mixed backbones share ids.

## Alternatives considered

- *Temperature sampling* (Arivazhagan et al., 2019; Conneau et al., 2020): loses the epoch; C-MAPSS
  gains nothing from more passes.
- *A weight per corpus now*: a dial before its first measurement.
- *Patience-based stopping*: runs end under different schedules.
- *Dropout 0.1*: a convention without a measurement, separating backbones.

## Revisit when

- A corpus rises while the mean falls → a weight per corpus.
- The mean still falls at epoch 8 → more epochs before a larger shape.
- A corpus is published outside the chain → a vocabulary mapping.

## Amendments

- **2026-09-19 — the first run** (`docs/verification/manual-handoff.md`): backbone `058188f2-…`,
  8 epochs × 4,493 steps at 0.39–0.42 s, 4.6 h; epoch 7 kept at 0.2295. Every corpus learnt:
  C-MAPSS 0.7 %, SKAB 26 %, SMD 37 %, ESA-AD 28 % of the trivial predictor. **The first revisit
  condition fires**: SMD rises from epoch 1 (0.348 → 0.397 → 0.373) while the mean falls, so a
  weight per corpus is the next parameter. The second does not (0.2308 at epoch 8). Validation now
  reads losses in at least fp32 (an fp16 sum overflowed in the first order).

## Sources

Kingma and Ba (2015), Adam; Arivazhagan et al. (2019), arXiv:1907.05019; Conneau et al. (2020), ACL.
