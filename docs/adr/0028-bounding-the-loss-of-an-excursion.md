# ADR-0028: Bounding what one excursion costs — the objective's loss is a parameter of the experiment, a mixed run reports validation per corpus, and the satellite corpus is split so that its excursions lie on both sides

- Status: accepted (2026-09-18, on the measurement below)
- Date: 2026-09-17
- Full text before condensation: commit `5f14447`

## Context

The saturation measurement (ADR-0027) read ESA-AD as *not learnt*. A diagnostic on the stored
backbones (`docs/verification/corpus-saturation.md`) found that:

- ordinary behaviour was learnt (a quarter to a half of the channel mean's error on 20 of 21
  held-out months);
- under a squared error, 45 % of the training gradient comes from 0.13 % of tokens — excursions of
  10–200 standard deviations that no month repeats. The four heaviest training months enter only
  at the whole corpus, where the loss on ordinary tokens jumps. Under a Huber reading the same
  backbones sit flat across shares: the curve measured the objective, not the data;
- one held-out month (88 % of that side's squared magnitude, a mission's first weeks) resembles no
  training month.

## Decision

- **The objective's loss is an experiment parameter**: `mse`, or `huber` with `huber_delta` in
  normalised units, in the file's `[objective]` section and in `parameters()`, so it reaches
  tracker, signature and handoff documents. Validation is reported under the same reading. The
  mixed run and the satellite curve use Huber with the knee at one deviation: squared error on the
  ~90 % of tokens within one deviation, bounded gradient beyond. Control and classical experiments
  keep `mse`.
- **The target stays what the tokeniser produced**; the anomaly task needs the excursions.
- **A multi-corpus run reports validation per corpus**, each with its trivial predictor's loss on
  the same tokens, logged as separate metrics. Stopping and best-epoch selection read a stated
  aggregate of per-corpus relative losses, the mean by default, so no corpus's magnitude decides.
- **ESA-AD is republished with its held-out months stated by a rule**: months ranked by mean
  square of their tokens; months past one alternate between sides in rank order, heaviest to
  training; the rest drawn by seed to keep the fraction. The Catalog split accepts an explicit list
  beside the seeded fraction. Rule, list and version are recorded before any run over it.
- **The satellite curve is rerun under the bounded loss** before the corpus's share of the mix is
  judged. ADR-0027's rules are unchanged.

## Consequences

- Every configuration's signature changes again; handoff documents move a version.
- Baselines and assessment are scored under the run's reading; per-unit summing holds for any
  additive loss.
- Epoch outcome, tracker, result document and persisted run gain per-corpus entries.
- The manifest's split seed becomes optional.
- A Huber-trained backbone reports Huber units; relative losses are still against the trivial
  predictor under the same reading.

## Alternatives considered

- *Clipping the target at k deviations*: at k = 10 an excursion still weighs 400 ordinary tokens,
  and the model is asked to predict a plateau not in the data.
- *Clipping in the tokeniser*: a second corpus under one name.
- *Dropping commissioning months*: a judgement about the data the reader was written not to make.
- *One token-weighted mixed validation loss*: the satellite corpus would decide it.
- *Stratified hold-out inside the Catalog*: month mass depends on statistics that depend on the
  split.
- *Keeping `mse` and reading only ordinary tokens*: a reading does not change what the gradient does.

## Revisit when

- The knee matters in the mixed run → it becomes an ablation dial.
- Classical curves move under Huber → `mse` stops being the default.
- A corpus whose excursions are its signal → the bound per corpus.

## Amendments

- **2026-09-18 — measured.** Small shape, same steps, Huber at one deviation, stated split:
  **data-limited** — 0.555, 0.474, 0.397, 0.270 of the channel mean's loss at a tenth, quarter, half
  and whole; the gap closes from 3.8× to 1.2×; still falling at the end. Squared error on the drawn
  split had read *not learnt* at every share. The corpus enters the mix as an ingredient still
  learning. Status → accepted.

## Sources

Huber (1964), Annals of Mathematical Statistics 35(1); Kotowski et al. (2024), arXiv:2406.17826.
