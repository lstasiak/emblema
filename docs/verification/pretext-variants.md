# Variants of the pretext

Which masks the backbones of the curve over the scale of pretraining are pretrained under. Two
variants of the mixture of four's masks, each pretrained once in today's shape, read by the
closed-form probe on a fifth of the intensive-care task's tuning side. The rule that chooses is
in `docs/preregistration.md`, "The scale of pretraining", row "The pretext".

## 2026-10-05 — declared before the run: a forecast tail, and harder masks chosen by a diagnostic

**Question.** Does a pretext that asks for the future of a window, or one that hides more of
it, give the closed-form probe at 50 stays more than the mixture's masks do? Which kinds of
mask does today's backbone learn beyond its trivial baseline on the real corpora? Until now
that has been measured only on the synthetic control (`masked-reconstruction.md`).

**Today's masks** (`backbone-mixed4-m`, `2aaca0ba-…`): channel 0.15, block 0.6 over half the
window, token 0.1; expected hidden share 0.4645.

**Variant (b), the forecast tail** (`experiments/backbone-mixed4-m-forecast.toml`). Every window
loses its tail, every channel at once. The tail's length is drawn uniformly between 0.15 and
0.5 of the window, the range Moirai draws its horizon from (Woo et al., 2024, arXiv
2402.02592). STraTS (Tipirneni and Reddy, 2022) pretrains on the same intensive-care stays by
forecasting and reports a gain at small label fractions. Blocks and single tokens are not
drawn, because both are answered from the channel's own neighbours on either side. Whole
channels are drawn at 0.15, because inferring a channel from the others is not interpolation.
Expected hidden share 0.426.

Masks drawn at seed 1 over 512 validation windows per corpus (commit of this section, M1 Pro,
CPU draw):

| Corpus | Tokens per window | Hidden, today's | Hidden, (b) | Of (b)'s hidden: channel / tail |
| --- | --- | --- | --- | --- |
| C-MAPSS | 1,050 | 0.462 | 0.412 | 0.354 / 0.646 |
| SKAB | 738 | 0.469 | 0.420 | 0.373 / 0.627 |
| Satellite telemetry | 827 | 0.469 | 0.432 | 0.351 / 0.649 |
| Intensive-care stays | 446 | 0.444 | 0.389 | 0.375 / 0.625 |

Every window kept a visible token under both strategies, and three steps on MPS ran with a
finite, falling loss on each corpus.

**Variant (a), harder masks, chosen by a rule declared now.** Point-level masking in Ti-MAE
(Li et al., 2023) works best at 75 % hidden, and random points beat contiguous spans there.
Which of today's kinds is too easy on the real corpora is not known. So (a) is read off a
diagnostic of today's backbone, not chosen by hand:

- *Diagnostic.* `backbone-mixed4-m`'s weights (encoder and decoder) on each corpus's validation
  side, under today's masks drawn at seed 1. Up to 2,000 validation windows per corpus, evenly
  spaced. The regression baselines are fitted on up to 2,000 training windows under masks
  drawn at seed 2. For each kind, the model's error is compared with the error of the baseline
  matched to that kind. The interval comes from a bootstrap over validation units, 2,000
  resamples, 95 %. A kind is *learnt* where the interval of the matched baseline's error less
  the model's lies above zero. The stays decide, as the corpus of the task that chooses; the
  other corpora are reported.
- *Shape.* If the channel kind is learnt on the stays and the block or the token kind is not,
  each kind not learnt is no longer drawn. The survival of every remaining draw is raised to
  one common power, so the expected share stays at 0.4645. A block keeps a span of half the
  window and gains frequency; its span grows only once its rate reaches 1.
- *Amount.* Otherwise (every kind learnt, or the channel kind not), today's survivals are raised
  to one common power so the expected share is 0.75.

| Outcome on the stays | channel | block (rate / span) | token | Expected share |
| --- | --- | --- | --- | --- |
| token not learnt | 0.1776 | 0.6978 / 0.5 | 0 | 0.4645 |
| block not learnt | 0.3154 | 0 | 0.2178 | 0.4645 |
| block and token not learnt | 0.4645 | 0 | 0 | 0.4645 |
| otherwise | 0.3028 | 1.0 / 0.5469 | 0.2085 | 0.7500 |

The chosen row becomes `experiments/backbone-mixed4-m-harder.toml`, everything else equal to
`backbone-mixed4-m.toml`. Each variant's backbone is also diagnosed under its own masks, with
(b)'s tail scored against its last visible value carried forward.

**A second range for the tail, only if the first is not learnt.** Moirai's range was chosen for
regular series in patches. On the stays it hides 7 to 24 hours of 48, which may be close to
unpredictable. If (b)'s diagnostic under its own masks finds the tail kind not learnt on the
stays (the interval of the last value's error less the model's does not lie above zero), the
range is taken as wrong for these data, not forecasting as useless. One run (b′) then uses the
range STraTS forecasts over on the same stays: 2 hours after windows of 12 to 44 hours, so a
tail of 0.04 to 0.15 of the window, everything else as (b). (b′) enters the reading in (b)'s
place. No further range is tried: each variant read by the same probe on the same fifth adds a
chance of a false replacement. If (b′)'s tail is not learnt either, the forecast variant is
recorded as trivial on the stays and leaves the reading.

**Pretraining.** Each variant is pretrained by the file of `backbone-mixed4-m` to the step:
8 epochs, seed 1, the epoch kept by the mean relative validation loss under its own masks. Runs
are on Kaggle T4, one order per GPU.

**Reading.** `campaigns/pretext-physionet2012.toml` covers the intensive-care task
(`a6a9c653…`) at 20, 50 and 200 stays, with `frozen_ridge`, the trained probe
(`frozen_probe@learning_rate=0.03`) and `untrained_ridge`. It is scored on the fifth held out
by seed 101 over seeds 1 to 10 (the fifth the recipe was chosen on). `campaigns/pretext-50-fd001.toml`
covers FD001 at 50 windows on its recipe fifth. Both are defined under four backbones:
`backbone-mixed4-m` at seeds 1 and 2, (a) and (b). Cells run here on MPS from orders. Pairs are
computed with `scripts/campaign_pairs_report.py`, over stays and seeds.

- A variant **replaces** today's masks when its `frozen_ridge` at 50 stays, against the
  `frozen_ridge` under seed 1, meets three conditions: the paired interval lies above zero;
  the mean over seeds exceeds twice its standard error; and the mean exceeds the absolute
  difference between seed 2 and seed 1 on the same fifth. If both variants replace, the larger
  mean wins. If neither replaces, the curve continues under today's masks, and that is recorded
  here.
- 20 and 200 stays, the trained probe and FD001 are read descriptively.

**Predictions.**

1. The two seeds of the mixture differ by less than 0.012 at 50 stays on the fifth, the upper
   edge of their difference on the validation side (+0.004 [−0.004; +0.012],
   `intensive-care-curve.md`, 2026-10-05).
2. Neither variant replaces today's masks. Each one's gain at 50 stays lies within ±0.021 of
   the probe under seed 1. That is the upper edge of all pretraining adds to the same probe
   over an untrained encoder at 50 stays (+0.009 [−0.002; +0.020] and +0.013 [+0.005; +0.021]
   under the two seeds, validation side). A change of masks is not expected to move the probe
   by more than pretraining moves it.
3. The diagnostic, and so (a)'s row, gets no prediction: no kind has been measured against its
   baseline on a real corpus before.

**Limitations.**

- Only the tail has a fallback: its range is the one setting taken from another kind of data.
  (a)'s values follow from the diagnostic, and (a) is read whatever its pretext's outcome.
- One pretraining seed per variant. The seed condition bounds the comparison by the mixture's
  seed difference, not by the variant's own.
- (b) hides less than today's masks on every corpus, most on the stays (0.389 against 0.444).
  Its range is Moirai's, not fitted to these corpora, so a difference under (b) is read as a
  difference of pretext and amount together.
- The fifth was also used to choose the recipe; the validation side stays unread.
- The probe alone chooses. A pretext that helps fine-tuning but not a linear reading would be
  missed.
