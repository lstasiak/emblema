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

## 2026-10-05 — M1 Pro, MPS: the kinds of mask read off the diagnostic, and the harder masks set

**Question.** Which kinds of mask does the mixture of four's backbone learn beyond their trivial
baselines on its real corpora, which row of the table above do the stays choose for (a), and is
the forecast tail of (b) learnt on the stays, so that (b′) stays unrun?

**Conditions.** Commit `f5c4d502`; `scripts/pretext_triviality_report.py --backbone <id>` on
the M1 Pro (MPS, fp32, torch 2.14). Per corpus: 2,000 validation windows spaced evenly (SKAB
has 768), the model scored under its own masks drawn at seed 1; the regression baselines fitted
on 2,000 training windows under masks drawn at seed 2; 2,000 bootstrap resamples over
validation units, 95 %. Errors are the run's reading of the loss (Huber at one deviation) on
hidden tokens, in normalised units. Channels reported apart (timeless, constant) are left out
here. CSV under `data/report/pretext/<backbone id>/`; 1.5 to 3.5 minutes a backbone.

**`backbone-mixed4-m` (`2aaca0ba-…`) under today's masks.** Excess = the matched baseline's
error less the model's; bold where learnt.

| Corpus | Kind | Tokens | Units | Model | Matched | Excess [95 %] | Beyond linear [95 %] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C-MAPSS | channel | 254,140 | 141 | 0.105 | ridge 0.137 | **+0.031** [+0.013; +0.057] | — |
| | block | 409,001 | 141 | 0.099 | interpolation 0.167 | **+0.068** [+0.065; +0.071] | **[+0.011; +0.032]** |
| | token | 94,965 | 141 | 0.100 | interpolation 0.160 | **+0.060** [+0.056; +0.064] | **[+0.020; +0.027]** |
| SKAB | channel | 85,175 | 7 | 0.157 | ridge 0.205 | **+0.048** [+0.022; +0.078] | — |
| | block | 142,405 | 7 | 0.123 | interpolation 0.178 | **+0.054** [+0.047; +0.064] | [−0.001; +0.002] |
| | token | 33,995 | 7 | 0.121 | interpolation 0.159 | **+0.038** [+0.036; +0.041] | [−0.004; −0.001] |
| Satellite telemetry | channel | 261,680 | 21 | 0.119 | ridge 0.307 | **+0.188** [+0.058; +0.408] | — |
| | block | 418,391 | 21 | 0.079 | interpolation 0.157 | **+0.078** [+0.006; +0.140] | **[+0.056; +0.112]** |
| | token | 98,433 | 21 | 0.075 | interpolation 0.061 | −0.014 [−0.084; +0.036] | [−0.032; +0.031] |
| Intensive-care stays | channel | 133,182 | 1,979 | 0.240 | ridge 0.258 | **+0.019** [+0.013; +0.025] | — |
| | block | 207,806 | 1,998 | 0.144 | interpolation 0.228 | **+0.085** [+0.079; +0.090] | **[+0.023; +0.031]** |
| | token | 50,565 | 1,989 | 0.122 | interpolation 0.145 | **+0.023** [+0.019; +0.027] | **[+0.001; +0.006]** |

**`backbone-mixed4-m-forecast` (`4ae060e1-…`) under its own masks.** The tail's matched
baseline is the channel's last visible value carried forward.

| Corpus | Kind | Tokens | Units | Model | Matched | Excess [95 %] | Beyond linear [95 %] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C-MAPSS | channel | 248,385 | 141 | 0.107 | ridge 0.159 | **+0.052** [+0.015; +0.114] | — |
| | tail | 428,573 | 141 | 0.106 | last value 0.196 | **+0.090** [+0.086; +0.093] | **[+0.016; +0.057]** |
| SKAB | channel | 85,175 | 7 | 0.156 | ridge 0.203 | **+0.046** [+0.022; +0.075] | — |
| | tail | 153,552 | 7 | 0.128 | last value 0.207 | **+0.079** [+0.073; +0.085] | **[+0.000; +0.004]** |
| Satellite telemetry | channel | 261,680 | 21 | 0.133 | ridge 0.285 | **+0.153** [+0.037; +0.349] | — |
| | tail | 461,088 | 21 | 0.094 | last value 0.216 | **+0.122** [+0.021; +0.209] | **[+0.076; +0.128]** |
| Intensive-care stays | channel | 132,178 | 1,963 | 0.275 | ridge 0.263 | −0.012 [−0.018; −0.005] | — |
| | tail | 205,800 | 1,986 | 0.181 | last value 0.239 | **+0.057** [+0.052; +0.064] | **[+0.021; +0.027]** |

**Conclusions.**

1. On the stays today's backbone learns every kind: the channel kind by 0.019, the block kind by
   0.085 and the token kind by 0.023, each by the interval and each beyond the linear answer on
   the same sources. By the rule above, **(a) is the amount branch**: channel 0.3028, block 1.0
   over 0.5469 of the window, token 0.2085, expected hidden share 0.75
   (`experiments/backbone-mixed4-m-harder.toml`). No kind is switched off.
2. On the other corpora the same holds except the satellite telemetry's token kind, which the
   model matches but does not beat (−0.014 [−0.084; +0.036]) over its 21 months, and SKAB's
   block and token kinds, which the model beats the matched baseline on but not the linear
   answer that reads the other channels too.
3. The forecast tail is learnt on every corpus, on the stays by 0.057 [+0.052; +0.064] over the
   last value and beyond the linear answer. **(b′) is not run.**
4. Under the forecast tail the backbone reads the stays' hidden channels *worse* than the
   cross-channel regression (−0.012 [−0.018; −0.005]), where today's backbone reads them better
   by 0.019. A pretext that asks for the future has not taught the stays' channels about one
   another; what that does to the task is the probes' question.
5. The channel kind's intervals on C-MAPSS and the satellite telemetry are wide (a few units
   carry most of the error); the token kind on the satellite corpus is the one case the mixture's
   masks ask for something the model does not learn.

**Limitations.**

- One draw of the masks per backbone; the intervals hold the variation between units under it.
- SKAB has seven validation units, the satellite telemetry twenty-one months; their intervals
  are wide and their verdicts coarse.
- The baselines are fitted on 2,000 training windows, not the training side; a ridge fitted on
  more could be a little stronger.
- Forward passes in fp32 on MPS, not the run's fp16 on the T4; the loss the run reports is not
  reproduced here and is not what is compared.

