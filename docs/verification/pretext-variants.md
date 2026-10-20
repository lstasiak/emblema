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


## 2026-10-06 — M1 Pro, MPS: the harder masks' backbone under its own masks

**Question.** Does the backbone pretrained under (a), three quarters of a window hidden, learn
each of the three kinds beyond its baseline on the real corpora, as today's backbone does?

**Conditions.** As the section above: commit `a406ed0e`, `scripts/pretext_triviality_report.py
--backbone 7092e656-…` on the M1 Pro (MPS, fp32), 2,000 windows a side, masks of (a) at seeds
1 and 2, 2,000 resamples, 95 %; 2.2 minutes. The backbone is the one accepted in
`manual-handoff.md` (2026-10-06), weights `sha256:8937743e…`. The masks differ from the
section above, so the errors are read against each backbone's own baselines, not across the
two tables.

**`backbone-mixed4-m-harder` (`7092e656-…`) under its own masks.**

| Corpus | Kind | Tokens | Units | Model | Matched | Excess [95 %] | Beyond linear [95 %] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C-MAPSS | channel | 517,388 | 141 | 0.109 | ridge 0.146 | **+0.037** [+0.021; +0.065] | — |
| | block | 596,122 | 141 | 0.101 | interpolation 0.172 | **+0.071** [+0.067; +0.075] | **[+0.015; +0.031]** |
| | token | 102,972 | 141 | 0.102 | interpolation 0.171 | **+0.069** [+0.064; +0.074] | **[+0.025; +0.034]** |
| SKAB | channel | 173,433 | 7 | 0.171 | ridge 0.244 | **+0.072** [+0.040; +0.105] | — |
| | block | 215,433 | 7 | 0.126 | interpolation 0.182 | **+0.056** [+0.050; +0.064] | [−0.000; +0.002] |
| | token | 37,099 | 7 | 0.123 | interpolation 0.165 | **+0.042** [+0.038; +0.047] | [−0.001; +0.001] |
| Satellite telemetry | channel | 514,222 | 21 | 0.140 | ridge 0.308 | **+0.169** [+0.087; +0.271] | — |
| | block | 628,273 | 21 | 0.089 | interpolation 0.176 | +0.086 [−0.003; +0.162] | **[+0.079; +0.149]** |
| | token | 108,460 | 21 | 0.073 | interpolation 0.061 | −0.012 [−0.080; +0.037] | [−0.015; +0.041] |
| Intensive-care stays | channel | 269,073 | 1,997 | 0.284 | ridge 0.294 | **+0.010** [+0.006; +0.015] | — |
| | block | 307,832 | 1,998 | 0.189 | interpolation 0.243 | **+0.054** [+0.049; +0.060] | **[+0.008; +0.015]** |
| | token | 55,869 | 1,986 | 0.167 | interpolation 0.165 | −0.002 [−0.006; +0.003] | [−0.020; −0.014] |

**Conclusions.**

1. Under three quarters of a window hidden, the stays' single tokens are no longer learnt: the
   model matches the interpolation between neighbours (−0.002 [−0.006; +0.003]) and reads them
   worse than the linear answer from the other channels (−0.020 to −0.014). Today's backbone
   learnt them by 0.023 under today's masks. The channel kind is learnt by half of today's
   margin (0.010 against 0.019), the block kind by two thirds (0.054 against 0.085).
2. On C-MAPSS and SKAB every kind is learnt, by margins at least those under today's masks.
   On the satellite telemetry the block kind drops to matched (its interval crosses zero by
   0.003 over 21 months) and the token kind stays matched.
3. The run hid what it was asked to hide (0.749 of the tokens) and learnt a pretext of the same
   kinds, but the harder the window, the more the stays' answer approaches the one a line
   between neighbours gives. Whether that reading of a stay is worth more or less to the probe
   is the campaigns' question; nothing here changes the rule.

**Limitations.**

- The same as the section above: one draw of the masks, seven SKAB units, twenty-one months of
  telemetry, baselines fitted on 2,000 windows, fp32 on MPS.
- The two backbones are diagnosed under different masks, so "learnt by less" compares each to
  its own baseline on its own hidden tokens, not the two models on one set of tokens.

## 2026-10-06 — M1 Pro, MPS: the probes under the four backbones, read by the rule

**Question.** The one declared on 2026-10-05: does the closed-form probe at 50 stays gain more
under the forecast tail (b) or the harder masks (a) than under the mixture's masks, past the
mixture's own difference between two pretraining seeds?

**Conditions.** Commit `7d641c16`; `campaigns/pretext-physionet2012.toml` and
`campaigns/pretext-50-fd001.toml` defined under four backbones — `backbone-mixed4-m` at seed 1
(`2aaca0ba-…`, weights `0d81c01e…`) and seed 2 (`3d0c4468-…`, `94f00d72…`), (b)
(`4ae060e1-…`, `345ade99…`) and (a) (`7092e656-…`, `8937743e…`) — eight campaigns in all,
ordered here and fulfilled on this machine's accelerator one after another (MPS, fp32;
about 15 minutes for 90 cells on the stays and 5 minutes for 30 on FD001, 80 minutes in all).
Every cell is scored on the fifth of the tuning side held out by seed 101, 800 stays or 16
engines, over seeds 1 to 10, so the backbones pair by stay and seed; the validation side is
unread. Pairs by `scripts/campaign_pairs_report.py` (10,000 resamples, 95 %), CSV under
`data/report/l2/pairs/`. Gains are reductions of 1 − AUROC on the stays and of RMSE in cycles
on FD001; positive favours the candidate.

**The rule's comparison: `frozen_ridge` under a variant against `frozen_ridge` under seed 1.**

| Candidate | 20 stays | 50 stays | 200 stays | FD001 at 50 |
| --- | --- | --- | --- | --- |
| seed 2 | −0.021 [−0.039; −0.003] | −0.008 [−0.028; +0.011], −0.008 ± 0.007 | +0.005 [−0.019; +0.030] | +0.4 [−0.6; +1.4] |
| (b) forecast tail | −0.027 [−0.042; −0.012] | −0.017 [−0.038; +0.005], −0.017 ± 0.012 | −0.038 [−0.064; −0.011] | **+4.1 [+3.0; +5.3]** |
| (a) harder masks | −0.012 [−0.033; +0.008] | −0.015 [−0.030; +0.000], −0.015 ± 0.008 | +0.000 [−0.023; +0.024] | +1.4 [+0.3; +2.6] |

At 50 stays the second number is the mean over seeds with its standard error. The trained
probe (`frozen_probe@learning_rate=0.03`) reads the same way: seed 2 −0.004, (b) −0.029
[−0.051; −0.008], (a) −0.013 [−0.032; +0.006] at 50 stays.

**What pretraining adds: `frozen_ridge` against `untrained_ridge` under each backbone.**

| Backbone | 20 stays | 50 stays | 200 stays | FD001 at 50 (RMSE) |
| --- | --- | --- | --- | --- |
| seed 1 | +0.016 [−0.005; +0.037] | **+0.032** [+0.007; +0.057] | **+0.051** [+0.016; +0.085] | −5.1 [−6.2; −4.0] |
| seed 2 | −0.005 [−0.023; +0.013] | **+0.024** [+0.003; +0.045] | **+0.056** [+0.028; +0.084] | −4.7 [−5.7; −3.8] |
| (b) forecast tail | −0.011 [−0.030; +0.007] | +0.015 [−0.014; +0.044] | +0.013 [−0.024; +0.048] | −1.0 [−1.6; −0.3] |
| (a) harder masks | +0.004 [−0.019; +0.026] | +0.018 [−0.009; +0.043] | **+0.051** [+0.019; +0.082] | −3.7 [−4.5; −2.8] |

The probe over the untrained encoder scores 0.459, 0.429 and 0.394 at 20, 50 and 200 stays
and 18.8 RMSE on FD001; it is the same cell under every backbone.

**Conclusions.**

1. **Neither variant replaces the mixture's masks.** At 50 stays both lie below the probe under
   seed 1, (b) by 0.017 and (a) by 0.015, and neither interval lies above zero; the first
   condition fails and the other two are not reached. The curve over the scale of pretraining
   continues under today's masks, as the rule provides.
2. Both predictions hold. The mixture's two seeds differ by 0.008 at 50 stays, under the 0.012
   predicted from the validation side; each variant's gain lies within ±0.021 of seed 1.
3. **The forecast tail unlearns the stays for the probe.** Under (b) pretraining adds nothing
   over an untrained encoder at any budget (+0.015 [−0.014; +0.044] at 50 stays, +0.013 at
   200), where the mixture's masks add 0.032 and 0.051. This is the diagnostic's reading made
   concrete: a backbone that read the stays' hidden channels worse than a regression on the
   others gives the probe states no better than untrained ones.
4. **The harder masks keep the gain at 200 stays and lose it below.** Under (a) the probe gains
   0.051 at 200, as under seed 1, but 0.018 at 50 and 0.004 at 20, where the mixture's masks
   give 0.032 and 0.016.
5. **On FD001 every pretext hurts the linear reading of the remaining life, (b) least.** The
   probe over the untrained encoder scores 18.8; the mixture's states 23.9 and 23.5 (two
   seeds), (a)'s 22.5 and (b)'s 19.8. The tail is 4.1 [+3.0; +5.3] RMSE better than seed 1
   and (a) 1.4 [+0.3; +2.6]; both are read descriptively and neither reaches the untrained
   encoder.
6. At 20 stays seed 2 lies 0.021 [−0.039; −0.003] below seed 1: the budget where a
   pretraining seed moves the probe by the interval, and where no variant is read.

**Limitations.**

- One pretraining seed per variant, so a variant's own seed spread is unknown; the seed
  condition uses the mixture's.
- The fifth is the one the recipe was chosen on. The validation side stays unread.
- The probe alone is read; fine-tuning under either variant was not run and could order them
  otherwise.
- (b) hides less than today's masks on every corpus, so its deficit is pretext and amount
  together; (a) hides more at the same proportions, so its deficit at small budgets is amount
  alone.
- Forward passes on MPS in fp32, the first orders of the ml pool fulfilled on this machine; the
  cells are deterministic under their seeds.

## 2026-10-08 — the probes under the four backbones, read again by the seeds

The section of 2026-10-06 read its descriptive differences by the paired interval, which holds
the seeds and cannot see a draw of labels (`verdict-statistics.md`, 2026-10-08). Its rule
carried the seed condition, so its verdict stands: neither variant replaces the mixture's masks.
By the seed condition (mean over seeds above twice its standard error), one statement is
withdrawn and one is narrowed:

| Comparison | Stays | Mean ± SE over seeds | Seed condition |
|---|---|---|---|
| closed-form probe, pretraining seed 2 less seed 1 | 20 | −0.021 ± 0.012 | fails |
| closed-form probe, forecast tail less seed 1 | 50; 200 | −0.017 ± 0.012; −0.038 ± 0.012 | fails; holds |
| trained probe, forecast tail less seed 1 | 50 | −0.029 ± 0.011 | holds |
| closed-form probe less the untrained encoder, seed 1; seed 2 | 50 | +0.032 ± 0.012; +0.024 ± 0.012 | holds |
| the same under the forecast tail | 50; 200 | +0.015 ± 0.014; +0.013 ± 0.015 | fails |

- Withdrawn: that a pretraining seed moves the probe by the interval at 20 stays.
- Narrowed: the forecast tail lowers the probe at 200 stays and the trained probe at 50, past a
  draw of labels. That it adds nothing over an untrained encoder holds, but so would a gain
  smaller than about 0.03.

## 2026-10-11 — declared before the run: the objective, predicting the teacher's states

**Question.** Does a pretext that asks for the *state* of a hidden token rather than its value
give the closed-form probe at 50 stays more than the mixture's masks under the value objective
do? Under every backbone so far the probe reads FD001 worse than the same probe over an untrained
encoder, and on dense channels a hidden value is answered by a line through its neighbours
(ADR-0054). Does the gap on FD001 close? Is the new pretext learnt beyond its own trivial
answers, and do its targets keep their spread?

**The variant** (`experiments/backbone-mixed4-latent-m.toml`, `-seed2.toml`). The file of
`backbone-mixed4-m` to the step — the four corpora, 4.8M parameters, the mixture's masks (channel
0.15, block 0.6 over half the window, token 0.1), huber at one deviation, eight epochs, 16 × 2
windows a step, 1e-3 — with one section changed: `[objective] target = "latent"`,
`target_layers = 4`, `momentum = 0.996`, `final_momentum = 1.0`. The hidden tokens' targets are
the states a teacher gives them from the whole window: a copy of the encoder trailing it by an
exponential moving average whose momentum rises linearly from 0.996 to one over the run. The
target of a token is the mean of the teacher's top four of six blocks, each normalised without
a learnt scale and shift. The predictor is the decoder's shape with a head as wide as the state;
the loss is the reading per coordinate averaged over the width. The trivial predictor is the
mean state of the window's visible tokens; the kept epoch is the one with the lowest loss over
it, as under the value objective. Everything else the value objective does, this one does.

**Pretraining.** Two seeds, 1 and 2, on Kaggle T4 ×2, one per GPU, from the orders of the
committed experiment files. About 2.8 h each: the teacher's forward pass adds about a third of a
step over the mixture's 2.1 h.

**Diagnostic, before the reading.** `scripts/latent_prediction_report.py --backbone <id>` for
each backbone, on each corpus's validation side under the mixture's masks drawn at seed 1, up to
2,000 windows per corpus, 2,000 resamples over units. Per kind of mask the predictor's error is
read against the line between the hidden token's visible neighbours' states (the matched
baseline), the nearest neighbour's state and the visible mean; a channel hidden whole has no
neighbours and is read against the mean. A kind is *learnt* where the interval of the line's
error less the predictor's lies above zero. Beside it, the targets' variance over the observed
tokens and the effective rank of their covariance. The run's own curve is read off the tracker:
the trivial loss of each corpus per epoch is `validation_loss_<corpus>` over
`relative_validation_<corpus>`.

- *Collapse.* A backbone is read as collapsed, and leaves the reading, where on the stays the
  trivial loss at the kept epoch is below half of its value at epoch 1, or the effective rank of
  the targets is below 4 of 256. Neither is expected (prediction 4).

**Reading.** The campaigns of the pretext rule, defined under each of the two backbones:
`campaigns/pretext-concat-half-physionet2012.toml` (20, 50 and 200 stays on the half held out
by seed 101, seeds 1 to 10) and `campaigns/yardstick-concat-physionet2012.toml` (the validation
side, read descriptively). FD001 at 50 windows by `campaigns/pretext-concat-50-fd001.toml`, a
new file — the blocks side by side had not been read on FD001 under the mixture — defined under
the mixture's four pretraining seeds and the two of the objective. Cells run here on MPS from
orders. Pairs by `scripts/campaign_pairs_report.py`; a side of several campaigns, joined by
`+`, is read as the mean over them.

- The objective **replaces** the mixture's masks when, at 50 stays on the half,
  `frozen_ridge@layer=concat` under the two backbones of the objective, read as one side,
  against the same probe under the mixture's four, read as one side, has its paired interval
  above zero and its mean over seeds above twice √(SE² + (s²/2)·(1/2 + 1/4)) with s = 0.0060
  (`docs/preregistration.md`, "The pretext", 2026-10-11). With SE about 0.01 the threshold is
  about 0.02 to 0.025.
- Read descriptively: the eight differences of one pretraining against one; 20 and 200 stays;
  the validation side; FD001 — the probe less the untrained encoder's probe under each backbone,
  against the same under the mixture — and the probe against the network from nothing at 50
  stays on the validation side, whose cells under the mixture are already read.

**Predictions.** Where no earlier interval bears on a prediction, it is marked as a judgement.

1. The objective does not replace the mixture's masks: its mean over seeds at 50 stays on the
   half lies within [−0.02, +0.03]. Basis: the two variants of the masks lay at −0.017
   [−0.038; +0.005] and −0.015 [−0.030; 0.000] against seed 1 (2026-10-06), and two pretrainings
   of one recipe differ by up to 0.02; a gain past the threshold is the first that would.
2. On FD001 at 50 windows the probe under the objective lies above the untrained encoder's probe
   by less than 3.9 RMSE under both backbones, the lower edge of 3.9 to 5.9 under the value
   objective (`intensive-care-curve.md`, 2026-10-05). That the interval of the gap holds zero is
   a judgement from the question asked, not from an interval.
3. At 50 stays on the half the probe under the objective gains at least +0.015 over the
   untrained encoder's probe under both backbones, the lower edge of the mixture's +0.015 and
   +0.020 (`layer-readout.md`, 2026-10-09). At 20 stays nothing is settled either way
   (`verdict-statistics.md`).
4. The pretext is learnt for blocks and whole channels on the stays and C-MAPSS under both
   backbones, and the targets do not collapse: the trivial loss at the kept epoch is at least
   half of epoch 1's and the effective rank at least a quarter of the width (64 of 256), far
   above the floor of 4 that reads as a collapse. A judgement: no latent pretext has been read
   before.

**Limitations.**

- One set of knobs: the top four blocks, 0.996 to one, the decoder's depth. None is tuned; a
  worse setting reads as a worse objective.
- The probe alone chooses, as for the masks. A target that helps fine-tuning but not a linear
  reading would be missed.
- Two pretrainings on the candidate's side against four: the threshold's spread term is small
  (about 0.004) and the draw of labels still decides, so a gain below about 0.02 is not seen.
- The latent diagnostic's baselines are not the value diagnostic's: a kind learnt under one need
  not be learnt under the other, and the two are not compared.

## 2026-10-11 — Kaggle T4 and M1 Pro, MPS: the objective that predicts states, read by the rule

**Question.** The one declared above.

**Conditions.** Pretraining: `backbone-mixed4-latent-m` (`ff60eb67-…`, weights `d321d1f6…`) and
`-seed2` (`39cffeb1-…`, `0d548b33…`), commit `589757a9`, Kaggle T4 ×2, fp16, tier M, about
1,150 s an epoch against the mixture's 935 s (+23 %, the teacher's forward pass). Campaigns on
the M1 Pro, MPS, fp32 probes, ten campaigns in 3.5 h (`data/report/l13/`); pairs by
`scripts/campaign_pairs_report.py` with a side of several campaigns joined by `+`
(`data/report/l13/pairs/group-*/comparisons.csv`, rule read by `data/report/l13/rule.py`).
Diagnostic by `scripts/latent_prediction_report.py` at `121aff3f`
(`data/report/pretext-latent/<backbone>/`). A reduction is the control's shortfall less the
candidate's: positive favours the candidate.

**The runs.** The kept epoch and the pretext's validation, loss over the visible mean state
(`data/report/pretraining/l13-latent-seed{1,2}/epochs.csv`):

| Backbone | Kept epoch | Mean relative at the kept epoch | Trivial on the stays, epoch 1 → kept |
| --- | --- | --- | --- |
| seed 1 | 8 of 8 | 0.616 (epoch 1: 3.18, the predictor worse than the mean) | 0.014 → 0.072 |
| seed 2 | 2 of 8 | 0.526 (epoch 8: 0.591) | 0.105 → 0.104 |

**The diagnostic.** Every kind of mask is learnt on every corpus beyond the line between the
neighbours' states, except the stays' single tokens (beaten by the line under both) and the
satellite telemetry's single tokens under seed 2 (matched). The targets, read over 6,000 tokens
spread evenly across the validation windows:

| Corpus | Effective rank, teacher at initialisation | Effective rank, trained, seed 1 | seed 2 |
| --- | --- | --- | --- |
| C-MAPSS | 11.0 | 5.2 | 3.5 |
| SKAB | 8.0 | 2.7 | 2.2 |
| Satellite telemetry | 4.4 | 6.2 | 5.1 |
| Intensive-care stays | 16.1 | **4.7** | **3.1** |

The targets' variance over tokens fell from 0.61 to 0.68 at initialisation to 0.19 to 0.33.
The first run of the diagnostic sampled the first 4,096 tokens of each corpus, a few windows,
and read 2.2 to 4.3 on every corpus; the sample was spread over every window before the reading
(`121aff3f`) and the floor of 4 on the stays applied to that reading. **Seed 2 reads as
collapsed by the declared floor and leaves the rule's reading; seed 1 stays, at 4.7.**

**The rule.** `frozen_ridge@layer=concat` at 50 stays on the half, the mixture's four
pretrainings as one side (s = 0.0060):

| Candidate side | Mean over seeds ± SE | Paired 95 % interval | Threshold | Verdict |
| --- | --- | --- | --- | --- |
| seed 1 alone (n_c = 1) | −0.029 ± 0.012 | [−0.042; −0.015] | 0.026 | does not replace |
| both seeds (n_c = 2, declared before seed 2 left) | −0.034 ± 0.014 | [−0.045; −0.024] | 0.029 | does not replace |

The interval lies below zero and the mean exceeds the threshold on the other side: the objective
reads the stays **worse** than the mixture's masks, past the spread two pretrainings give.

**Descriptive.** At 20 and 200 stays on the half both sides lie below the mixture by 0.019 to
0.036 (seed 1) and 0.025 to 0.060 (seed 2), every interval below zero; on the validation side by
0.025 to 0.036. The eight differences of one pretraining against one at 50 stays run from −0.019
to −0.045, each interval below zero. Over the untrained encoder's probe at 50 stays: seed 1
−0.014 [−0.033; +0.005] on the half and +0.004 [−0.008; +0.016] on validation, seed 2 −0.025
[−0.044; −0.006] and −0.016 [−0.028; −0.005]. Against the network from nothing at 50 stays on
validation: −0.028 [−0.042; −0.013] and −0.048 [−0.060; −0.036]. FD001 at 50 windows, RMSE, the
probe over the blocks side by side:

| Backbone | Untrained encoder | Pretrained | Pretrained less untrained |
| --- | --- | --- | --- |
| mixture, seeds 1–4 | 19.0 | 20.4 to 22.9 | +1.4 to +3.8 |
| objective, seed 1 | 19.0 | 28.5 | +9.5 [+8.8; +10.1] |
| objective, seed 2 | 19.0 | 24.6 | +5.5 [+4.9; +6.2] |

**Conclusions.**

- **Predicting the teacher's states does not replace masked reconstruction**; it reads the stays
  worse by 0.03 at 50 stays, past the spread of pretraining, and worse at every budget on both
  sides.
- **Its probe is no better than an untrained encoder's** on the stays (seed 1) or worse (seed 2),
  and worse than the network from nothing by 0.03 to 0.05.
- **The gap on FD001 widens**, from +1.4 to +3.8 RMSE under the mixture to +5.5 and +9.5: the
  states this objective learns read the remaining life worse, not better.
- **The targets collapsed in dimension**: from 16 to 18 directions at initialisation to 3 to 5
  after training on the stays, while the predictor still beat the line along the channel. A
  teacher trailing by a moving average with targets normalised without parameters was not enough
  to hold the states apart at this width and depth; the pretext stayed "learnt" because the line
  through collapsed states is a poor answer too.
- The mixture's masks stay the pretext of the curve; the largest point's rule is not triggered
  by this reading.

**Predictions.** 1 held in its verdict and failed in its range: the mean lay at −0.034, below
−0.02. 2 withdrawn: the gap on FD001 grew. 3 withdrawn: nothing over the untrained encoder.
4 withdrawn: the targets' rank fell to 3 to 5 of 256, below the floor under seed 2.

**What follows.** The second objective stays in the code as a measured alternative. Its knobs —
the blocks averaged, the momentum, a variance term in the loss (Bardes et al. 2022) — were not
tuned here and would each be a declaration of its own; the collapse is the first thing such a
run would have to prevent. The curve over the scale of pretraining, and any larger point, keep
the mixture's masks.

**Limitations.**

- One setting of the objective, two pretrainings. A setting that holds the targets apart may
  read differently; this reading says nothing of it.
- The collapse floor was declared on a sample of the first tokens and read on a sample spread
  over windows; the floor itself did not move, and both samples read seed 2 below it.
- The probe alone chooses; the trained probe and full fine-tuning were not read under this
  objective.

## 2026-10-12 — M1 Pro, MPS: the objective's states held against the mixture's own

**Question.** The reading above names a collapse of the targets as the mechanism. Is the
concentration specific to this objective, or does the encoder trained on values concentrate its
states the same way? And what else the runs' records say about why the probe reads these states
worse than an untrained encoder's.

**Conditions.** `scripts/state_spread_report.py --backbone <id>` at commit `d30a7951`, M1 Pro,
MPS, fp32: per corpus 500 validation windows, 4,096 observed tokens spread evenly over them
(seed 1), the same tokens under every reading. Readings: `top`, the mean of the top four blocks
each normalised without parameters (the second objective's target, read off any encoder);
`last`, the encoder's output as a probe of the last block reads it; `teacher`, the objective's
targets where it has one; the same under the encoder at its initialisation. Of each: the
variance over tokens, the effective rank of the covariance, and the share of the sum of squares
that the token's channel accounts for, alone and with the decile of its instant.
Backbones: the objective's two pretrainings and the mixture of four's seeds 1 to 3
(`2aaca0ba-…`, `3d0c4468-…`, `2929a139-…`). Files: `data/report/state-spread/<backbone>/`.
The runs' curves: `data/report/pretraining/l13-latent-seed{1,2}/epochs.csv` and the mixture's.

**The stays (PhysioNet 2012), the `last` reading unless stated.**

| States | Variance | Effective rank | Channel share | Channel and decile share |
| --- | --- | --- | --- | --- |
| any encoder at initialisation | 0.75 to 0.76 | 27.7 to 28.7 | 0.77 to 0.79 | 0.80 to 0.84 |
| objective, seed 1, `teacher` | 0.28 | 5.0 | 0.08 | 0.14 |
| objective, seed 1 | 0.27 | 5.6 | 0.07 | 0.13 |
| objective, seed 2, `teacher` | 0.27 | 3.0 | 0.07 | 0.12 |
| objective, seed 2 | 0.19 | 3.8 | 0.06 | 0.11 |
| mixture, seed 1 | 0.13 | 7.6 | 0.24 | 0.44 |
| mixture, seed 2 | 0.26 | 7.4 | 0.60 | 0.73 |
| mixture, seed 3 | 0.14 | 7.6 | 0.24 | 0.36 |

**The other corpora, effective rank of the `last` reading.**

| Corpus | Initialisation | Objective, seeds 1 and 2 | Mixture, seeds 1 to 3 |
| --- | --- | --- | --- |
| C-MAPSS | 25.3 to 25.7 | 6.6, 4.3 | 7.1, 3.7, 6.2 |
| SKAB | 9.3 to 9.7 | 3.5, 2.4 | 5.6, 5.9, 4.0 |
| Satellite telemetry | 4.8 to 5.1 | 7.5, 5.8 | 6.6, 6.6, 8.9 |

**The runs.** The pretext's validation at the kept epoch, loss over the trivial predictor (the
channel mean for the mixture, the visible mean state for the objective):

| Corpus | Objective, seed 1 (epoch 8 of 8) | Objective, seed 2 (epoch 2 of 8) | Mixture, seed 1 (epoch 8) |
| --- | --- | --- | --- |
| C-MAPSS | 0.57 | 0.47 | 0.31 |
| SKAB | 0.50 | 0.52 | 0.27 |
| Satellite telemetry | 0.53 | 0.41 | 0.27 |
| Intensive-care stays | 0.86 | 0.70 | 0.49 |

Under the objective the trivial predictor's loss on the stays was 0.014 after the first epoch
(seed 1) against a variance of the initial teacher's targets over tokens of 0.75: the targets had
narrowed before the first validation, and rose to 0.07 by the end. The declared collapse check
read the trivial loss against epoch 1, so it could not see this. The kept epoch is the lowest
loss over trivial, a rule written for a fixed baseline: under a teacher that moves, seed 2's
baseline grew faster than its loss fell and the run kept its second epoch.

**Conclusions.**

- **The concentration is about half the mixture's own rank on the stays, not a collapse**: 3.8
  and 5.6 directions against 7.4 to 7.6 under three pretrainings of the mixture. On the other
  corpora the mixture spans the same range (3.7 to 8.9), so the rank alone does not mark this
  objective's failure. The earlier conclusion of a collapse is narrowed to this.
- **What the states lose is their channel and instant.** At initialisation the channel and the
  decile account for 0.80 to 0.84 of the spread; the mixture keeps 0.36 to 0.73 of it, the
  objective 0.11 to 0.14. A probe that pools tokens then cannot tell which channel a value came
  from, which is consistent with its reading at or below an untrained encoder's.
- **The pretext was barely learnt on the stays**: 0.86 and 0.70 of the trivial predictor's loss
  against the mixture's 0.49 of its own, and 0.41 to 0.57 against 0.27 to 0.31 on the other
  corpora.
- **The two pretrainings are not two repetitions of one training**: one kept its eighth epoch, the
  other its second. The rule's verdict stands, since each alone lies below the mixture by more
  than its threshold (−0.029 and −0.040 at 50 stays, thresholds 0.026), but the mean over the
  two averages two lengths of training.
- **Three weaknesses of this design**, for any successor rather than for this reading: the
  target carries the embedding through the residual stream, so at initialisation it is mostly
  the channel the predictor is handed; the kept epoch is chosen over a baseline that moves with
  the teacher; the collapse check compared with epoch 1, after the targets had narrowed. None is
  changed in the code here: a change to the targets would alter what the stored artefacts
  predict, and a change to the epoch rule needs a run to read it.

**Limitations.**

- 500 windows a corpus and one draw of tokens; the shares are descriptive, without intervals.
- The mixture's three pretrainings differ among themselves (channel share 0.24 to 0.60 on the
  stays); what a value objective's states are made of is not one number either.
- Whether a target no line can answer teaches better features is not settled by this
  objective's one setting; the reading says what this design did, not what the idea does.

## 2026-10-14 — declared before the run: a decoder of four blocks, and the intensive-care corpora alone

**Question.** Read at 500 stays, the point of about 10⁹ values loses most in the encoder's top
blocks and nothing in its first three, and the one backbone above the mixture of four is the
PhysioNet 2019 stays alone, pretrained on half the mixture's steps and a tenth of its tokens
(`layer-readout.md`, 2026-10-14). Does a decoder deep enough to take the reconstruction back from
the encoder's top blocks read the stays better? Do the two intensive-care corpora alone, at the
mixture's number of steps, read them better than the mixture does?

**The variants.** Each is the file of `backbone-mixed4-m` to the step but for one change, at
seeds 1 and 2.

| Variant | Files | The change | Steps | Tokens |
|---|---|---|---|---|
| a decoder of four blocks | `backbone-mixed4-m-decoder4`, `-seed2` | `decoder_layers` 1 → 4 | 22,312 | 617 M |
| the intensive-care corpora | `backbone-clinical-m`, `-seed2` | PhysioNet 2012 and 2019 instead of the four corpora; 14 epochs; warm-up 0.45 of an epoch | 21,952 | 134 M |

The clinical windows are short (169 tokens a 2019 window, 438 a 2012 one, against 750 to 1,050
in the mixture's other corpora), so equal steps hold fewer tokens; tokens could not be matched
short of 65 epochs. Fourteen epochs repeat the data well past the four after which repetition
stops paying (Muennighoff et al., 2023); the kept epoch is the one whose mean relative validation
is lowest, so the mixture's number of steps is a limit. The warm-up matches the mixture's in
steps (about 706 against 697). Counts from the manifests and blocks the runs read
(`data/report/l12/corpus_steps.py`).

**Pilot, before any order** (2026-10-14, this machine, MPS, fp32; the M shape at a tenth of each
corpus, one epoch; registered as `pilot-l12-*`, read by nothing else). All three trained; the
clinical run took 159 steps an epoch, about a tenth of the 1,568 counted. A step of the four-block
decoder cost 1.42 times the one-block's. The encoder's states (`scripts/state_spread_report.py`,
mean of its top four blocks, on 2012's validation windows):

| Pilot | Effective rank of 256 | Channel share | At initialisation |
|---|---|---|---|
| decoder of one block | 2.2 | 0.08 | 27.7, 0.79 |
| decoder of four blocks | 11.0 | 0.38 | 27.7, 0.79 |
| the intensive-care corpora | 2.7 | 0.68 | 27.8, 0.78 |

After one short epoch under a one-block decoder the top blocks fill two directions and lose the
token's channel; under four blocks they keep five times the rank. This is a mechanism after one
epoch over a tenth of each corpus, not a reading of the task.

**Pretraining.** Kaggle T4 ×2 in one session from orders of the committed files: on each GPU the
clinical run (about 0.8 h), then the decoder's (about 3 h, the mixture's 2.1 h times 1.42).

**Diagnostics, before the reading.** `scripts/pretext_triviality_report.py` and
`scripts/state_spread_report.py` under each backbone.

**Reading.** Under each of the four backbones, `campaigns/reader-probes-physionet2012.toml` (the
blocks side by side at 200, 500, 1,000 and every stay of the half, seeds 1 to 10); under the
decoder's two, `campaigns/review-layers-500-physionet2012.toml` too (each block at 500 stays).
Cells on this machine's accelerator from orders. Pairs by `scripts/campaign_pairs_report.py`,
10,000 resamples, 95 %.

- *The rule* (`docs/preregistration.md`, "The pretext", amended before this run). A variant
  **replaces** the mixture's recipe when, at 500 stays, the probe under its two pretrainings
  against the probe under the mixture's four (`reader-probes` under `backbone-mixed4-m` at seeds
  1 to 4), each side read as one, has its paired interval above zero and its mean over seeds
  above twice √(SE² + (s²/2)·(1/2 + 1/4)), s = 0.0057 — about 0.012 to 0.018 for SE 0.005 to
  0.008. Where both replace, the larger mean gain is chosen; the two together are a new variant.
- *Descriptively.* 200, 1,000 and every stay. The pooled s of 0.0079 beside the registered one.
  Each block under the decoder less its initialisation, and less the same block under the
  mixture's two pretrainings read block by block (`layer-readout.md`, 2026-10-14). The clinical
  pair against `backbone-physionet2019-m`. Each probe less its untrained encoder's.
- *Faults, read first.* The decoder changes nothing the untrained encoder is built from, so its
  untrained cells are the mixture's to the bit; the clinical backbone's vocabulary is the 2019
  publication's, so its untrained cells are those under `backbone-physionet2019-m`. Any other
  outcome stops the reading.

**Predictions.** From an earlier interval where one bears on it, its lower edge for a gain;
otherwise marked a judgement.

1. *The decoder at 500 stays*: the mean gain over the mixture lies within −0.01 to +0.02 and does
   not replace it. Basis: under the mixture the top blocks lose little to their initialisation
   (blocks 4, 5 and the last, −0.020 to +0.013 under either pretraining, `layer-readout.md`,
   2026-10-14), so a decoder that only gives the top blocks back has little to recover there; a
   gain past the threshold would need the lower blocks to gain too. A judgement on how much.
2. *The decoder block by block*: blocks 4, 5 and the last at or above their initialisation less
   0.01 under the mean of the two pretrainings. A judgement from the pilot's rank.
3. *The clinical corpora at 500 stays*: the mean gain over the mixture lies within +0.003 to
   +0.038, the interval of the 2019 stays alone against the mixture
   (+0.020 [+0.003; +0.038], one pretraining); whether it replaces is a judgement, and the
   judgement is that it does.
4. *The clinical corpora against the 2019 stays alone*: within ±0.02, a judgement.
5. *1,000 stays and every stay*: neither variant lies below the mixture by more than 0.01. The
   mixture there loses 0.01 to 0.04 to its own untrained encoder (`verdict-statistics.md`,
   2026-10-13); a judgement.
6. *States*: the top blocks under the decoder fill more directions on the stays than the
   mixture's 7.4 to 7.6 of 256 (2026-10-12, above). A judgement from the pilot.

**Limitations.**

- One deeper decoder (four blocks, the encoder's width), not a sweep; MAE's best linear probe
  came from eight narrower blocks.
- The clinical recipe changes corpora and tokens together: at equal steps it reads a fifth of the
  mixture's tokens, and repeats them fourteen times.
- Two pretrainings a variant against four: a gain below about 0.015 at 500 stays is not seen.
- The probe alone decides; fine-tuning is not read.

## 2026-10-15 — Kaggle T4 and M1 Pro, MPS: the decoder of four blocks and the intensive-care corpora, read by the rule

**Question.** The one declared above.

**Conditions.** Pretraining from orders at `a90a0643`, Kaggle T4 ×2, fp16, tier M, one session:
`backbone-clinical-m` (`1fa0ef3a-…`, weights `e2daa1b5…`) and `-seed2` (`7dbd7131-…`,
`1b43da19…`), 14 epochs of about 116 s; `backbone-mixed4-m-decoder4` (`8f5914e9-…`, `9bba238f…`)
and `-seed2` (`220d1136-…`, `07fac23a…`), 8 epochs of about 1,290 s against the mixture's 935 s.
Every run kept its last epoch, every curve fell without a non-finite value. Diagnostics and six
campaigns on the M1 Pro, MPS, fp32 probes, from orders at `a90a0643` (`data/report/l12/`); the
mixture's four `reader-probes` campaigns, its two per-block campaigns and the 2019 stays' concat
campaign are those of 2026-10-13 and 2026-10-14. Pairs by `scripts/campaign_pairs_report.py`,
10,000 resamples, 95 %; a side of several campaigns joined by `+`. A gain is the candidate's area
less the control's.

**Faults, read first.** Every untrained cell under the decoder equals the mixture's to the bit
(20,000 unit errors a budget, 200 to every stay, and each block at 500); every untrained cell
under the clinical backbones equals that under `backbone-physionet2019-m` at 500 and the other
clinical backbone's at every budget. The reading proceeds.

**The diagnostics.** On 2012's validation windows; the triviality columns are the learnt loss's
gain over the trivial baseline of each kind of mask (`scripts/pretext_triviality_report.py`),
the states the mean of the top four blocks (`scripts/state_spread_report.py`):

| Backbone | Channel | Block | Token | Effective rank of 256 | Channel share |
| --- | --- | --- | --- | --- | --- |
| mixture, seed 1 (2026-10-05, 2026-10-12) | +0.019 | +0.085 | +0.023 | 7.8 | 0.26 |
| decoder, seed 1 | +0.007 | +0.070 | +0.008 | **2.4** | 0.05 |
| decoder, seed 2 | +0.023 | +0.085 | +0.022 | **3.0** | 0.13 |
| clinical, seed 1 | +0.036 | +0.098 | +0.042 | 12.5 | 0.58 |
| clinical, seed 2 | +0.042 | +0.104 | +0.048 | 12.3 | 0.63 |

At initialisation every encoder fills 27.7 to 28.4 directions. On the stays every kind is learnt
beyond its baseline under all four; the masks far from any observation are beaten or matched under
all four, as under the mixture. The pretext's relative validation on 2012 is 0.49 and 0.53 under
the decoder, 0.49 to 0.51 under the mixture's four, 0.43 and 0.45 under the clinical corpora.

**The rule.** `frozen_ridge@layer=concat` at 500 stays on the half, each variant's two
pretrainings against the mixture's four, s = 0.0057:

| Candidate side | Gain over the mixture ± SE | Paired 95 % interval | Threshold | Verdict |
| --- | --- | --- | --- | --- |
| the intensive-care corpora | +0.009 ± 0.004 | [−0.002; +0.020] | 0.010 | does not replace |
| the decoder of four blocks | **−0.037** ± 0.005 | [−0.052; −0.022] | 0.012 | does not replace |

**Descriptive.** Area under each pretraining, means over seeds 1 to 10:

| Backbone | 200 | 500 | 1,000 | all (2,000) |
| --- | --- | --- | --- | --- |
| mixture, seeds 1–4 | 0.654–0.673 | 0.692–0.710 | 0.717–0.740 | 0.738–0.755 |
| clinical, seeds 1, 2 | 0.658, 0.677 | 0.693, 0.719 | 0.729, 0.741 | 0.774, 0.755 |
| decoder, seeds 1, 2 | 0.635, 0.639 | 0.654, 0.668 | 0.674, 0.686 | 0.713, 0.737 |
| untrained, the mixture's vocabulary | 0.635 | 0.693 | 0.750 | 0.779 |
| untrained, the clinical vocabulary | 0.633 | 0.681 | 0.749 | 0.779 |

- Over the mixture at 200, 1,000 and every stay: clinical +0.006, +0.011 [−0.002; +0.025] and
  +0.018 [+0.002; +0.034]; decoder −0.025, −0.044 and −0.022, every interval below zero.
- One pretraining against the mixture's four at 500: clinical −0.004 and +0.022 [+0.007;
  +0.037]; decoder −0.044 and −0.030.
- The clinical corpora against the 2019 stays alone (one pretraining, half the steps): −0.011
  [−0.029; +0.006].
- Each probe over its own untrained encoder: clinical +0.034, +0.026, −0.014, −0.014 at the four
  budgets; decoder +0.002, −0.032, −0.070, −0.054. The clinical vocabulary's untrained encoder
  reads 0.012 below the mixture's at 500, so against one reference the clinical gain at 500 is
  about +0.013, not +0.026.
- The decoder block by block at 500, mean of its two pretrainings:

| Block | 0 | 1 | 2 | 3 | 4 | 5 | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| less its initialisation | −0.028 | +0.003 | −0.018 | −0.026 | −0.030 | −0.043 | −0.044 |
| less the mixture's same block | −0.004 | −0.012 | **−0.027** | −0.021 | **−0.040** | **−0.041** | **−0.036** |

Bold: the whole interval below −0.01.

- The spread of pretraining by the registered formula (`data/report/l16/s.py`), from one pair a
  recipe: clinical 0.025 at 500 stays, the decoder 0.009; pooled over both and every budget
  0.011. The clinical pair changes sign with the budget (seed 2 ahead by 0.026 at 500, seed 1 by
  0.019 at every stay), and among the mixture's four one pretraining leads the other three by
  0.015 to 0.017 at 500; one pair is one degree of freedom.

**Conclusions.**

- **A decoder of four blocks does not give the top blocks back; it takes the encoder's states
  away.** It reads the stays worse than the mixture at every budget, under both pretrainings, in
  every block, most from the third up; the states fill 2 to 3 directions and barely carry the
  token's channel. Its reconstruction on the stays is no better than the mixture's. A plausible
  reading, not tested: the deeper decoder answers the masks from the hidden tokens' channel and
  time and from little context, so the encoder is not asked to carry more.
- **The pilot misled.** After one epoch over a tenth of each corpus the four-block decoder kept
  five times the rank; after eight epochs it kept less than the one-block mixture. A short pilot
  predicts nothing of the states after full training.
- **The intensive-care corpora alone are at least as good as the mixture, and not shown better.**
  The mean gain at 500 is +0.009, inside the interval of the 2019 stays alone; at every stay the
  interval lies above zero, descriptively. Their states are the richest read so far, and their
  reconstruction on the stays the best.
- **Two pretrainings a side do not resolve a gain of 0.01.** The spread between two pretrainings
  of one recipe reaches 0.015 to 0.026 here and in the mixture's four; the registered s of 0.0057
  comes from the mixture alone, and the pooled figures of 0.008 to 0.011 put the threshold for
  two against four nearer 0.02.
- **From 1,000 stays up every backbone still loses to its untrained encoder**, the clinical by
  0.014, the decoder by 0.05 to 0.07: the change of corpus moves the reading by about 0.01; it
  does not reverse the sign found on 2026-10-13.

**Predictions.** 1 failed: the decoder lay at −0.037, below −0.01; it does not replace, as
predicted. 2 failed: blocks 4, 5 and the last lie 0.030 to 0.044 below their initialisation.
3 held in its range (+0.009) and failed in its judgement: the corpora do not replace. 4 held:
−0.011. 5 held for the clinical corpora and failed for the decoder (−0.044 at 1,000 stays).
6 failed: 2.4 and 3.0 against 7.4 to 7.6.

**What follows.** The mixture's recipe stays the one later variants are read against. Depth of
the decoder is closed as the cause of the loss in the top blocks. The intensive-care corpora are
the one lead; two more pretrainings would read them four against four.

**Limitations.**

- One deeper decoder: four blocks at the encoder's width, not MAE's eight narrower ones.
- The clinical recipe changes the corpora and the tokens together, and repeats each window 14
  times; how much of its reading is the corpus is not separated.
- Each spread of pretraining here is one pair; the mixture's four give six.
- The probe alone decides; fine-tuning was not read under either variant.

## 2026-10-15 — declared before the run: the intensive-care corpora, four pretrainings against four

**Question.** Read two pretrainings against four, the intensive-care corpora lie +0.009
[−0.002; +0.020] above the mixture at 500 stays, and their two pretrainings differ by 0.026
(above). Read four against four, do they replace the mixture?

**The runs.** `backbone-clinical-m-seed3` and `-seed4`: the file of `backbone-clinical-m` to the
step but for the seed. Kaggle T4 ×2, one run a GPU, about 0.8 h. The same diagnostics as above
under each, then `campaigns/reader-probes-physionet2012.toml` under each on this machine.

**The rule** (`docs/preregistration.md`, "The pretext", amended before this run). The clinical
corpora's four pretrainings against the mixture's four at 500 stays, each side read as one: the
paired interval above zero and the mean over seeds above twice √(SE² + (s²/2)·(1/4 + 1/4)). s is
the larger of 0.011 and the value the clinical four give by the registered formula
(`data/report/l16/s.py`). 0.011 is pooled at 500 stays over the ten pairs of pretrainings of one
recipe read there: the mixture's six, 25.3M's, the objective of states', the clinical pair's and
the decoder's (`data/report/l19/pairs/S/`, `data/report/l12/pairs/S/`). The threshold is about
0.013 to 0.015 for SE 0.003 to 0.005. The mixture's s of 0.0057 rests on its four alone, and two
recipes pretrained twice each gave one pair near 0.025.

- *Descriptively.* 200, 1,000 and every stay; seeds 3 and 4 each against the mixture's four;
  the six pairs of clinical pretrainings.
- *Faults, read first.* The untrained cells under seeds 3 and 4 equal those under seeds 1 and 2 to
  the bit; any other outcome stops the reading.

**Predictions.** From the interval two against four, its lower edge for a gain.

1. *At 500 stays*: the mean gain lies within −0.002 to +0.020 and does not replace the mixture.
   The interval above; that it does not replace is a judgement, the point at +0.009 lying under
   the threshold.
2. *At every stay*: the mean gain lies at or above +0.002, the lower edge two against four.
3. *The states*: seeds 3 and 4 fill 10 to 15 directions of 256 on 2012's stays, as seeds 1 and
   2 (12.3, 12.5); a judgement.

**Limitations.** Four pretrainings bound the clinical side's spread as they bound the mixture's;
a gain below about 0.013 at 500 stays is not seen. The recipe still changes corpora and tokens
together.

## 2026-10-15 — Kaggle T4 and M1 Pro, MPS: the intensive-care corpora four pretrainings against four, read by the rule

**Question.** The one declared above.

**Conditions.** Pretraining from orders at `eb52192c`, Kaggle T4 ×2, fp16, tier M, one session:
`backbone-clinical-m-seed3` (`c9479369-…`, weights `af179566…`) and `-seed4` (`3ff0dfb7-…`,
`31cf75ba…`), 14 epochs of about 105 s. Both kept their last epoch, both curves fell without a
non-finite value; the pretext's relative validation is 0.456 and 0.448 against 0.460 and 0.449
under seeds 1 and 2. Diagnostics and two `reader-probes` campaigns on the M1 Pro, MPS, fp32
probes, from orders at `eb52192c` (`data/report/l12/`, files ending `-34`); seeds 1 and 2 and
the mixture's four campaigns are those read above. Pairs as above; each side of four campaigns
read as one.

**Faults, read first.** Every untrained cell under seeds 3 and 4 equals seed 1's to the bit
(20,000 unit errors a budget, 200 to every stay). The reading proceeds.

**The diagnostics.** As above, on 2012's validation windows:

| Backbone | Channel | Block | Token | Effective rank of 256 | Channel share |
| --- | --- | --- | --- | --- | --- |
| clinical, seed 1 | +0.036 | +0.098 | +0.042 | 12.5 | 0.58 |
| clinical, seed 2 | +0.042 | +0.104 | +0.048 | 12.3 | 0.63 |
| clinical, seed 3 | +0.041 | +0.097 | +0.038 | 12.4 | 0.59 |
| clinical, seed 4 | +0.044 | +0.104 | +0.047 | 11.3 | 0.67 |

Every kind is learnt beyond its baseline under all four; the masks far from any observation are
beaten or matched.

**The rule.** `frozen_ridge@layer=concat` at 500 stays on the half, the clinical four against
the mixture's four. The clinical four's own spread by the registered formula is 0.020, above the
floor of 0.011, so s = 0.020:

| Candidate side | Gain over the mixture ± SE | Paired 95 % interval | Threshold | Verdict |
| --- | --- | --- | --- | --- |
| the intensive-care corpora, four pretrainings | +0.015 ± 0.004 | [+0.005; +0.025] | 0.022 | does not replace |

Under the floor alone the threshold would be 0.013, and the corpora would replace the mixture.

**Descriptive.** Area under each pretraining, means over seeds 1 to 10:

| Backbone | 200 | 500 | 1,000 | all (2,000) |
| --- | --- | --- | --- | --- |
| mixture, seeds 1–4 | 0.654–0.673 | 0.692–0.710 | 0.717–0.740 | 0.738–0.755 |
| clinical, seeds 1–4 | 0.654–0.699 | 0.693–0.729 | 0.729–0.757 | 0.755–0.774 |
| untrained, the mixture's vocabulary | 0.635 | 0.693 | 0.750 | 0.779 |
| untrained, the clinical vocabulary | 0.633 | 0.681 | 0.749 | 0.779 |

- Four against four at 200, 1,000 and every stay: +0.010 [+0.001; +0.020], +0.019 [+0.006;
  +0.032] and +0.021 [+0.006; +0.035]; every interval above zero.
- One pretraining against the mixture's four at 500: seeds 1 to 4 at −0.004, +0.022, +0.009
  [−0.006; +0.024] and +0.032 [+0.016; +0.047]. They rank as the pretext's relative validation
  does, 0.460, 0.456, 0.449 and 0.448 for seeds 1, 3, 2 and 4: a match chance gives once in 24.
- The spread of pretraining over the six clinical pairs:

| Stays | 200 | 500 | 1,000 | all |
| --- | --- | --- | --- | --- |
| s, clinical four | 0.027 | 0.020 | 0.012 | 0.002 |

  At 500 the pairs differ by −0.013 to +0.036. Pooled at 500 over every pair of pretrainings of
  one recipe, now fifteen, s is 0.014.
- The clinical four over their untrained encoder: +0.039, +0.032, −0.006 [−0.019; +0.007] and
  −0.011 [−0.025; +0.003] at the four budgets. The clinical vocabulary's untrained encoder reads
  0.012 below the mixture's at 500; at 1,000 and every stay the two are equal.

**Conclusions.**

- **The intensive-care corpora read the stays above the mixture at every budget, and do not
  replace it.** Four against four the gain is +0.010 to +0.021 with every interval above zero;
  at 500 the mean of +0.015 lies under the threshold their own spread sets.
- **Their recipe is unsteady across pretrainings.** s is 0.020 at 500, 3.5 times the mixture's
  0.0057; one pretraining lies under the mixture and another 0.032 above it. Resolving +0.015
  under this spread would take about a dozen pretrainings a side.
- **The spread falls as labels grow**, from 0.027 at 200 stays to 0.002 at every stay: the
  pretraining changes what a few hundred labels can read, much less what all of them reach.
- **At 1,000 stays and above they still gain nothing over their untrained encoder.** Read over
  four pretrainings the loss of 0.014 seen over two shrinks to −0.006 and −0.011, both intervals
  across zero; the sign found on 2026-10-13 is not reversed.

**Predictions.** 1 held: +0.015, inside −0.002 to +0.020; it does not replace. 2 held: +0.021
at every stay. 3 held: 12.4 and 11.3 directions.

**What follows.** The mixture's recipe stays the one later variants are read against. The
intensive-care corpora are not pursued further by more seeds; the work turns to an objective
that does not reconstruct values. The registered s of 0.0057 describes the mixture's four alone;
a new recipe brings its own spread, here 0.020.

**Limitations.**

- The recipe still changes the corpora and the tokens together.
- s from four pretrainings is itself uncertain: six pairs that share their members.
- At every stay the seeds share their labels, so the error over seeds is zero there and the
  interval carries all the noise.
- The match between the reading and the relative validation is four points, not a selection
  rule.
- The probe alone decides; fine-tuning was not read.

## 2026-10-20 — declared before the run: the contrast between the two halves of a window

**Question.** Does a pretext that, beside reconstructing the hidden values, asks the encoder to
tell the two halves of a window apart from other windows' halves (ADR-0058) give fine-tuning at
200 stays more than the mixture's masks alone, and does it leave the reader of the stage at 500
stays no worse? A contrast's gains in the clinic live at few labels under fine-tuning (Chowdhury et al.
2023: +0.03 to +0.07 at one to four stays, +0.013 at full labels), below what the probe at 500
resolves, so the probe screens for harm and fine-tuning at 200 is the reading.

**The variant** (`experiments/backbone-mixed4-contrast-m.toml`, `-seed2.toml`). The file of
`backbone-mixed4-m` to the step — the four corpora, 4.8M parameters, the mixture's masks, huber
at one deviation, eight epochs, 16 × 2 windows a step, 1e-3 — with one section changed:
`[objective] contrast_weight = 1.0`, `contrast_temperature = 0.1`. The masks cut every window
into the tokens the encoder sees and the tokens hidden from it; each half goes through the
encoder with the other as padding, is pooled to one point and carried through two linear maps
that are thrown away, and must pick the other half of its window out of both halves of the
other fifteen windows of the micro-batch (NT-Xent, 31 candidates an anchor, cosine at the
temperature). The term weighs one anchor's loss against one hidden token's; the kept epoch is
the lowest combined loss over trivial. Seed 1 draws the encoder and the decoder the mixture's
seed 1 drew.

**Pilot, before any order** (2026-10-19/20, this machine, MPS, fp32; the M shape at a tenth of
each corpus, one epoch; registered as `pilot-l24-*`, read by nothing else). Five runs: the file
without the term and four pairs of knobs, weights 0.5, 1 and 2 at temperature 0.2 and weight 1
at temperature 0.1. The rule choosing the pair was written before the first run
(`data/report/l24/pilot/RULE.md`): weight 1 at 0.2 unless a setting beat the file without the
term by more than its paired SE under the reader of the stage, beat that default by more than
the paired SE of the difference, and kept its reconstruction within 10 % of the plain run's on
every corpus. The file without the term repeats the pilot of 2026-10-14 to five decimals for
200 steps and parts at step 250, the accelerator's accumulated precision: the refactoring behind
the term changed nothing in the run without it.

| Setting | Reconstruction vs plain (cmapss / skab / esa / 2012) | Contrast over chance (cmapss / skab / esa / 2012) | Effective rank, last block | Probe at 500, 1 − AUROC | vs plain |
|---|---|---|---|---|---|
| plain | — | — | 1.3 / 1.7 / 1.4 / 1.4 | 0.284 (+0.023 over untrained) | — |
| 0.5, 0.2 | −5 / −22 / −2 / −1 % | 0.87 / 1.00 / 1.02 / 0.84 | 2.5 / 1.4 / 1.9 / 2.3 | 0.328 | −0.044 ± 0.005 |
| 1, 0.2 | −5 / −16 / −3 / −2 % | 0.86 / 1.02 / 1.01 / 0.86 | 2.4 / 1.4 / 1.7 / 2.3 | 0.296 | −0.012 ± 0.006 |
| 2, 0.2 | −9 / −20 / −2 / −4 % | 0.87 / 1.05 / 1.01 / 0.79 | 2.1 / 1.4 / 1.5 / 2.0 | 0.307 | −0.023 ± 0.012 |
| **1, 0.1** | −5 / −14 / −3 / +1 % | 0.90 / 1.02 / 0.85 / 0.93 | 2.3 / 1.4 / 2.2 / 3.4 | 0.279 (+0.028 over untrained) | +0.005 ± 0.005 |

Validation per corpus, the kept epoch; the probe is `frozen_ridge@layer=concat` at 500 stays on
the half held out by seed 101, seeds 1 to 10, paired by seed
(`data/report/l24/pilot/{pretext.csv,pairs/table.md,spread-*/}`). The term trains in one epoch at
this scale on C-MAPSS and the stays, not on SKAB, and on the satellite corpus only at 0.1; every
setting lowers the reconstruction's validation loss on every corpus, so the term costs the first
objective nothing here. An epoch costs 2.1 times the plain run's. The rule chose weight 1 at
temperature 0.1: it beats plain by 0.0051 against a paired SE of 0.0048, a hair, and the default
by +0.017 [+0.001; +0.033]; every setting at 0.2 reads below plain.

**Pretraining.** Two seeds, 1 and 2, on Kaggle T4 ×2, one per GPU, from the orders of the
committed experiment files; the first steps of seed 1 on the T4 are the trial of the family
(memory and pace) before the second order is placed. About 4 h each: the second pass of the
encoder doubles the encoder's share of a step over the mixture's 2.1 h.

**Diagnostics, before the reading.** The run's log per epoch (each term per token against its
chance, per validation corpus); `scripts/state_spread_report.py --backbone <id>` and
`scripts/pretext_triviality_report.py --backbone <id>` under the mixture's masks, as every value
objective is read.

- *Not trained.* A backbone whose contrast at the kept epoch lies above 0.9 of chance on every
  corpus has not trained the term and is read with that said.

**Reading.** Every cell on the half of the tuning side held out by seed 101 (`one_in = 2`),
seeds 1 to 10, the same labels per seed, so every difference is paired by seed; a side of
several pretrainings is read as the mean over them (`scripts/campaign_pairs_report.py`, sides
joined by `+`).

- *Screen, the reader of the stage.* `review-concat-500-physionet2012.toml` under both
  backbones: `frozen_ridge@layer=concat` at 500 against the mixture's four (`backbone-mixed4-m`
  seeds 1 to 4, already read). Δ₅₀₀ = mean over seeds of (the two − the four); threshold
  T₅₀₀ = 2·√(SE² + (s²/2)(1/2 + 1/4)), s = max(0.014, the two pretrainings' own). The objective
  is **closed for harm** where Δ₅₀₀ < −T₅₀₀.
- *Primary.* `reader-200-physionet2012.toml` under both backbones: `full_fine_tuning` at 200
  under the settings in force, against the same cells under the mixture's four — seeds 1 and 2
  read on 2026-10-13 and 2026-10-16 (0.682 and 0.688 in area), seeds 3 and 4 ordered with this
  declaration. Δ₂₀₀ = mean over seeds of (the two −
  the four); threshold T₂₀₀ = 2·√(SE² + (s_FT²/2)(1/2 + 1/4)), s_FT = max(0.014, the spread
  between pretrainings under fine-tuning at 200 on the half by the registered formula, pooled
  over the mixture's six pairs and the candidate's one). With SE about 0.012 the threshold is
  about 0.03; a gain below it is not seen.
- *Descriptive.* `reader-low-physionet2012.toml`, a new file: `full_fine_tuning` and the network
  from nothing at 20 and 50 stays on the half under both backbones and the mixture's seeds 1
  and 2; read as fine-tuning less the network from nothing, the two against the two. Resolution
  0.05: one seed's area moves by 0.03 to 0.04 there (`verdict-statistics.md`, 2026-09-29). The
  validation side is not read.
- *What follows.* The screen passed and Δ₂₀₀ above zero → pretrained to four and read by the
  replacement rule at 200 and at 500, as a screened recipe is; Δ₂₀₀ within ±T₂₀₀ with the screen
  passed → closed as "no gain the instrument sees"; the screen failed → closed for harm. One
  confirmatory comparison, no correction.

**Predictions** (from intervals; a judgement where none bears).

1. Screen: Δ₅₀₀ within [−0.02, +0.02]. The two variants of the masks lay at −0.017 and −0.015
   against seed 1 (2026-10-06), the objective of states at −0.03 (2026-10-11), and at tier S the
   chosen setting reads +0.005 ± 0.005 against the plain run.
2. Primary: Δ₂₀₀ within [−0.02, +0.03]. Fine-tuning at 200 gains +0.012 and +0.028 over the
   network from nothing under the mixture's two seeds (`fine-tuning-recipe.md`, 2026-10-16), and
   the recipe's variants moved fine-tuning at 200 by at most 0.025 (`intensive-care-curve.md`,
   2026-10-07); a gain past 0.03 would be the first. A judgement from those intervals.
3. Few-shot: fine-tuning less the network from nothing at 20 and 50 under the objective lies
   within ±0.05 of the same under the mixture's two. A judgement from one seed's spread there.
4. The term is trained: the contrast at the kept epoch lies below 0.9 of chance on three of the
   four corpora under both seeds (the pilot: three of four at 0.1 after a tenth of an epoch of
   the recipe), and the reconstruction's relative validation lies within 5 % of the mixture's
   per corpus (the pilot: below it everywhere).
5. The two backbones' probes at 500 differ by less than 0.03 (s pooled 0.014, two draws).

**Limitations.**

- One pair of knobs, chosen at tier S by a rule whose first clause held by a hair; the
  ordering of settings at a tenth of the data for one epoch may not hold at tier M.
- 31 candidates an anchor, the micro-batch; a queue of past halves was not built.
- Two pretrainings against four, so a gain below about 0.03 at 200 is not seen.
- The kept epoch is chosen by the combined loss, a second difference beside the gradient.
- Seeds 3 and 4 of the mixture enter fine-tuning at 200 on the half with this declaration, so
  s_FT is pooled over pairs read at two times; it is named at the reading.
