# Findings

What the project has measured, on one page. Every number is on the **validation side** and
preliminary: each task's frozen test side is opened once, at the end. Each comparison was judged
by a rule registered before its run ([preregistration](preregistration.md)); each row links the
note that holds its evidence ([`verification/`](verification/README.md)). A row states direction
and size; conditions and caveats are in its note.

Last updated 2026-11-09.

## Status

- **The claim is not confirmed; a screen of four pretrainings gains 0.016 at 50 stays.** Full
  fine-tuning of a small encoder over cells, pretrained on intensive-care stays, the eICU demo and
  raw satellite telemetry, gains +0.016 [+0.007; +0.025] at 50 stays over the network from scratch
  at its chosen rate, the same at 20 within the noise threshold, and +0.004 [−0.007; +0.015] at
  200. At the lower edge the registered confirmation has power below 0.25
  ([note](verification/composition-screen.md)).
- **Pretraining over cells leaves a representation above its initialisation.** A frozen probe
  gains 0.060 to 0.091 over the same encoder untrained at 500 stays, and still reads about 0.03
  below the trees ([note](verification/cell-pretraining.md)).
- **Classical models lead at few labels.** Trees over per-channel summaries match or beat every
  network from 200 labels up on both tasks, except that a small encoder from scratch over cells
  meets them at 200 stays (+0.003 [−0.011; +0.017]), leads by 0.028 at 50, and reads 0.008 below
  the tuned trees on set B, not shown no worse by 0.015
  ([note](verification/encoder-candidates.md)).
- **On satellite telemetry pretraining adds nothing without labels.** No distance over the states
  shows an advantage of pretraining on either mission, and the pretext's error trails a ridge. A few
  taught events lift either encoder alike and beat trees taught the same events by 0.10. Without a
  label the best detectors are classical ([note](verification/telemetry-readout.md)). Fine-tuning on
  a few events shows no large gain over its initialisation. With all 99 events the frozen direction
  reaches about 0.26, with or without pretraining ([note](verification/taught-fine-tuning.md)), and
  the same network trained on all 99 events, at the length chosen on later fitting months (500
  steps, the shortest), reaches the same 0.26 (−0.001 [−0.044; +0.040] against the direction at
  its initialisation) and misses the same events: no room above 0.26 shows for either readout ([note](verification/telemetry-supervised-reference.md)).
  A longer input does not lift it either: over windows of 6 to 72 hours ending at the scored
  hour no layout is chosen, and read beside every context arm over the untrained states falls
  below the hour (−0.130 [−0.195; −0.066] under the direction), since the hours after an event
  inherit its trail and the mean over cells dilutes the hour ([note](verification/telemetry-context.md)).
  Each channel against its own recent hours does carry the missed events: with no label the
  largest departure over the channels reads 0.18, level with the best bar and twice the
  backbone's zero-label reading, and finds as many events at 1 % false alarms as the taught
  direction; taught linearly the departures add nothing to the state, taught by trees they find
  50 to 54 of 61 events at 1 % against 30 to 40, at a lower area. On the second mission the
  same zero-label detector reads 0.885 against the best bar's 0.627 and the taught direction's
  0.900 ([note](verification/telemetry-channel-local.md)). The detector is now part of the
  package, a bar the backbone is held to, read by area and by the events found at 1 % false
  alarms, the product's measure. A pretext that forecasts each channel's hour from the two days before it, pretrained twice
  on the satellites' first halves, forecasts 17 times better than the history's median on the first mission, yet its
  error reads the events no better than the departures with no model (−0.018 [−0.074; +0.039])
  and not demonstrably better than the last hour carried forward (+0.036 [−0.033; +0.112]);
  every detector referenced to its own history ranks an event's first hour high and its later
  hours at chance, so the event-weighted area reads the tail: an onset-only oracle would read
  0.30 here, these read 0.12 to 0.21
  ([note](verification/telemetry-history-pretext.md)).
- **Ruled out as levers:** more pretraining data, a five times larger model, continuing on the
  task's own domain, the adaptation recipe, and six variants of the pretext.
- **What transfers is the encoder's machinery.** Fitting only the channel rows of an unread corpus
  brings the pretext to within 0.006 of a backbone that read it, and carries part of a task
  (+0.031 on sepsis at 4,000 patients) ([note](verification/task-transfer.md)).

## Setup

| | |
| --- | --- |
| Tasks | Remaining life on C-MAPSS FD001 (RMSE; 79 tuning, 21 validation engines); death in hospital on PhysioNet 2012 (AUROC, first 48 h); sepsis on PhysioNet 2019; events in telemetry (event-weighted PR-AUC) |
| Backbone | Set encoder over (channel, time, value) tokens or cells, 4.8M parameters (25.3M for scale; small shape: 2 blocks, 64 wide), pretrained by masked reconstruction ([ADR-0017](adr/0017-encoder-architecture.md)) |
| Pretraining corpora | Mixture of four ([ADR-0029](adr/0029-pretraining-over-a-mixture-of-corpora.md)); up to about 10⁹ values for scale; the stays are 4.5 % of the mixture's windows ([note](verification/pretraining-corpus-sampling.md)) |
| Arms | From scratch (control), frozen probe, closed-form probe, LoRA, full fine-tuning; trees, MiniRocket and a patch model as classical baselines |
| Statistics | Paired bootstrap over engines or stays, seeds paired; a backbone comparison counts past 2·√(SE² + (s²/2)(1/n₁ + 1/n₂)), s the spread between pretrainings ([ADR-0032](adr/0032-statistics-of-a-paired-comparison.md)) |
| Instrument | Closed-form probe over all blocks at 500 stays, half the tuning side, ten seeds; resolves 0.02 to 0.03 in area, not 0.01 ([note](verification/verdict-statistics.md)) |

## Results

### Pretraining against training from scratch

FD001, RMSE, lower is better; one paired campaign, five seeds, every arm tuned per budget
([note](verification/label-efficiency-curve.md), [classical](verification/classical-baselines.md)).
The registered endpoint, full fine-tuning at 200 windows, is 10.5 % worse than from scratch
([−2.56, −0.54] RMSE).

| Labelled windows | From scratch | Frozen probe | Full fine-tuning | Trees | MiniRocket |
| --- | --- | --- | --- | --- | --- |
| 50 | **16.34** | 16.93 | 17.61 | 19.35 | 17.17 |
| 200 | 14.29 | 14.66 | 15.77 | **14.08** | 15.30 |
| 1,000 | 12.77 | 14.53 | 13.40 | **12.04** | 15.40 |
| all (2,568) | **11.47** | 14.18 | 12.71 | 11.76 | 15.42 |

PhysioNet 2012, AUROC, higher is better; half the tuning side, ten seeds. The 4.8M encoder over
raw tokens ([note](verification/verdict-statistics.md),
[classical](verification/gap-to-the-trees.md)):

| Labelled stays | 200 | 500 | 1,000 | all (2,000) |
| --- | --- | --- | --- | --- |
| Network from scratch | 0.65 | 0.75 | 0.79 | 0.79 |
| Small network from scratch, 6-bin cells ([note](verification/encoder-candidates.md)) | 0.73 | 0.78 | | 0.83 |
| Full fine-tuning, mixture of four | 0.69 | 0.74 | 0.76 | 0.79 |
| Probe, mixture of four | 0.65–0.67 | 0.69–0.71 | 0.72–0.74 | 0.74–0.76 |
| Probe, untrained encoder | 0.64 | 0.69 | 0.75 | 0.78 |
| Trees per channel | **0.74** | **0.79** | **0.82** | **0.84** |
| MiniRocket | 0.71 | 0.76 | 0.78 | 0.81 |
| Logistic head over the summaries | 0.73 | 0.77 | | 0.81 |
| Network over the summaries (diagnostic) | 0.71 | 0.75 | 0.79 | 0.81 |

Under the mixture, fine-tuning's gain at 200 stays, +0.024 [+0.014; +0.033], does not replicate on
two unread pretrainings (+0.014 against 0.021) ([note](verification/few-label-fine-tuning.md)). On
set B the small encoder reads 0.850 against the tuned trees' 0.858 (−0.008 [−0.017; +0.001]); the
network 0.81 ([note](verification/encoder-candidates.md),
[full grid](verification/intensive-care-curve.md)).

The small encoder over cells, pretrained without labels, each arm at its chosen setting:

| Labelled stays | 20 | 50 | 200 | 500 |
| --- | --- | --- | --- | --- |
| Fine-tuning, stays only ([note](verification/cell-pretraining.md)) | 0.62 | 0.66 | 0.74 | |
| Fine-tuning, stays + eICU demo + satellites ([note](verification/composition-screen.md)) | 0.63 | 0.67 | 0.74 | |
| Network from scratch, its chosen rate | 0.62 | 0.65 | 0.74 | |
| Probe, pretrained / untrained (composition) | | | | 0.76 / 0.67 |
| Trees per channel | 0.62 | 0.63 | 0.74 | **0.79** |

### Is the pipeline sound?

| Check | Result | Source |
| --- | --- | --- |
| Generated data with known shared structure | pretrained error 0.094 [0.089; 0.099] below a fresh encoder | [synthetic-transfer](verification/synthetic-transfer.md) |
| Published network (STraTS) on the same stays | 0.827; ours from scratch 0.818 with its best epoch and two blocks | [intensive-care-curve](verification/intensive-care-curve.md) |
| The network from scratch in two campaigns | differs by at most 0.005 | [verdict-statistics](verification/verdict-statistics.md) |
| Gain on stays the backbone read, less on those it did not | −0.007 / −0.003 / −0.010 at 20 / 50 / 200: no gain from having read them | [cell-pretraining](verification/cell-pretraining.md) |
| Pretext over cells answered without a model | stays: mean 0.12, presence 0.59, under a tail 0.22; on grids interpolated at 0.53 to 0.80 | [cell-pretext](verification/cell-pretext.md) |
| Added data measured as the stays are | eICU demo matches the 2012 stays; the Water Quality Portal lies nearer a grid | [task-regime-sources](verification/task-regime-sources.md) |

### What changes the backbone

Each recipe by the instrument at 500 stays against the mixture of four. Bold passes the threshold
in force, s = max(0.014, the recipe's own): about 0.024 for one pretraining, 0.021 for two.

| Recipe | Pretrainings | Gain in area | Source |
| --- | --- | --- | --- |
| Scale, about 10⁸ and 3·10⁸ values | 1 each | −0.018, −0.021 | [layer-readout](verification/layer-readout.md) |
| Scale, about 10⁹ values | 1 | 0.02 to 0.05 below every mixture | [verdict-statistics](verification/verdict-statistics.md) |
| 25.3M parameters at about 10⁹ | 2 + 1 | −0.021 / +0.004; four epochs **−0.032** | [layer-readout](verification/layer-readout.md) |
| Intensive-care corpora alone | 4 | +0.015 [+0.005; +0.025], threshold 0.022 | [pretext-variants](verification/pretext-variants.md) |
| PhysioNet 2019 alone | 1 | +0.020 ± 0.005 | [layer-readout](verification/layer-readout.md) |
| Stays four times as often | 1 | +0.012 ± 0.003 | [layer-readout](verification/layer-readout.md) |
| Masks over three quarters of a window | 1 | +0.008 ± 0.005 | [layer-readout](verification/layer-readout.md) |
| Masks over the forecast tail | 1 | −0.019 ± 0.005 | [layer-readout](verification/layer-readout.md) |
| Water Quality Portal added | 2 | −0.014 on the probe, threshold 0.020 | [composition-screen](verification/composition-screen.md) |
| Decoder of four blocks | 2 | **−0.037** | [pretext-variants](verification/pretext-variants.md) |
| Predicting a teacher's states | 2 | **−0.055** | [pretext-variants](verification/pretext-variants.md) |
| Contrast between a window's halves | 2 | **−0.103** [−0.125; −0.081] | [pretext-variants](verification/pretext-variants.md) |
| Hidden part's summaries beside reconstruction | 2 | not read: term learnt below a linear answer (0.57 against 0.63) | [pretext-variants](verification/pretext-variants.md) |
| 10⁹ continued on the task's domain, rate 1e-4 / 1e-3 | 2 / 1 | **−0.025** / −0.002 | [domain-continuation](verification/domain-continuation.md) |

### How the backbone is used

| Change | Result | Source |
| --- | --- | --- |
| All blocks side by side, not the last | +0.020 to +0.036 at 50 stays; registered as the reader | [layer-readout](verification/layer-readout.md) |
| Encoder slowed during fine-tuning (LP-FT) | loses up to 0.11; at the full rate within 0.01 | [fine-tuning-recipe](verification/fine-tuning-recipe.md) |
| Mean per channel, not over the stay | +0.037 / +0.038 at 200 / 500; same at initialisation; diagnostic | [head-and-representation](verification/head-and-representation.md) |
| Mean over the channels a stay measured | −0.058 / −0.052 behind the per-channel mean | [head-and-representation](verification/head-and-representation.md) |
| Network kept at its best epoch | +0.027 at every stay | [intensive-care-curve](verification/intensive-care-curve.md) |

### Transfer to a corpus the backbone never read

| Reading | Result | Source |
| --- | --- | --- |
| Pretext, unseen channels as drawn | loses to interpolation and regression by 0.12 to 0.21 | [zero-shot-pretext](verification/zero-shot-pretext.md) |
| Pretext, channel rows fitted without labels | within 0.006 of the reading backbone (2012, C-MAPSS); 0.02 behind on 2019 | [zero-shot-pretext](verification/zero-shot-pretext.md) |
| Pretext, rows copied from the same quantity | within 0.01 to 0.02 of the fitted rows | [zero-shot-pretext](verification/zero-shot-pretext.md) |
| Sepsis at 4,000 patients, rows fitted | +0.031 [+0.015; +0.047] over drawn rows; none at 500 stays on 2012 | [task-transfer](verification/task-transfer.md) |

### Events in telemetry

Frozen encoder against its initialisation; M1, M2 = the two satellite missions.

| Reading | Result | Source |
| --- | --- | --- |
| Nearest neighbours, unread sensors | M1 +0.033 [+0.005; +0.090], below the least gain 0.05; OPS-SAT **−0.069** [−0.110; −0.026]; M2, SKAB indistinguishable | [anomaly-tasks](verification/anomaly-tasks.md) |
| Channels' summaries against the untrained encoder | SKAB **+0.188** [+0.057; +0.340]; OPS-SAT −0.086 [−0.160; −0.002] | [anomaly-tasks](verification/anomaly-tasks.md) |
| Each month as its own reference | alarm on 1–4 % of normal windows, from 64–98 %; M1 +0.007 [−0.011; +0.037] | [anomaly-tasks](verification/anomaly-tasks.md) |
| Counts of readings alone | near chance on M1, M2; OPS-SAT 0.387, below the plain count 0.51 | [anomaly-tasks](verification/anomaly-tasks.md) |
| States of pretraining on the satellites' first halves | M1 +0.018 [−0.008; +0.045]; 87 % of M1's events hold no reading beyond ten deviations | [composition-screen](verification/composition-screen.md) |
| Raised windows: is a reading's size kept? | kept, 0.85 to 1.41 times the initialisations' at ×3; normal windows lie 2.3 to 3 times as far apart | [magnitude-in-states](verification/magnitude-in-states.md) |
| Other distances over the states | no advantage of pretraining by the declared rule | [anomaly-signal-in-states](verification/anomaly-signal-in-states.md) |
| Whitened Gaussian against each month | M1 +0.009 [−0.009; +0.026]; M2 −0.41 below initialisation | [telemetry-readout](verification/telemetry-readout.md) |
| Pretext's error per channel, less its ridge | **−0.121** [−0.194; −0.058]; raw −0.026 | [telemetry-readout](verification/telemetry-readout.md) |
| Taught 3, 5 or 10 events | +0.01 to +0.05 under either encoder, interaction −0.017 to +0.010; over taught trees +0.10 to +0.12 | [telemetry-readout](verification/telemetry-readout.md) |
| Best zero-label bar on M1 | the cells' farthest neighbour 0.154 against the states' 0.090; the bar's number rests on a near-constant feature's spread and read 0.017 refitted with 39 windows fewer, the same within every month; under one alignment of the months both read 0.21, the states 0.18; recomputed under the floored standardisation in the channel-local note | [telemetry-readout](verification/telemetry-readout.md) |
| Fine-tuning on 5, 10 events, 2,000 steps, against its initialisation | +0.045 [+0.010; +0.089], −0.024 [−0.043; +0.015]; neither a large gain nor harm | [taught-fine-tuning](verification/taught-fine-tuning.md) |
| Fine-tuning at 200 steps against 2,000, every encoder | 5 events +0.035 [+0.014; +0.082]; 10 events +0.105 [+0.064; +0.154]; mechanism not shown, length chosen on the months read | [taught-fine-tuning](verification/taught-fine-tuning.md) |
| All 99 events taught, M1 | frozen direction: pretrained 0.261, 0.217, initialisation 0.266; fine-tuning at 2,000 steps 0.093 to 0.096; trees 0.154 | [taught-fine-tuning](verification/taught-fine-tuning.md) |
| Trained on all 99 events, length chosen on later fitting months, against the direction at its initialisation | 0.265 against 0.266, −0.001 [−0.044; +0.040]; 500 steps chosen, longer runs read worse; 11 of 62 events found by no arm; pretrained 0.152, 0.219; the direction finds the six unseen classes (0.57 to 0.84), the trained network 0.27 to 0.34 | [telemetry-supervised-reference](verification/telemetry-supervised-reference.md) |
| Each channel's hour against its own recent hours (6 to 72), untrained encoder's states beside | no label, largest departure over channels: 0.179 against the cells' bar 0.160, +0.019 [−0.093; +0.086], recall 0.61 at 1 %; taught, departures beside the state **−0.079** [−0.138; −0.036], as order statistics −0.001 [−0.011; +0.003]; trees over the departures recall **0.84 to 0.88** at 1 % (direction 0.60), area 0.17 to 0.19, 7 of the 11 unfound events found | [telemetry-channel-local](verification/telemetry-channel-local.md) |
| The same on the second mission, as a confirmation | no label, largest departure **0.885** against the bar 0.627, **+0.258** [+0.199; +0.322], against the taught direction 0.900 and the whitened states 0.687; taught, departures beside the state **−0.052** [−0.079; −0.028], order statistics +0.001 [+0.001; +0.002]; trees +0.006 [−0.017; +0.026], every taught arm finds 228 of 229 events at 1 % | [telemetry-channel-local](verification/telemetry-channel-local.md) |
| The same detector in the package, a campaign cell against the report's answers | every month ranked alike, least Spearman 1.00000 (M2) and 0.99998 (M1), top 1 % held; M2 the same to the bit; area and recall at 1 % equal at four decimals (M1 0.179, 0.607; M2 0.885, 0.996) | [telemetry-channel-local](verification/telemetry-channel-local.md) |
| Windows of 6, 24, 72 hours ending at the scored hour, untrained encoder, layout chosen on later fitting months, against the hour | no layout chosen; read beside, 6x6 under the direction **−0.130** [−0.195; −0.066], under the whitened distance **−0.059** [−0.136; −0.008]; smoothing the hour's scores −0.18 to −0.23; pooling the previous hours' states −0.006 [−0.037; +0.021]; trees over the cells with 24 hours +0.023 [−0.037; +0.060]; none of the 11 unfound events found, which are local (10 of 11), touch 3 channels of 48 (median) and last 5.5 hours against 12 channels and 15 hours for the found | [telemetry-context](verification/telemetry-context.md) |
| The pretext that forecasts each channel's hour from the 47 before it, two pretrainings, its error read by the largest channel | against `channel_local` −0.018 [−0.074; +0.039], against the last hour carried forward +0.036 [−0.033; +0.112]: around or below; recall at 1 % 0.59 against the bar's 0.60, the raw error's 0.67; every history-referenced detector ranks an event's later hours at chance, so the area reads the tail (an onset-only oracle 0.30, these 0.12–0.21); M2 0.889 against the bar 0.886 and the carried hour 0.928 | [telemetry-history-pretext](verification/telemetry-history-pretext.md) |

### What the failures have in common

- **A better pretext loss is not a better backbone.** Thirteen passes over the mixture lower it and
  lose 0.057 under fine-tuning to eight passes.
- **The loss sits in the top blocks.** Under about 10⁹ values the early blocks read like an
  untrained encoder and the deficit grows to the last.
- **The probe's average loses what the task reads, not the tokens.** A linear map recovers a median
  86 % of a channel's level from its own tokens, 15 % from the mean over the stay (31 % before
  training); pooling per channel adds 0.04, and pretraining has no share in it
  ([note](verification/head-and-representation.md)).
- **Pretexts are solved by cues that are not the task:** interpolation under heavy masks, the clock
  of measurements under contrast, a linear answer for hidden summaries.
- **Pretraining reads little that is sampled like the tasks.** Only the stays read channels at
  their own times: 1 to 3 % of the values along the scale curve
  ([note](verification/pretraining-corpus-sampling.md)).
- **Below 500 labels the network's gap is representation; above it, the trees' table.** From 500
  stays both the network and a linear head over the summaries stop 0.03 below the trees
  ([note](verification/gap-to-the-trees.md)).

## Limitations

- **Validation only.** Each configuration was chosen on the side it is reported on; the single test
  run removes that bias.
- **Resolution.** At 500 stays the instrument resolves 0.02 to 0.03; published gains of
  self-supervision in the clinic at 10 % of labels or more are 0.004 to 0.013, below it.
- **Few independent units.** FD001 has 21 validation engines; its 95 % interval covers about 92 %
  on known answers.
- **Two main tasks.** The intensive-care task learns from set A only, so its levels are not
  comparable with published benchmarks.
- **Controls' rates.** The network from scratch and fine-tuning chose the lowest rate on their grid;
  a lower one was not read. Between two runs the network moved by 0.011 at 200 stays.
- **Accelerators.** Arms ran on Colab G4, A100 and Kaggle T4; between them an arm drifts by up to
  0.7 RMSE.

## Next

1. The ceiling without labels on the first mission, before any further pretext: which validation
   events anything finds with no label and which only the labels' recurring classes reach; the
   events found at 1 % false alarms registered as the telemetry measure, since every detector
   referenced to its own history ranks an event's later hours at chance; the missed local events
   read finer than the hour. The forecasting pretext over a channel's history is closed as a
   detector ([note](verification/telemetry-history-pretext.md)); a perturbation term waits on
   the ceiling.
2. Read channels added without labels on sepsis, across pretrainings that never read 2019.
3. Confirm the gain at 50 stays on fresh pretrainings, in a design with power at the screen's
   lower edge, and read there whether the order of pretrainings holds across tasks and detectors.
4. Open the frozen test side once.
