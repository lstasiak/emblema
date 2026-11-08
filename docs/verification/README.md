# Verification notes — lab notebook

The archive of evidence behind the decisions in [`docs/adr/`](../adr/README.md) and the results in
[`docs/findings.md`](../findings.md). Each note records measurements that CI cannot make (MPS, a
GPU platform, the remote bucket, the full corpora), with the commit, machine, tier and command.

Notes are append-only: a new run is a new dated section, and a superseded section is marked with
one sentence, never rewritten. The preregistration cites notes by file, date and title.

A section has this shape: question → conditions → tables and figures → conclusions → limitations.
It is at most about 1,500 words, not counting table bars. Sections written before 2026-09-24
predate this format. Those over the limit were condensed on 2026-11-03 with titles, tables and
numbers kept, each pointing to its full text at commit `85858e92`. Three stay over it because their
measurements would not fit: the satellite backbones in `corpus-saturation.md`, the Mac measurements
in `data-spike.md` and the first grid in `label-efficiency-curve.md`.

## Infrastructure

| Note | Conclusion |
| --- | --- |
| [wheels.md](wheels.md) | torch and onnxruntime wheels exist for cp312 and cp314 on every target; Kaggle and Colab ship Python 3.12. |
| [architecture-rules.md](architecture-rules.md) | Every import-linter contract fails on an injected violation, and the coverage gate honours its threshold. |
| [compose.md](compose.md) | The local stack comes up with one command on linux/arm64 and passes its smoke test and S3 contract against Garage. |
| [remote-bucket.md](remote-bucket.md) | The S3 contract and lifecycle bootstrap pass against Cloudflare R2 by configuration alone. |
| [onnx-export.md](onnx-export.md) | The set encoder exports to ONNX with one dynamic token axis at opset 20, and so does the fitted candidate under every transfer mode, answering within 1.9e-05 of eager. |
| [published-corpus.md](published-corpus.md) | A published corpus round-trips through the remote bucket byte for byte, and its checksums repeat across machines. |
| [loader-throughput.md](loader-throughput.md) | Batching costs a negligible share of a training step on MPS; windows as Python objects are too large to hold a corpus. |
| [training-loop.md](training-loop.md) | A run resumed mid-epoch ends in the uninterrupted run's weights: bit for bit on the host, within repeat spread on MPS. |
| [manual-handoff.md](manual-handoff.md) | Runs ordered here, made on a GPU platform and accepted back keep full provenance; on a G4, runs sharing the GPU under MPS reached about 1.7 times the throughput of the same runs one after another. |

## Data

| Note | Conclusion |
| --- | --- |
| [data-spike.md](data-spike.md) | Licences, units, windows and tokens are counted from the raw corpora; the published chain for the curve over the scale of pretraining holds 1.04 billion observed training values over 30 corpora, against 0.056 billion in the mixture of four. |
| [window-sanity.md](window-sanity.md) | Raw windows and their token reconstructions agree to 1e-14 on every corpus; per-channel shapes and tails are recorded. |
| [corpus-saturation.md](corpus-saturation.md) | C-MAPSS saturates from a quarter of its engines; SMD and SKAB are data-limited; ESA-AD is data-limited once its loss is bounded and its split names its excursion months. |
| [pretraining-corpus-sampling.md](pretraining-corpus-sampling.md) | Only the two sets of intensive-care stays are sampled the way the intensive-care tasks are, so the mixture of four gives the stays' regime 4.5 % of its windows and 3.1 % of its values. |

## Model and objective

| Note | Conclusion |
| --- | --- |
| [encoder.md](encoder.md) | Exact self-attention fits the default window of every corpus at the published tier; exact parameter counts per tier. |
| [synthetic-control.md](synthetic-control.md) | The coupled control pair shares structure and the null pair does not, by a margin fixed beforehand; checksums agree on arm64 and x86. |
| [masked-reconstruction.md](masked-reconstruction.md) | On the positive control every mask kind is learnt against its trivial baseline with an interval, at the published tier. |
| [pretext-variants.md](pretext-variants.md) | The summary term fails its gate on the stays (0.57 against the stack's 0.63) and is closed as not learnt at this scale, at no cost to the reconstruction. |
| [layer-readout.md](layer-readout.md) | At 500 stays, only PhysioNet 2019 alone reads the stays better than the mixture of four over its threshold (+0.020, one pretraining); more pretraining, larger shape and the objective of states lie at or below it. |
| [state-rank.md](state-rank.md) | Over ten backbones held out, the probe's reading at 500 stays rises with the effective rank of the top states (Spearman +0.55), short of the declared one-sided p of 0.05 (0.052), so the rank is not a screen. |
| [zero-shot-pretext.md](zero-shot-pretext.md) | A row copied from the counterpart's learnt row, with no fit, reads the mapped channels' block and token masks within 0.01 to 0.02 of the fitted rows, so the quantity's identity transfers in use. |
| [sepsis-reader.md](sepsis-reader.md) | No pretraining helps the probe on sepsis; within a backbone 4,000 patients resolve 0.02 in 0.97, but between the four intensive-care pretrainings the rule resolves 0.02 in 0.05. |
| [task-transfer.md](task-transfer.md) | On sepsis at 4,000 patients the probe gains 0.031 [+0.015; +0.047] from channel rows fitted on unlabelled windows, but every pretrained backbone still reads sepsis below the initialisation. |
| [domain-continuation.md](domain-continuation.md) | The continuation is closed at both rates: at a tenth of the peak it leaves the reader at the point (−0.000 [−0.007; +0.007] at 500 stays), and at the peak it gains +0.023 [+0.007; +0.038]. |

## Evaluation

| Note | Conclusion |
| --- | --- |
| [transfer-modes.md](transfer-modes.md) | Every transfer mode trains on the registered backbone and answers the task; the untuned control learnt only the mean. |
| [synthetic-transfer.md](synthetic-transfer.md) | At a window spanning the factors' periods the coupled pair transfers (+0.094, floor 0.053); what transfers is mostly the signals' family. |
| [label-efficiency-curve.md](label-efficiency-curve.md) | The endpoint is not confirmed: full fine-tuning is 10.5 % above from scratch at 200 labels and no pretrained arm or baseline beats the control at any budget. |
| [intensive-care-curve.md](intensive-care-curve.md) | On in-hospital death the endpoint is not confirmed: full fine-tuning gains 0.024 in area at 200 stays under the mixed backbone, below a floor of 0.040, and a classical baseline has the highest mean area from 200 stays up. |
| [fine-tuning-recipe.md](fine-tuning-recipe.md) | Slowed-encoder fine-tuning replaces the setting at neither 200 nor 500 stays under either pretraining: it loses up to 0.11, so the loss from 500 stays up lies in the features the pretraining made, not in the recipe. |
| [few-label-fine-tuning.md](few-label-fine-tuning.md) | Not replicated: under all four pretrainings the gain of full fine-tuning at 200 stays is +0.024 [+0.014; +0.033], half the least gain of 0.050. |
| [verdict-statistics.md](verdict-statistics.md) | On a known answer the registered 95 % interval over 21 units covers about 92 % and the rule stands, though a spread of 0.01 in area between seeds, which it cannot see, drops coverage to 70–80 %. |
| [classical-baselines.md](classical-baselines.md) | Tuned trees per channel beat the network from scratch by 26 % at 200 labels and sit about 13 % below the best pretrained arm (not paired); MiniRocket matches aeon. |
| [head-and-representation.md](head-and-representation.md) | The logistic head over the summaries, once its preparation is corrected, reads 0.730 / 0.769, within 0.006 of the trees at 200 stays and 0.018 below them at 500; the per-channel probe lies 0.031 / 0.034 below it. |
| [gap-to-the-trees.md](gap-to-the-trees.md) | Learning from every drawn stay or under a penalty does not move the network over the table, which stays 0.026 to 0.030 below the trees from 200 stays up, so parity needs more than the table. |
| [encoder-candidates.md](encoder-candidates.md) | On set B the small encoder reads 0.850 against the tuned trees' 0.858 (−0.008 [−0.017; +0.001]), not shown no worse by the registered 0.015. |
| [cell-pretext.md](cell-pretext.md) | The objective over cells trains at tier S with every term under its trivial cost from the first epoch, at 21 s an epoch on MPS. |
| [cell-pretraining.md](cell-pretraining.md) | At the rate in force fine-tuning gains 0.021 [0.010; 0.031] at 200 stays and noise at 20 and 50; with each arm at its chosen rate the gain is −0.004 at 200 stays. |
| [task-regime-sources.md](task-regime-sources.md) | The Water Quality Portal fails the regime check through its reader (co-observed 0.68, rate spread 0.36), but published it passes the shortcut audit; ZTF and Mars Express fail. |
| [inference-budget.md](inference-budget.md) | One 3780-token window costs the API 6.6 s and about 1.2 GiB, and bounding batches by token-pair cost makes a burst of eight answer two and refuse six with `503`. |
| [anomaly-tasks.md](anomaly-tasks.md) | Under all four backbones no task reaches the least gain of 0.05 over the initialisation (M1 +0.03 [+0.005; +0.09], OPS-SAT −0.07); with each unit as its own reference the gain stays below 0.05 (M1 +0.007). |
| [composition-screen.md](composition-screen.md) | Fine-tuning gains 0.016 [0.007; 0.025] at 50 stays over the network from nothing at its chosen rate, and the portal joins nowhere. |
| [magnitude-in-states.md](magnitude-in-states.md) | The pretrained states move as far as their initialisation's at every size on both missions; the detector's distance rises a seventh to a half as much because normal windows lie farther apart. |
| [anomaly-signal-in-states.md](anomaly-signal-in-states.md) | No detector enters the screen by the declared rule; on M1 a distance from the month's own centre reads the subtle events at 0.71–0.74 for every encoder alike, though not on PR-AUC. |
| [telemetry-readout.md](telemetry-readout.md) | On the first mission the pretrained states match their initialisation under the whitened distance (+0.009 [−0.009; +0.026]), and no readout qualifies for the confirmation, which reads the leg descriptively. |
| [taught-fine-tuning.md](taught-fine-tuning.md) | On unread months fine-tuning on 5 or 10 events shows no large gain (+0.045 [+0.010; +0.089], −0.024 [−0.043; +0.015]), longer runs read worse (+0.105 [+0.064; +0.154] at 200 steps against 2,000 at 10 events), and all 99 events reach at most about 0.26 under the frozen direction, with or without pretraining, a lower bound on what a detector can reach. |
| [telemetry-supervised-reference.md](telemetry-supervised-reference.md) | Trained on every event of the first mission's fitting side, its length chosen on later fitting months, the network reads 0.265 on the validation months against 0.266 for the frozen direction at the same initialisation (−0.001 [−0.044; +0.040]): no room above 0.26 shows, the shortest run wins, both readouts miss the same events, and the direction leads on the unseen classes. |
| [telemetry-context.md](telemetry-context.md) | Over windows of 6, 24 and 72 hours ending at the scored hour, no layout is chosen on the later fitting months, and read beside on the validation months every context arm over the untrained encoder's states falls below the hour (direction −0.130 [−0.195; −0.066], whitened −0.059 [−0.136; −0.008]): the state's mean over cells dilutes the hour and the hours after an event inherit its trail; smoothing the hour's scores harms more, pooling the previous hours' states changes nothing. |
| [telemetry-channel-local.md](telemetry-channel-local.md) | Each channel's hour against its own recent hours on the first mission: with no label the largest departure over the channels reads 0.179 against the cells' bar 0.160 (+0.019 [−0.093; +0.086], around) and finds 37 of 61 events at 1 %, as many as the taught direction; taught linearly the departures add nothing to the state (−0.079 [−0.138; −0.036] beside it, −0.001 [−0.011; +0.003] as order statistics); taught by trees they find 50 to 54 of 61 events at 1 %, seven of the 11 no arm found before, at an area of 0.17 to 0.19. On the second mission the largest departure reads 0.885 against the bar's 0.627 (+0.258 [+0.199; +0.322], under Holm), level with the taught direction, and the departures again add nothing taught linearly (−0.052 [−0.079; −0.028]); the trees' lead is the first mission's local events'. The package's detector, fitted by a campaign cell, ranks every month as the report did on both missions (Spearman at least 0.99998, the top 1 % held), the same to the bit on the second; on the first the fitting side it is handed holds 39 more windows. |
| [telemetry-pretext-input.md](telemetry-pretext-input.md) | What a telemetry pretext's window should hold, from the fitting months with no label: a ridge forecast of the hidden hour's departure from the median of the five hours before beats that median by 0.214 [0.153; 0.310] on the first mission at one-hour cells over 48 hours, and by nothing on the second (−0.013 [−0.040; 0.016]); finer cells do not help; the rule over both missions chooses one-hour cells over 48 hours and a three-month held-out block. The declared measure first forecast the level and failed its own checks; it was corrected before the rerun. |
| [telemetry-history-window.md](telemetry-history-window.md) | A 48-hour telemetry window concatenated from the hourly publication is the window the publication cuts whole, to the bit, on one month of each mission; its cells joined from the hours laid once match within 2 × 10⁻¹⁵ and are built 70 to 120 times faster on the host (2,812 and 6,727 windows a second against 40 and 56). |
| [telemetry-history-pretext.md](telemetry-history-pretext.md) | Arm A's forecast of each channel's hidden hour beats the history's median on both missions (Mission1 −0.113 [−0.115; −0.112], Mission2 −0.100 [−0.110; −0.091]) and beats one cycle back on Mission1; eight epochs left the values still learning and set no budget, so a sixteen-epoch run is declared. |
