# Architecture Decision Records

One record per decision that shapes the system. A decision is defended by a reference
implementation and a measurement, not a preference; where two options are viable, the record
states the threshold at which the choice flips.

Format: Status, Date, Context, Decision, Consequences, and optionally Alternatives considered,
Revisit when and Amendments. Status is `proposed`, `accepted`, `rejected` or
`superseded by ADR-XXXX`. A record is at most 800 words and holds one decision. A changed
decision is a new record that supersedes the old one; an amendment is a dated line for a change
of status or a short addition.

The records were condensed to this format on 2026-09-24 without changing any decision, number or
status. The full text as first written is in commit `5f14447`, e.g.
`git show 5f14447:docs/adr/0008-pretraining-corpora.md`.

| ID | Title | Status |
|----|-------|--------|
| [0001](0001-celery-for-background-jobs.md) | Celery for background jobs (arq was the default it replaces) | accepted |
| [0002](0002-python-version-floor.md) | Python lower bound set by the GPU platform, not the local machine | accepted |
| [0003](0003-immutable-domain-model.md) | Immutable domain model: aggregates and value objects as frozen dataclasses | accepted |
| [0004](0004-corpus-aggregate-boundary.md) | `Corpus` is the aggregate root; `CorpusVersion` is an entity inside it | accepted |
| [0005](0005-context-contracts.md) | Context integration: published contracts, shared vocabulary and in-process events | accepted |
| [0006](0006-artifact-store.md) | Artifact store: S3 as the protocol, content-addressed keys, Garage locally and Cloudflare R2 remotely | accepted |
| [0007](0007-onnx-inference-format.md) | ONNX as the inference format: one dynamic token axis, a pinned opset, attention that survives an empty window | accepted |
| [0008](0008-pretraining-corpora.md) | Pretraining corpora: which enter the mix, which only host a task, and the budget that bounds the model | accepted |
| [0009](0009-corpus-version-identity.md) | Corpus version identity: the checksum covers the bytes of the file set the reader declares | accepted |
| [0010](0010-ports-layer.md) | Ports are a layer of the bounded context: Protocols only, between domain and application | accepted |
| [0011](0011-token-representation.md) | Token representation: a window is a set of channel–time–value tokens, normalised per channel, timed relative to the window, with the gap to the previous token of its channel | accepted |
| [0012](0012-where-tokenisation-lives.md) | Where tokenisation lives: the window in the shared kernel, the array codec beside it, the tokeniser a port of the Catalog | accepted |
| [0013](0013-feeding-the-model.md) | Feeding the model: a dataset is a sequence of windows, an epoch's order comes from a digest framed part by part, and the loading layer is shared | accepted |
| [0014](0014-published-corpus-format.md) | A published corpus is a block of windows beside a manifest, in a format we own | accepted |
| [0015](0015-composition-root-by-hand.md) | The composition root is written by hand, without a DI container | accepted |
| [0016](0016-catalog-persistence.md) | The Catalog persists its aggregate as plain tables, migrated by one Alembic tree | accepted |
| [0017](0017-encoder-architecture.md) | The encoder: full self-attention over a set of tokens, time at fixed frequencies, one learned vector per channel | accepted |
| [0018](0018-synthetic-positive-control.md) | The positive control is a corpus reader: one latent process, two sensor layouts, disjoint trajectories | accepted |
| [0019](0019-self-supervised-objective.md) | The self-supervised objective: masked reconstruction with the hidden tokens removed from the encoder, a mixture of channel, block and token masks, and a trivial baseline per kind | accepted |
| [0020](0020-judging-a-masked-reconstruction-run.md) | Judging a masked-reconstruction run: rules in the domain, an interval per comparison, a noise floor, and a probe for the diagnostic the control cannot answer | accepted |
| [0021](0021-training-runtime-and-resumable-runs.md) | The training loop is an adapter behind a port that reports epochs, and a run is resumable mid-epoch | accepted |
| [0022](0022-experiment-configuration-as-a-file.md) | An experiment is a file: every parameter of scale written down, the shape from its tier | accepted |
| [0023](0023-tracking-runs-with-mlflow.md) | Runs are tracked through a port with an MLflow adapter, logged epoch by epoch while they run | accepted |
| [0024](0024-handing-a-run-to-another-machine.md) | A run is ordered here, made anywhere, and accepted back against the order: the backbone registry, the handoff exchange and the replaying runtime | accepted |
| [0025](0025-reading-the-satellite-telemetry.md) | Reading the satellite telemetry: the mission is the unit of independence, the calendar month the unit of splitting; the subsampling is constants, channels are named by mission, pandas is an extra | accepted |
| [0026](0026-splitting-a-downstream-task.md) | A task inherits the corpus's split, names its frozen side rather than holding it, and takes labels through a port | accepted |
| [0027](0027-measuring-corpus-saturation.md) | Measuring corpus saturation: one budget of steps over nested shares of whole units, the share a parameter of the run, and a verdict read by rules fixed beforehand | accepted |
| [0028](0028-bounding-the-loss-of-an-excursion.md) | Bounding what one excursion costs: the objective's loss is a parameter of the experiment, a mixed run reports validation per corpus, and the satellite corpus is split so that its excursions lie on both sides | accepted |
| [0029](0029-pretraining-over-a-mixture-of-corpora.md) | Pretraining over a mixture of corpora: one vocabulary chained through the publications, optimiser steps of one corpus each weighed by their count, the best epoch kept by the mean relative validation, and a run that says where it can be picked up from | accepted |
| [0030](0030-transfer-modes.md) | Transfer modes: four arms of one procedure, low-rank updates as a cap on degrees of freedom, and the encoder reached through a seam the process wires | accepted |
| [0031](0031-reading-the-clinical-stays.md) | Reading the clinical stays: a stay is the unit on the protocol's 48-hour axis, descriptors are timeless tokens with the ward as a presence token, the admission weight is the first weight, and the test set stays out | accepted |
| [0032](0032-statistics-of-a-paired-comparison.md) | Statistics of a paired comparison: the engine is the unit resampled, repeats pool per engine, and the arithmetic a verdict rests on lives in the Evaluation domain | proposed |
| [0033](0033-the-transfer-leg-of-the-synthetic-control.md) | The transfer leg of the synthetic control: a forecasting task whose truth is generated, one ground-truth port for every task, and the generator's specification shared by both sides | accepted |
| [0034](0034-the-turbofan-corpus-read-per-operating-condition.md) | The turbofan corpus read per operating condition: a channel per sensor and condition, scaled within it, and the default reading left as it was | accepted |
| [0035](0035-the-evaluation-campaign.md) | The evaluation campaign: a grid of candidates declared before it runs, a verdict only once it is whole, and one narrow message out of it | accepted |
| [0036](0036-the-classical-candidate.md) | The classical candidate: a port of its own, features that outlive a channel layout, and a process that declares a comparison without being able to run it | accepted |
| [0037](0037-promoting-what-a-campaign-kept.md) | Promoting what a campaign kept: a projection fed by the campaign's announcement, a served model per period of service, and a manual relay for a delivery that failed | accepted |
| [0038](0038-the-tuned-baseline.md) | The tuned baseline: spectra without a grid, a grid in the corpus's time, knobs chosen by a declared selection | accepted |
| [0039](0039-the-patch-baseline.md) | The patch baseline: a network from nothing on the grid, behind a port of its own, on the arms' budget | accepted |
| [0040](0040-the-kept-candidate-in-more-than-one-form.md) | The kept candidate in more than one form: a manifest through the store, an inference graph derived while the run still holds the candidate, and attested against the run's own answers | accepted |
| [0041](0041-the-pooling-of-the-head-as-a-knob.md) | The pooling of the head as a knob: free of the channel layout, turned by name on every network | accepted |
| [0042](0042-the-prediction-service.md) | The prediction service: open host services, a runtime routed by the form kept, and each context's API inside the context | proposed |
| [0043](0043-the-networks-memory-is-bounded-by-cost.md) | The networks' memory is bounded by cost: batches cut and admitted by token pairs | accepted |
| [0044](0044-the-probe-solved-in-closed-form.md) | The probe solved in closed form: a fifth arm whose head is a ridge over the pooled states, chosen leave-one-out, on the arms' budget | accepted |
| [0045](0045-a-binary-task.md) | A binary task: the kind of the target decides the head's link, the draw and what a cell keeps | accepted |
| [0046](0046-comparing-candidates-by-auroc.md) | Comparing candidates by AUROC: one orientation for every measure, strata in the resampling, thresholds in the measure's unit | accepted |
| [0047](0047-a-training-regime-beside-the-schedule.md) | A training regime beside the schedule: a stop on held-out labels, a class weight and withheld channels, as declared knobs | accepted |
| [0048](0048-authentication-at-the-edge-authorisation-in-the-use-case.md) | Authentication at the edge, authorisation in the use case: a shared identity port consumed by the driving side, a policy per context, an allowance per caller on the open routes | accepted |
| [0049](0049-the-scale-of-pretraining-as-a-stage.md) | The scale of pretraining as a stage: a curve of the gain over the data, read at one or two per cent of the labels, before the transfer matrix | accepted |
| [0050](0050-a-head-solved-first-a-patience-in-steps-and-a-probe-at-initialisation.md) | A head solved first, a patience in steps and a probe at initialisation: the knobs of the adaptation recipe at one or two per cent of the labels | accepted |
| [0051](0051-reading-the-scale-corpora.md) | Reading the scale corpora: one corpus per UTSD dataset read on each series' scale, sepsis stays cut by name | proposed |
| [0052](0052-a-task-may-read-each-units-first-window.md) | A task may read each unit's first window: the corpus is published for pretraining, the moment of prediction is the task's | proposed |
| [0053](0053-reading-the-encoders-states-at-another-layer.md) | Reading the encoder's states at another layer: a knob of the frozen modes, every layer through the final normalisation | accepted |
| [0054](0054-predicting-the-teachers-states-at-the-hidden-tokens.md) | Predicting the teacher's states at the hidden tokens: a second objective beside masked reconstruction, chosen by the experiment file | accepted |
| [0055](0055-the-encoders-share-of-the-learning-rate.md) | The encoder's share of the learning rate: a knob of the regime for the modes that train the backbone's weights, so the head learns at the whole rate and the pretrained states at a share of it | accepted |
| [0056](0056-a-pretraining-continued-from-stored-weights.md) | A pretraining continued from stored weights: the start and the trained parameters as knobs of the experiment file, the parent in the registry, the channel table carried by rows | accepted |
| [0057](0057-the-rows-of-a-corpus-the-backbone-never-read.md) | The rows of a corpus the backbone never read: one row per corpus and channel, fitted without labels as the first step of any reading on that corpus; a map of counterparts is a diagnostic, not a reading mode | accepted |
| [0058](0058-telling-the-two-halves-of-a-window-apart.md) | Telling the two halves of a window apart: a contrastive term beside masked reconstruction, the second view the complement of the first, weighted by the experiment file | rejected |
| [0059](0059-predicting-each-channels-summaries-of-the-hidden-part.md) | Predicting each channel's summaries of the hidden part: level and extremes beside masked reconstruction, one unit per channel and window, weighted by the experiment file | rejected |
| [0060](0060-a-run-under-a-second-term-keeps-the-epoch-its-reconstruction-chooses.md) | A run under a second term keeps the epoch its reconstruction chooses: validation reads the hidden tokens' own term alone | accepted |
| [0061](0061-the-windows-statistics-beside-or-instead-of-the-encoder.md) | The window's statistics per channel beside or instead of the encoder: a diagnostic of the network trained from nothing, never a candidate | accepted |
| [0062](0062-tokens-as-cells-of-channel-and-time-bin.md) | Tokens as cells of channel × time bin, laid out on the host before the same set encoder; a learnt time encoding beside the fixed one | accepted |
| [0063](0063-several-learnt-queries-in-the-attention-pooling.md) | Several learnt queries in the attention pooling: a readout per aspect of the window that stays free of the channel layout | accepted |
| [0064](0064-a-pretext-over-cells-masked-before-the-layout.md) | A pretext over cells: masks drawn on the cell grid before the layout, presence and count beside the hidden cells' summaries; the bins as the input part of the architecture | accepted |
| [0065](0065-reading-the-water-quality-portal.md) | Reading the Water Quality Portal: a location over fixed years is the unit, a reading has an instant of its own and one per channel and instant is kept, a characteristic is read in its most frequent unit, the 64 most widely measured characteristics are the channels | accepted |
| [0066](0066-reading-the-eicu-demo-as-2012-frames-a-stay.md) | Reading the eICU demo as 2012 frames a stay: stays of at least 48 hours on the unit, their first 48 hours, the charted vital signs, laboratory results and urine output in the 2012 variables, the monitor tables not read | accepted |
| [0067](0067-a-task-of-events-read-without-a-label.md) | A task of events, fitted on the tuning side without a label and read by the event-weighted area under the precision-recall curve; one cell per detector, the same nearest-neighbour detector under every arm, the benchmark's event-wise score beside | accepted |
| [0068](0068-a-detector-scored-by-the-pretext-error.md) | A detector scored by the error of the pretext's values under the backbone's own masks, drawn per window, against a ridge per channel over the same visible cells; the same alarm as the nearest-neighbour detectors | accepted |
| [0069](0069-a-whitened-distance-over-a-detectors-features.md) | A whitened distance over a detector's features: the squared Mahalanobis distance from the fitting mean under the fitting side's shrunk covariance, a knob beside the nearest neighbours, the same alarm and reference | accepted |
| [0070](0070-the-pretexts-error-read-per-channel.md) | The pretext's error read per channel: each channel's mean error over its typical miss on the fitting side, floored at a tenth of the channels' median; a knob beside the raw mean, the same cells, draws and alarm | accepted |
