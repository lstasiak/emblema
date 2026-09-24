# Architecture Decision Records

One record per decision that shapes the system. A decision is defended by a reference
implementation and a measurement, not a preference; where two options are viable, the record
states the threshold at which the choice flips.

Format: Status, Date, Context, Decision, Consequences, and optionally Alternatives considered,
Revisit when and Amendments. Status is `proposed`, `accepted` or `superseded by ADR-XXXX`. A
record is at most 800 words and holds one decision. A changed decision is a new record that
supersedes the old one; an amendment is a dated line for a change of status or a short addition.

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
