# Verification notes — lab notebook

The archive of evidence behind the decisions in [`docs/adr/`](../adr/README.md) and the results in
[`docs/findings.md`](../findings.md). Each note records measurements that CI cannot make (MPS, a
GPU platform, the remote bucket, the full corpora), with the commit, machine, tier and command.

Notes are append-only: a new run is a new dated section, and a superseded section is marked with
one sentence, never rewritten. The preregistration cites notes by file, date and title.

A section has this shape: question → conditions → tables and figures → conclusions → limitations.
It is at most about 1,500 words. Sections written before 2026-09-24 predate this format and are
kept as recorded.

## Infrastructure

| Note | Conclusion |
| --- | --- |
| [wheels.md](wheels.md) | torch and onnxruntime wheels exist for cp312 and cp314 on every target; Kaggle and Colab ship Python 3.12. |
| [architecture-rules.md](architecture-rules.md) | Every import-linter contract fails on an injected violation, and the coverage gate honours its threshold. |
| [compose.md](compose.md) | The local stack comes up with one command on linux/arm64 and passes its smoke test and S3 contract against Garage. |
| [remote-bucket.md](remote-bucket.md) | The S3 contract and lifecycle bootstrap pass against Cloudflare R2 by configuration alone. |
| [onnx-export.md](onnx-export.md) | The set encoder exports to ONNX with one dynamic token axis at opset 20, and so does the fitted candidate under every transfer mode, answering in the task's unit within 1.9e-05 of eager; ONNX Runtime is slower than TorchScript on the stand-in.; on arm64 the graph of a candidate fitted on MPS answers within 1.2e-04 of the run, a thousandth of the refusal threshold. |
| [published-corpus.md](published-corpus.md) | A published corpus round-trips through the remote bucket byte for byte, and its checksums repeat across machines. |
| [loader-throughput.md](loader-throughput.md) | Batching costs a negligible share of a training step on MPS; windows as Python objects are too large to hold a corpus. |
| [training-loop.md](training-loop.md) | A run resumed mid-epoch ends in the uninterrupted run's weights: bit for bit on the host, within repeat spread on MPS. |
| [manual-handoff.md](manual-handoff.md) | Runs ordered here, made on a GPU platform and accepted back keep full provenance; includes the first mixed-corpus backbone and its per-corpus curve. |

## Data

| Note | Conclusion |
| --- | --- |
| [data-spike.md](data-spike.md) | Licences, units, windows and tokens counted from the raw corpora; the classical corpora are small and ESA-AD makes the reference model legitimate. |
| [window-sanity.md](window-sanity.md) | Raw windows and their token reconstructions agree to 1e-14 on every corpus; per-channel shapes and tails are recorded. |
| [corpus-saturation.md](corpus-saturation.md) | C-MAPSS saturates from a quarter of its engines; SMD and SKAB are data-limited; ESA-AD is data-limited once its loss is bounded and its split names its excursion months. |

## Model and objective

| Note | Conclusion |
| --- | --- |
| [encoder.md](encoder.md) | Exact self-attention fits the default window of every corpus at the published tier; exact parameter counts per tier. |
| [synthetic-control.md](synthetic-control.md) | The coupled control pair shares structure and the null pair does not, by a margin fixed beforehand; checksums agree on arm64 and x86. |
| [masked-reconstruction.md](masked-reconstruction.md) | On the positive control every mask kind is learnt against its trivial baseline with an interval, at the published tier. |

## Evaluation

| Note | Conclusion |
| --- | --- |
| [transfer-modes.md](transfer-modes.md) | Every transfer mode trains on the registered backbone and answers the task; the untuned control learnt only the mean. |
| [synthetic-transfer.md](synthetic-transfer.md) | At a window spanning the factors' periods the coupled pair transfers (+0.094, floor 0.053); what transfers is mostly the signals' family. |
| [label-efficiency-curve.md](label-efficiency-curve.md) | The endpoint is confirmed on the per-condition corpus: full fine-tuning 12.3 % below from scratch at 200 labels; no pretrained arm leads once every label is used. |
| [verdict-statistics.md](verdict-statistics.md) | On a known answer the registered 95 % interval over 21 units covers about 92 % and excludes a true zero about 10 % of the time two-sided, 5 % above zero; the shortfall is reported, the rule stands. |
| [classical-baselines.md](classical-baselines.md) | MiniRocket matches aeon; tuned trees per channel beat the network from scratch by 26 % at 200 labels and sit about 13 % below the best pretrained arm (not paired); a patch model from scratch ties the network from scratch at 200 and trails the trees by the same margin. |
| [head-and-representation.md](head-and-representation.md) | Pooling the last fifth of the window instead of the whole of it takes a linear head on frozen states from 18.9 % behind the tuned trees to 5.6 % and not distinguishable, and a network trained from nothing from 18.7 to 15.0 RMSE at 200 labels; under that head the advantage of pretraining at 200 labels is not distinguishable from zero over three seeds; attention from a zero query does not help; the last reading alone is not the task. |
