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
| [manual-handoff.md](manual-handoff.md) | Runs ordered here, made on a GPU platform and accepted back keep full provenance; includes the first mixed-corpus backbone and its per-corpus curve, and the intensive-care backbones, where adding the stays to the mixture leaves C-MAPSS four times higher for a reason not yet known. |

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
| [label-efficiency-curve.md](label-efficiency-curve.md) | Repeated by the harness with the control under the tail and a rate per budget, the endpoint is not confirmed: full fine-tuning is 10.5 % above from scratch at 200 labels and no pretrained arm or baseline beats the control at any budget; the first curve's 12.3 % was the control's handicap. |
| [intensive-care-curve.md](intensive-care-curve.md) | On in-hospital death the endpoint is not confirmed: full fine-tuning gains 0.024 in area at 200 stays under the mixed backbone, below a floor of 0.040; the stays-alone backbone's frozen probe gains up to 0.055, from 200 stays up a classical baseline has the highest mean area, and the methods that need a grid lead the network on raw readings, so the grid costs them nothing measurable; learnt from 6,400 stays of sets A and B the network still reaches 0.80 against the trees' 0.86, so its deficit is not one of size; a dropout of 0.2 gains it nothing, readings on an hourly grid about 0.01 that no comparison confirms, and neither closes the patch model's lead of 0.043 at 200 stays; the published STraTS on the same stays reaches 0.82 on its own preparation and 0.83 on this project's tokens, 0.04 above the network, so the deficit lies in the network, not in the data; setting the static features apart in the head, or pooling under attention, closes none of it. |
| [verdict-statistics.md](verdict-statistics.md) | On a known answer the registered 95 % interval over 21 units covers about 92 % and excludes a true zero about 10 % of the time two-sided, 5 % above zero, and the rule stands; over the 3,994 stays of the intensive-care task it keeps its level also near chance, under weak or strong pairing and with tied answers, while a spread of 0.01 in area between seeds, which it cannot see, drops its coverage to 70–80 %. On that task one seed's area moves by about 0.04 at 200 labels and 0.01–0.02 at every stay, on a fixed fifth of the tuning side; the selections' repeats understated it. |
| [classical-baselines.md](classical-baselines.md) | MiniRocket matches aeon; tuned trees per channel beat the network from scratch by 26 % at 200 labels and sit about 13 % below the best pretrained arm (not paired); a patch model from scratch ties the network from scratch at 200 and trails the trees by the same margin; at 1,000 labels and at all the trees want a finer ensemble rather than a shallower one and MiniRocket keeps its default grid; over intensive-care outcomes its logistic regression wants a penalty of 4,640, past the published grid, from 200 stays up. |
| [head-and-representation.md](head-and-representation.md) | Pooling the last fifth of the window instead of the whole of it takes a linear head on frozen states from 18.9 % behind the tuned trees to 5.6 % and not distinguishable, and a network trained from nothing from 18.7 to 15.0 RMSE at 200 labels; under that head the advantage of pretraining at 200 labels is not distinguishable from zero over three seeds; a pilot selection on held-out tuning engines finds the arm from nothing short of rate, not steps (three times the peak buys 8 %, twice the steps 2.4 %), while full fine-tuning keeps its setting and loses 38 % at that rate; at the chosen rates twice the floor of steps gains the arm from nothing nothing distinguishable and costs full fine-tuning 11 %, so the floor of 2,000 stands; on the intensive-care outcomes the probe solved in closed form chooses its penalty inside the list it has, under both backbones and both poolings. |
| [inference-budget.md](inference-budget.md) | One 3780-token window costs the API 6.6 s and about 1.2 GiB; eight at once killed the process, and bounding batches by token-pair cost makes the same burst answer two and refuse six with `503`, peak 2.8 GiB of 4; the runtime's arena kept 1.2 GiB per model until given back after each run. |
