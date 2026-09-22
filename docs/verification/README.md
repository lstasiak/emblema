# Manual verifications

Checks that cannot run in GitHub Actions: no MPS, no CUDA, no secrets for the remote bucket, no
notebook runtime. Each run is recorded here as a dated note with the exact versions observed, so a
result is reproducible and its age is visible.

| Note | What it verifies |
|------|------------------|
| [wheels.md](wheels.md) | torch / onnxruntime wheels for cp312 and cp314 on Linux aarch64, Linux x86_64, macOS arm64; GPU-platform Python versions |
| [architecture-rules.md](architecture-rules.md) | every import-linter contract breaks on an injected violation; coverage gate honours `fail_under` |
| [compose.md](compose.md) | the local stack comes up with one command on linux/arm64 (M1): image architectures, health checks, smoke test, S3 contract against Garage |
| [remote-bucket.md](remote-bucket.md) | the S3 contract suite and the lifecycle bootstrap pass against the Cloudflare R2 bucket by configuration alone |
| [onnx-export.md](onnx-export.md) | ONNX export of a set-shaped encoder with one dynamic axis: opset behaviour, empty-window edge case, TorchScript and latency numbers, arm64 leg |
| [data-spike.md](data-spike.md) | licences read at the sources of record; units, windows and tokens counted from the raw corpora; GPU-hour tables per tier from `scripts/corpus_budget.toml` |
| [loader-throughput.md](loader-throughput.md) | batching windows costs a negligible fraction of a training step on MPS; what a window costs held as Python objects |
| [window-sanity.md](window-sanity.md) | a raw window and the reconstruction of its tokens agree by eye and to 1e-14; the shape of every channel of the corpus |
| [published-corpus.md](published-corpus.md) | a corpus publishes to a remote bucket and comes back byte for byte; the checksum of a block and of its manifest repeat across machines |
| [synthetic-control.md](synthetic-control.md) | the generated control pair shares structure and the null pair does not, by a margin fixed before the measurement; checksums agree on arm64 and x86 |
| [encoder.md](encoder.md) | what full self-attention costs at the default window of every measured corpus: exact parameters per tier, attention buffers per window, a training step per window |
| [masked-reconstruction.md](masked-reconstruction.md) | the objective trained on the positive control: the loss falls, the reconstructions follow the signal, and each kind of mask is measured against its trivial baseline with an interval; which frequencies a hidden channel gets back; that the verdicts survived the report handing its training to the training runtime |
| [training-loop.md](training-loop.md) | a run interrupted inside an epoch and resumed from its checkpoint ends in the weights the uninterrupted run ended in — bit for bit on the host, within the spread between repeats on MPS at fp32 and fp16; a run logged to the local MLflow server; what a checkpoint costs and which precisions a machine runs |
| [manual-handoff.md](manual-handoff.md) | a run ordered on this machine, made on another (this machine's accelerator, then the GPU platform and a paid notebook, against the remote bucket) and accepted back: the backbone in the registry with its provenance, the curve replayed into MLflow, what each step costs; the first backbone over the mixture of four corpora, with its validation curve per corpus; the C-MAPSS backbones to their plateau by doublings — over the four subsets, over FD001 and FD003 normalised within one operating condition, and over the four subsets read per operating condition |
| [corpus-saturation.md](corpus-saturation.md) | whether more of a corpus still lowers the loss of a model trained on it at one budget of steps: one curve per corpus over nested shares of its units, judged by rules fixed beforehand; what an evening of it costs on the M1; the satellite backbones scored apart from the excursions of their held-out months, and where a side's squared magnitude lies |
| [transfer-modes.md](transfer-modes.md) | every transfer mode trains on the registered backbone and answers the task's validation engines: what each cost and scored at one budget, against the trivial predictors read off the same predictions; where the control arm stands before the grid |
| [synthetic-transfer.md](synthetic-transfer.md) | the transfer leg of the synthetic control on both pairs: at a window of 32 both fail and the ceiling over shared trajectories puts the fault above the data; at 128, a window spanning the factors' periods, the coupled pair passes (+0.094 against a floor of 0.053) and the null pair fails its equivalence, because it shares the family of the signals — swapped backbones show that the family is what transfers, so the null pair is a control of leakage, not of structure; a backbone pretrained on noise is a worse start than none, so the mechanics alone give nothing; and the pretext's window must span the process's time scales as the task's does |
| [label-efficiency-curve.md](label-efficiency-curve.md) | the label-efficiency curve on the turbofan task, preliminary and on the validation side, read by the registered rules — the paired interval over engines, the endpoint, the secondary family under Holm, the practical floor: the first grid is not confirmed; the configuration changes, each change registered before its run; on the four subsets read per operating condition the endpoint is confirmed, full fine-tuning 12.3 per cent below training from scratch at 200 labelled windows, every pretrained arm 16 to 22 per cent below it at 50 and none ahead once every label is used; the grid over FD001 and FD003 kept as that configuration's record |

Add one file per verification, named `<topic>.md`. Never overwrite a previous run: append a new
dated section, so that the note reads as the history of the measurement and not only its latest
value.
