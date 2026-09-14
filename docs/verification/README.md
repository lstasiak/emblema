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
| [masked-reconstruction.md](masked-reconstruction.md) | the objective trained on the positive control: the loss falls, the reconstructions follow the signal, and each kind of mask is measured against its trivial baseline with an interval; which frequencies a hidden channel gets back |

Add one file per verification, named `<topic>.md`. Never overwrite a previous run: append a new
dated section, so that the note reads as the history of the measurement and not only its latest
value.
