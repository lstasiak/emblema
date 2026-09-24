# ADR-0006: Artifact store — S3 as the protocol, content-addressed keys, Garage locally and Cloudflare R2 remotely

- Status: accepted
- Date: 2026-09-09
- Full text before condensation: commit `5f14447`

## Context

Pretraining writes checkpoints and backbones, the Catalog publishes tokenised corpora, Serving
reads promoted artifacts. Training runs on free GPU platforms (Kaggle, Colab) that cannot reach a
laptop, so the store must be on the public internet, at zero cost. The domain refers to an artifact
through `ArtifactRef(key, checksum)`. MinIO, the first choice for the local service, stopped
publishing images in 2025 and archived its repository; the last official image predates the fix
for CVE-2025-62506.

## Decision

- **Port** `ArtifactStore` in `shared/ports`: `put(content, retention) -> ArtifactRef`,
  `get(ref) -> bytes`, `exists(ref)`. The store hashes what it receives and derives the key, so a
  reference can never point at different bytes and equal content gives an equal reference. `get`
  verifies the checksum (`ArtifactIntegrityError`); a missing object raises
  `ArtifactNotFoundError`. No `delete`: durable artifacts are provenance; transient ones expire.
- **Retention** is vocabulary of the port: `DURABLE` for everything the domain registers,
  `TRANSIENT` for intermediates such as mid-training checkpoints. It is the first key segment, so
  renaming a member is a data migration.
- **Key layout** `<retention>/<algorithm>/<digest>` (`durable/sha256/3b8f…`), identical in every
  adapter. The environment prefix (`dev`, `test`) is S3 adapter configuration, not part of the
  reference.
- **Adapters.** `LocalDirectoryArtifactStore` (fake; atomic writes) and `S3ArtifactStore` (boto3,
  synchronous). `connect` sets path-style addressing and checksums only when required (boto3
  ≥ 1.36 adds a CRC trailer R2 rejects).
- **Local service: Garage** v2.4 — maintained, multi-arch, lifecycle expiry by prefix, single-node
  bootstrap from environment variables. The choice lives in `compose.yaml`.
- **Remote service: Cloudflare R2.** Free tier (10 GB-month, 1M class A / 10M class B operations),
  **no egress charge** — decisive, because each GPU session downloads the corpus. R2 has no bucket
  versioning; content-addressed keys give the same guarantee.
- **One bucket, prefixes per role**: `dev/`, `test/` for artifacts, `mlflow/` for MLflow's own run
  artifacts.
- **Lifecycle rules** by `scripts/bootstrap-bucket.sh` (AWS CLI): expiry of
  `<prefix>/transient/`, abort of incomplete multipart uploads after a day; run against Garage by
  compose and against R2 with the remote env file.
- **Public distribution** of a finished backbone (Hugging Face Hub) is a separate port,
  `ModelDistribution`, added with publication.
- **Tests**: one contract suite over both adapters. The S3 case builds from `Settings`, so it runs
  against Garage in CI and against R2 by hand (`uv run --env-file .env.r2`); see
  `docs/verification/remote-bucket.md`.

## Consequences

- **Whole objects in memory.** Streaming, resumable multipart and presigned URLs wait until an
  artifact is larger than a worker should hold. Content addressing then needs a two-pass hash or a
  staged upload; the layout does not change.
- Every `get` hashes the content: O(size) per read, the price of provenance.
- `put` issues a `HEAD` first and uploads identical content once. A truncated object from an
  earlier crash is not repaired on write; the `get` that finds it raises.
- No store setting has a default; a process that names no bucket fails at start. Credentials have
  no fallback anywhere; CI mints them per run.
- `shared/adapters` never imports `config`; the composition root builds the store.
- The CI unit job needs Docker for the S3 contract so the adapter counts towards coverage.

## Alternatives considered

| Alternative | Why not |
|---|---|
| MinIO from a community mirror | third-party build of archived software, no security fixes |
| MinIO built from the archived tag | frozen code, slow image build |
| RustFS | lifecycle rules "under testing" at the time; revisit at a stable 1.0 |
| LocalStack | AWS API emulator, heavier than a storage server |
| Caller-chosen keys | a key could be reused for different bytes |
| fsspec / smart_open | a second abstraction under the port for two adapters |
| aiobotocore | the application is synchronous (ADR-0001) |
| testcontainers | a second definition of services compose already defines |
| Backblaze B2 | viable; egress beyond 3× storage is billed — kept as the second provider |
| Kaggle Datasets | no S3 API, no lifecycle, one-way |
