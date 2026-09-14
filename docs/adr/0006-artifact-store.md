# ADR-0006: Artifact store — S3 as the protocol, content-addressed keys, Garage locally and Cloudflare R2 remotely

- Status: accepted
- Date: 2026-09-09

## Context

Pretraining writes checkpoints of hundreds of megabytes and registers final backbones, the
Catalog publishes tokenised corpora, Serving reads promoted artifacts. Training runs on free GPU
platforms (Kaggle, Colab) that cannot reach a laptop, so the store must be reachable from the public
internet, and the project must run at zero cost. The domain refers to
an artifact through `ArtifactRef(key, checksum)` and lets a backbone become `ready` only with an
artifact whose checksum matches (section 6.4).

The plan fixes S3 as the protocol with two adapters, a local directory and an S3 client, and named
MinIO as the local S3 service. Between that choice and this record MinIO stopped publishing community
images (2025-10-23) and archived its repository (2026). The last official image on Docker Hub
predates the fix for CVE-2025-62506; the fixed release exists only in community mirrors.

## Decision

**Port.** `ArtifactStore` in `shared/ports` has three operations: `put(content, retention)`
returning an `ArtifactRef`, `get(ref)` returning the bytes, `exists(ref)`. The store computes the
checksum of what it receives and derives the key from it, so an existing reference can never come
to point at different bytes and storing the same content twice yields the same reference. `get`
verifies the bytes against the reference's checksum before returning them and raises
`ArtifactIntegrityError` otherwise; a missing object raises `ArtifactNotFoundError`. There is no
`delete`: durable artifacts are the provenance of every published result, transient ones expire by
lifecycle rule.

**Retention as vocabulary of the port.** `Retention.DURABLE` is the class of everything the domain
registers; `Retention.TRANSIENT` is for intermediate output such as mid-training checkpoints. The
caller chooses; the store only guarantees that transient objects sit under a prefix a lifecycle rule
can expire. The enum sits beside the `Protocol`, as `HashAlgorithm` sits beside `Checksum`: today
it is an argument of one port and nothing else, so the shared kernel would be a home built ahead of
a tenant. Revisit when a bounded context persists or publishes a retention class — a checkpoint
recorded by Pretraining, a read model exposing one; the kernel then takes it as the closed
vocabulary of two contexts, and the key layout becomes a creation method on `ArtifactRef`.

Because the retention class is the first segment of every key, its value is not free to change:
objects already stored answer to the old word, and the bucket's lifecycle rules match on it.
Renaming a member is a data migration, not an edit.

**Key layout.** `<retention>/<algorithm>/<digest>`, e.g. `durable/sha256/3b8f…`, shared by every
adapter so that a reference produced by one resolves in another once the bytes are copied. The
environment prefix (`dev`, `test`) is configuration of the S3 adapter, not part of the reference: a
reference recorded in one environment names the same content in another.

**Adapters.** `LocalDirectoryArtifactStore` is the fake of the port: tests and Docker-less
development, atomic writes (temporary file moved into place). `S3ArtifactStore` wraps a boto3
client, synchronous like the rest of the application (ADR-0001). Its `connect` factory carries the
two settings every S3-compatible service we target agrees on: path-style addressing (a local
endpoint cannot resolve bucket subdomains) and checksums only when an operation requires them
(boto3 ≥ 1.36 signs every upload with a CRC trailer by default, which R2 rejects). `None`
credentials defer to the SDK credential chain.

**Local service: Garage** (`dxflrs/garage`, v2.4). Actively maintained, multi-arch images,
lifecycle `Expiration` with prefix filters, and since v2.3 a single-node mode that lays out the
cluster and creates the bucket and access key from environment variables, which is all the
bootstrap MinIO offered. The code never knows which service is behind the endpoint; the choice
lives in `compose.yaml` and one TOML file.

| Alternative | Why not |
|---|---|
| MinIO, last release from a community mirror | third-party build of archived software; no future security fixes |
| MinIO built from source at the archived tag | no third-party trust, but frozen code and a slow image build |
| RustFS | 1.0 still in beta; lifecycle management marked "under testing" with open reports of rules not deleting |
| LocalStack | S3 emulation aimed at AWS API testing; heavier than a storage server |

Revisit when RustFS reaches 1.0 with stable lifecycle rules, or if Garage stops releasing. Either
way the swap is a compose change.

**Remote service: Cloudflare R2.** Free tier: 10 GB-month storage, 1 million class A and 10
million class B operations per month, no egress charge. Zero egress is decisive: a notebook
downloads the tokenised corpus in every GPU session, which on S3, GCS or (beyond 3× storage) B2 is
the one line item that could produce a bill. Lifecycle configuration and multipart are supported
through the S3 API. R2 has no bucket versioning, which was wanted as a property of the
store; content-addressed keys give the same guarantee — new content is a new key, and `get`
verifies what it reads — so versioning is not a requirement on any provider.

**One bucket, prefixes per role.** `dev/` and `test/` hold the platform's content-addressed
artifacts; `mlflow/` holds MLflow's own run artifacts (`--artifacts-destination`). MLflow keeps raw
training logs, Postgres keeps the domain facts (section 6.6); the two roles share storage but not a
key space.

**Lifecycle rules by one script.** `scripts/bootstrap-bucket.sh` (AWS CLI) makes sure the bucket
exists and installs one expiry rule per environment prefix for `<prefix>/transient/` plus an abort
of incomplete multipart uploads after a day. The compose service `bootstrap` runs it against
Garage; the same service with the remote environment file runs it against R2.

**Two roles, two ports.** The system artifact store (private, versioned by content, with full
provenance) and public distribution of the finished backbone (Hugging Face Hub) are different
responsibilities. The latter is a separate port, `ModelDistribution`, that arrives with publication.

**Tests.** One contract suite, parametrised over adapters. The S3 case builds its store from
`Settings`, so the same tests run against Garage in CI (started from `compose.yaml`) and against R2
by hand with `uv run --env-file .env.r2`. That run is the proof of "switching stores is a matter of
configuration" and is recorded in `docs/verification/`.

## Consequences

- **Bytes in, bytes out.** The port carries whole objects in memory. Streaming, multipart upload
  with resumption and presigned URLs (section 7) are deferred to the first real checkpoint: the
  threshold is an artifact larger than a worker should hold in memory. Content addressing then
  needs either a two-pass hash or a staged upload renamed after hashing; the layout does not change.
- Every `get` hashes the content: O(size) per read, accepted as the price of provenance.
- `put` costs a `HEAD` before the `PUT`; identical content is uploaded once. It trusts what the
  `HEAD` reports: an object truncated by an earlier crash is not repaired, and the `get` that
  discovers it raises `ArtifactIntegrityError`. Verifying on every write would read back the whole
  object and cost more than the case is worth.
- No field describing the store has a default value. A process that names no bucket fails on
  startup, rather than resolving a default that quietly points at another environment's prefix;
  the values of the local stack live in `env.example`, and nothing that authenticates has a
  fallback anywhere, including `compose.yaml`. CI mints the stack's password and access key per
  run, so the only credentials written down in the repository are the ones in the template a
  developer copies.
- `shared/adapters` never imports `config`; the composition root builds the store from
  `Settings.artifact_store`.
- The CI unit job depends on Docker for the S3 contract so that the adapter counts towards
  coverage; a separate job brings the whole stack up and runs the smoke test on amd64.
- Postgres schemas are created by the database's own init scripts; the first migration of each
  context uses `CREATE SCHEMA IF NOT EXISTS`.

## Alternatives considered

- **Caller-chosen keys** (`put(key, content)`) — rejected: a key could be reused for different
  bytes; provenance would rest on the checksum in the reference alone.
- **A filesystem abstraction** (fsspec, smart_open, cloudpathlib) — rejected: a second abstraction
  under the port, with its own semantics for atomicity and errors, for two adapters.
- **Provider SDKs** (MinIO Python client) — rejected: boto3 speaks to every S3-compatible service.
- **aiobotocore** — rejected: the application layer is synchronous (ADR-0001).
- **testcontainers** — rejected for now: the compose file already defines the service, and the
  tests should exercise the stack developers run; a per-test container adds a library and a second
  definition of the same service.
- **Backblaze B2** — viable and configuration-switchable, kept as the second provider of risk
  egress beyond 3× stored volume is billed.
- **Kaggle Datasets as the channel** — fallback only: no S3 API, no lifecycle, one-way.
