# Local stack on linux/arm64

What CI cannot check: GitHub runners are amd64. The stack must also come up on the development
machine (MacBook M1, Docker Desktop, linux/arm64 containers), and the S3 contract suite must pass
against Garage from the host.

## Procedure

Prerequisites the first run discovered: Docker's virtual disk needs about 2.5 GB free (aws-cli
~400 MB, the MLflow build ~1.2 GB, plus layer cache and the volumes) — `initdb` fails with "no
space left on device" long before anything says the word disk. Host ports 5432 and 5000 are the
ones in use on a developer machine; the overrides sit in `env.example`.

```sh
cp env.example .env
docker compose config --quiet
time docker compose up -d --wait --wait-timeout 300
docker compose ps
docker compose logs bootstrap
bash scripts/smoke.sh
uv run pytest -m integration tests/shared/ports/test_artifact_store_contract.py
docker compose down -v
```

Record: Docker Desktop and Compose versions, image digests and architectures (`docker image inspect
--format '{{.Architecture}}' <image>`), time to healthy, smoke output, pytest summary.

Points the Windows machine could not settle and this run must:

- Garage accepts the `GARAGE_DEFAULT_ACCESS_KEY` format used in `env.example` (`GK` + 32 hex).
- The Garage health check (`garage -c /etc/garage.toml status`, which in v2 talks to the admin API
  with the `admin_token` from the config file) reports healthy inside the container.
- The MLflow image builds on arm64 and artifact serving writes to Garage (smoke uploads and reads
  back one artifact). The compose command passes `--artifacts-destination` but not
  `--serve-artifacts`, relying on it being enabled by default; if the upload fails, pass the flag.
- `docker compose up --wait` treats the finished one-shot `bootstrap` service as success. If it
  reports the exited container as a failure, keep the one command by moving `bootstrap` behind a
  profile — `up` then leaves it alone and a wrapper script runs it afterwards, which is also how
  `compose run` already reaches it.
- Lifecycle configuration with an empty `Filter.Prefix` (abort-multipart rule) is accepted by Garage.
- Nothing that authenticates has a default, so `cp env.example .env` is required rather than
  convenient: without it `docker compose config` refuses the file. The services publish on
  `127.0.0.1` only; the host still reaches them, the network does not.
- `scripts/smoke.sh` runs to the end. It reads the AWS CLI's output through a pipe, which aws-cli
  v2 would page if it were given a terminal; `compose run -T` and `AWS_PAGER=""` are meant to
  prevent that, and this run is the first place the combination is exercised. A stall at the step
  "bucket: lifecycle rules" is that failure.

Settled by the first run, and fixed on the branch: publishing on `127.0.0.1` makes `localhost` the
wrong address, because it resolves to `::1` first. On macOS the smoke test then reached AirPlay
Receiver on port 5000 and reported `curl: (22) ... 403`. Every address the host dials is now an IPv4
literal. Prerequisite discovered the same way: Docker's virtual disk needs roughly 2.5 GB free, or
`initdb` fails with "no space left on device".

## Runs

### 2026-09-09 — MacBook M1, Docker Desktop, linux/arm64

Result: pass, after two fixes made during the run. The whole stack came up with
`docker compose up -d --wait`, `scripts/smoke.sh` reported all three services ready, and
`uv run pytest -m integration` passed against Garage.

Settled here:

- Images pull and run on linux/arm64; the MLflow image builds locally from `python:3.12-slim`.
- Garage accepts the `GK` + 32 hex access key from `env.example` and its `garage status` health
  check reports healthy in the container.
- `docker compose up --wait` accepts the one-shot `bootstrap` finishing: the profile fallback is
  not needed.
- MLflow stores and serves the artifact through Garage with `--artifacts-destination` alone; the
  explicit `--serve-artifacts` flag is not required on MLflow 3.
- Lifecycle rules install on Garage, empty `Filter.Prefix` included.
- `scripts/smoke.sh` runs to the end: `compose run -T` plus `AWS_PAGER=""` keeps aws-cli out of a
  pager.
- `docker compose config` refuses to run without `.env`, and the services answer on the host while
  listening on `127.0.0.1` only.

Found and fixed on the branch:

1. `localhost` is the wrong address once ports are published on `127.0.0.1`: it resolves to `::1`
   first, where on macOS AirPlay Receiver holds port 5000 and answers `403`, which surfaced as
   `curl: (22)` in the smoke test. Fixed in `08bdc96` and `46fa75a` — every address the host dials
   is an IPv4 literal, including the MLflow health check inside the container.
2. Two host ports were already taken (5432 by a Homebrew Postgres, 5000 by AirPlay Receiver), and
   Docker's virtual disk ran out during `initdb`, which reports "no space left on device". Both are
   now prerequisites of this procedure.

Not captured in this record: Docker Desktop and Compose versions, image digests, time to healthy,
the exact smoke and pytest output. Append them from the terminal that ran it.
