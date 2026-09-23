# ADR-0001: Celery for background jobs

- Status: accepted (arq was the default this replaces)
- Date: 2026-09-07

## Context

Evaluation campaigns are grids of hundreds of small runs; pretraining hand-offs, ONNX export and
efficiency benchmarks also run outside the request cycle. The API is FastAPI. Two worker modes share
one code base: a containerised `worker-cpu` for classical baselines, statistics and export, and a
natively run `worker-ml` for neural runs, because MPS is not available inside Docker on macOS (plan
section 10). The queue is behind the `JobQueue` port in the shared kernel; the broker is an
adapter detail. Campaign state (which runs exist, which finished, whether the grid is complete)
lives in the `EvaluationCampaign` aggregate in Postgres, never in the queue.

The plan proposed arq as the default because it is asyncio-native and small. The maintainer has
production experience operating Celery and wants a mature, backend-grade tool whose additional
features are available when needed rather than adopted under pressure later.

## Decision

Use **Celery** with Redis as the broker as the default `JobQueue` adapter.

- Maturity and operations: retries with backoff, late acknowledgement, visibility timeouts, task
  routing, monitoring (Flower), periodic tasks (beat). Drift monitoring and scheduled re-pretraining
  map directly onto beat without a second scheduler.
- Two queues, `cpu` and `ml`, routed with `task_routes`; `worker-cpu` consumes `cpu` in the
  container, `worker-ml` consumes `ml` natively on the host. Same code, two queue names.
- Operational experience lowers risk on the one resource that is actually scarce here: the
  maintainer's time.

## Consequences

- **Synchronous execution model.** Tasks run in a worker process, not on an event loop. Use cases in
  the application layer stay synchronous; asyncio is confined to the FastAPI edge. If an async use
  case ever appears, the task adapter bridges it with `asyncio.run`, not the other way round.
- **Native ML worker pool.** `fork()` after importing torch is unsafe with MPS (and CUDA). The
  `worker-ml` runs with `--pool=solo` and concurrency 1; GPU jobs are sequential anyway. `worker-cpu`
  uses the default prefork pool.
- **No result backend by default** (`task_ignore_result = True`). Facts the domain depends on are
  written to Postgres by the use case the task invokes; Celery results would be a second source of
  truth (the MLflow/Postgres role split applies here too).
- **Broker URL credentials are URL-quoted** when the URL is built from settings. kombu/urllib fail
  in non-obvious ways on raw special characters in DSNs (lesson carried over from other projects).
- Larger dependency footprint (kombu, billiard, vine) in the API and worker images. The inference
  image (T-8a.2) does not include a worker, so the size target is unaffected.
- One `celery_app` factory in `entrypoints/workers`, configured from `Settings`; task modules import
  use cases, never adapters directly. Import-linter keeps `celery` out of domain and application.
- `celery[redis]` is added by the first change that enqueues work, not by the initial setup.

## Alternatives considered

- **arq** — the default this replaces: asyncio-native, tiny surface, Redis-only. Rejected: the asyncio
  advantage is marginal for CPU/GPU-bound jobs, the ecosystem and operational tooling are thin, and
  the maintainer has no operational experience with it. Its simplicity does not outweigh learning a
  new tool on the project's critical path.
- **Dramatiq** — simpler than Celery, thread-based, fewer moving parts. Rejected: no advantage over
  Celery given existing experience; smaller ecosystem.
- **Synchronous in-process execution** — kept as the test adapter of `JobQueue`, not as the
  production path.

## 2026-09-23 — RabbitMQ replaces Redis, and the port is renamed

**The broker.** This record chose Redis in passing: the alternatives it weighed were arq, Dramatiq
and running work in process, never another broker. Redis turned out to serve nothing else here —
no result backend, no cache — so it stood on its own merits, and on those it loses. Redis has no
acknowledgement of its own; Celery emulates one with a visibility timeout, a number that has to be
guessed above the longest a job can take and that redelivers anything slower to a second consumer.
A campaign's cell trains a network for as long as it takes. RabbitMQ acknowledges natively: a job
stays unacknowledged for exactly as long as the worker holds it, and publishing waits for the
broker's confirmation, so a submission that returns is one the broker took responsibility for.
Both are now configured in `celery_application`.

What this costs is a heavier service in the local stack, and it removes nothing the project used:
the state a campaign is resumed from was never in the queue.

**RabbitMQ 4 withdrew transient non-exclusive queues, and Celery declares them.** Not a
configuration detail — a worker that opens one never finishes starting, which is how this was
found. Celery uses that kind of queue for the mailbox workers talk to each other and to `celery
inspect` through. It is switched off rather than the broker asked to permit what it has
deprecated: `worker_enable_remote_control = False` in the application, and `--without-mingle
--without-gossip` on the worker, because the startup handshakes are command-line flags with no
setting behind them. The cost is live inspection of workers, Flower included, which nothing in
this project uses and which ADR-0001 listed as available rather than needed.

**The port is `JobQueue`, not `JobScheduler`.** Scheduling in Celery's vocabulary is Beat, which
runs work on a clock; this port hands work over and returns. The name would have collided with
the beat scheduler this record anticipates for drift monitoring. The enum that named the two
pools of workers gave up the name `JobQueue` for `WorkerPool`, which is what it always described,
and `ScheduledJob` became `QueuedJob`. Celery's own word, `task`, stays inside the Celery adapter
and `@app.task`, where it is Celery's to use: the plan wrote this port for arq, a Kubernetes Job
and an in-process runner as well, and only one of those four says "task".
