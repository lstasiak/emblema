# ADR-0001: Celery for background jobs

- Status: accepted (arq was the default this replaces)
- Date: 2026-09-07; amended 2026-09-23 (broker, port name, images, pool names)
- Full text before condensation: commit `5f14447`

## Context

Evaluation campaigns are grids of hundreds of runs; pretraining handoffs, ONNX export and
benchmarks also run outside the request cycle. The API is FastAPI. Work goes through the
`JobQueue` port in the shared kernel; the broker is an adapter detail. Campaign state (which runs
exist, which finished, whether the grid is complete) lives in Postgres in the
`EvaluationCampaign` aggregate, never in the queue.

arq was the proposed default: asyncio-native and small. The maintainer has run Celery in
production and wants a mature tool whose features are there when needed.

## Decision

- **Celery** is the `JobQueue` adapter: retries with backoff, late acknowledgement, routing,
  periodic tasks (beat) for later drift monitoring.
- **RabbitMQ is the broker** (2026-09-23; Redis before). Redis served nothing else here and has no
  native acknowledgement — Celery emulates one with a visibility timeout that must be guessed above
  the longest job and redelivers anything slower. A campaign cell trains for as long as it takes.
  RabbitMQ holds a job unacknowledged exactly while a worker has it, and publishing waits for the
  broker's confirmation.
- **RabbitMQ 4 refuses transient non-exclusive queues**, which Celery uses for its remote-control
  mailbox. Remote control is off (`worker_enable_remote_control = False`,
  `--without-mingle --without-gossip`). The cost is live inspection (Flower), which nothing uses.
- **The port is `JobQueue`**, not `JobScheduler`: in Celery's vocabulary scheduling is beat. The
  pool enum is `WorkerPool`, a job is a `QueuedJob`. `task` stays inside the Celery adapter.
- **Every process is a container image** (2026-09-23). One Dockerfile, one image per set of extras;
  the command picks the process. Running the same entry point on the host, to reach MPS, is an
  escape hatch, not the architecture. Long runs go to a GPU platform through the artifact handoff
  (ADR-0024).
- **Two pools, `ml` and `general`**, named for what the image carries, not hardware (the target has
  no accelerator in either). The split holds because a process that imported torch cannot safely
  `fork()`, so `ml` runs `--pool=solo` one job at a time; a cell that trains for minutes must not
  block seconds-long work; and the `ml` image is more than twice the size. The default queue is
  `general`.
- A job carries its pool; a worker is started with its queue; the task declaration states neither.
- Each worker assembles one `CampaignProcess` (store, registries, queue, clock, identifiers, label
  drawing) and differs only in the candidate provider on top. A worker missing required
  configuration exits at start with the reason.

## Consequences

- **Synchronous execution.** Use cases stay synchronous; asyncio is confined to the FastAPI edge.
- **No result backend** (`task_ignore_result = True`). Facts the domain depends on are written to
  Postgres by the use case; Celery results would be a second source of truth.
- Broker credentials are URL-quoted when the URL is built from settings.
- Larger dependency footprint in API and worker images; the inference image carries no worker.
- Import-linter keeps `celery` out of domain and application; task modules call use cases only.
- **The pre-merge gate runs in the image**: same interpreter, wheels and OS as production. Eight
  MPS-gated tests skip there, so the suite also runs on the development machine.

## Alternatives considered

- **arq**: asyncio advantage is marginal for CPU/GPU-bound jobs; thin tooling; no operational
  experience.
- **Dramatiq**: no advantage over Celery given existing experience; smaller ecosystem.
- **Redis as broker**: see above — visibility timeout instead of acknowledgement.
- **In-process execution**: kept as the test adapter of `JobQueue`.
