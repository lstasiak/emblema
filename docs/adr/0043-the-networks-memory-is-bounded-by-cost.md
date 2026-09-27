# ADR-0043: The networks' memory is bounded by cost — batches cut and admitted by token pairs, and the arena given back

- Status: accepted
- Date: 2026-09-27

## Context

The API runs a network through ONNX Runtime on a thread pool (ADR-0042). Attention compares every
token with every other, so a batch's working memory grows with the square of its longest window
and with the windows padded to it. Two limits bound a request: 4096 tokens a window and 64
windows. Nothing bounded what the process runs *at once*, and nothing bounded what the runtime
*keeps*: each session holds what its runs allocated in an arena of its own.

Measured (`docs/verification/inference-budget.md`): one 3780-token window took 6.6 s and held
about 1.2 GiB; eight such requests at once, inside every request limit, made the kernel kill the
process; and two models served one after another held 3.4 GiB with nothing running. Trees and
MiniRocket cost tens of milliseconds and are not affected.

## Decision

- **Cost is counted in token pairs**: a batch of `n` windows padded to `L` tokens costs `n·L²`.
  A proxy for memory, not a measurement: what a concurrency limit needs is the right ordering
  of what is heavier, in a unit an operator can reason about.
- **The budget is stated as windows of the longest admitted length** that may run at once, per
  server process (`inference_budget_windows`). Capacity is that count times
  `max_tokens_per_window²`, so it follows the window limit and needs no second number.
- **A request is cut into batches by cost, not only by count.** Windows are sorted longest
  first and packed until the next would cost more than one longest window. A batch always
  fits, and short windows are not padded to long ones. Answers return in the order asked.
- **A weighted semaphore admits batches first come, first served.** Each batch holds its cost
  while it runs; a heavy batch is not overtaken by light ones; weight is released however the
  work ends. One instance guards the process, in the graph runtime only.
- **A batch not admitted within `inference_wait_seconds` is refused**: the request answers `503`
  as a problem document with the reason and `Retry-After`. Not a failure, not logged as one.
  The wait bounds the queue by time, not length: a service that queues without limit answers
  everyone late and is killed for its memory in the end.
- **The arena is given back after every run** (`memory.enable_memory_arena_shrinkage`). Kept,
  it would hold a long window's memory per session, past any budget; given back, the process
  returns to its idle size, at no measured cost in time.
- **The container has a memory limit and restarts on failure**, so an overrun ends this process
  alone. The default budget of two longest windows is sized against 4 GiB and one worker.
- Held, waiting and refused are reported as metrics.

## Consequences

- The networks' memory is bounded whatever arrives: about idle plus the budget's working set.
  Short windows still run in parallel (about 40 of 630 tokens fit the cost of two of 4096).
- A burst of long windows is refused after the wait. Measured: two answered, six refused,
  peak 2.8 GiB against the 4 GiB limit; two models in turn, back to 215 MiB after each.
- Two settings are required and have no defaults; `env.example` states them. `workers`
  processes each hold a budget, so the container limit is sized for `workers × budget`.
- A request's batches wait under one deadline, so it waits the configured time as a whole;
  refused part-way, its earlier batches were computed for nothing.
- Requests beyond the thread pool queue in the server before they reach the semaphore, without
  a wait of their own; the pool's size is the only bound there.

## Alternatives considered

- *A plain counting semaphore*: bounds requests, not their cost; too tight for short windows or
  too loose for long ones.
- *A lower window limit*: shifts the threshold and leaves the cause.
- *A container limit alone*: turns the overrun into a restart and loses every request in flight.
- *No arena at all* (`enable_cpu_mem_arena=False`): measured the same as shrinkage; shrinkage
  keeps the arena's speed within a run.
- *One shared arena with a hard cap* (`OrtArenaCfg`): registered process-wide, which two
  applications in one process — a test — cannot share; the semaphore and the limit cap it.
- *Dynamic batching across requests*: more throughput, for a scheduler not yet warranted.

## Revisit when

- Refusals while memory is idle → tune the budget from the metrics, or state it in bytes.
- A GPU provider is used → the arena named, the cost proxy and the unit change.
- Refusals in normal use → an asynchronous job resource (ADR-0042) or scaling out.
