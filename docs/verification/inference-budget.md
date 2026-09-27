# Inference budget: what a network costs the API in time and memory, and what bounds it

## 2026-09-27 — cost of one window, the overrun, and the bounded service

**Question.** How much time and memory does one request cost the API when it runs a network, and
does bounding the networks by cost keep the process alive under a burst that used to kill it?

**Conditions.**

| | |
|---|---|
| Code | branch `feature/T-6.3-prediction-api` at `e61c6483`, plus the budget (ADR-0043), uncommitted |
| Machine | MacBook Pro M1 Pro; Docker Desktop VM with 8.7 GiB for all services |
| Process | the `api` image, ONNX Runtime on `CPUExecutionProvider`, 2 threads a call, batch size 16 |
| Model | the `from_scratch` network of a turbofan campaign, 126 channels, 30 readings a channel |
| Requests | synthetic readings, so the numbers say what a request costs, not how accurate it is |
| Before | no container memory limit, no bound on concurrent batches, 8 request threads |
| After | 4 GiB container limit, budget of 2 windows of 4096 tokens, wait 5 s |
| Client | Python `urllib`, one connection per request |

**One request, before the budget** (memory read with `docker stats`; the API was idle at about
320 MiB after start):

| Readings in a window | Time | Memory after the request |
|---|---|---|
| 630 | 0.6 s | 423 MiB |
| 1890 | 2.2 s | 679 MiB |
| 3780 | 6.6 s | 1.51 GiB, and it did not come back down |

The classical candidates answered the 3780-reading window in 28–46 ms at the median.

**Bursts, before the budget.**

| Load | Result |
|---|---|
| 32 requests of 630 readings, 8 at once | all 200, 3.9 s, 1.52 GiB |
| 16 requests of 1890 readings, 4 at once | all 200, 13.7 s, 1.57 GiB |
| 8 requests of 3780 readings, 8 at once | process killed by the kernel (`OOMKilled`, exit 137) |

**After the budget** (same burst of 8 × 3780 readings, 8 at once):

| Result | Count | Time |
|---|---|---|
| 200 | 2 | 6.9–7.0 s |
| 503 with `Retry-After`, reason in the body | 6 | 5.3 s (the wait, plus the request) |

Peak memory was 2.83 GiB against the 4 GiB limit; the process was not killed and `/health`
answered throughout. 32 requests of 400 readings, 8 at once, all answered 200 in 1.9 s. The
refusals were counted in `emblema_inference_refused_total`; the budget in use, its capacity and
the batches waiting are reported as gauges.

**What the runtime keeps.** Two models served one after another through the API, one
3780-reading window each, twice (`docker stats`, idle 232 MiB): 1.50 → 1.85 → 3.11 → 3.45 GiB.
Nothing was running between requests. In the image, the same two graphs run directly through
ONNX Runtime, three rounds each, RSS read from `/proc` after every run:

| Session options | After model A | After model B | After round 3 |
|---|---|---|---|
| default | 1447 MiB | 2673 MiB | 3362 MiB |
| arena shrinkage after every run | 212 MiB | 215 MiB | 215 MiB |
| no arena | 229 MiB | 231 MiB | 232 MiB |
| one shared arena capped at 2 GiB, shrinkage | 213 MiB | 215 MiB | 215 MiB |

A run took 6.1–6.5 s under each option and 6.2–9.7 s under the default. With shrinkage in
the service, the same two models through the API, twice each: 171 → 248 → 248 → 283 → 293 MiB;
the burst of eight answered as before, peak 2.4 GiB.

**Conclusions.**

1. Cost grows faster than linearly with a window's tokens: 6× the tokens took 11× the time, and
   working memory rose from about 100 MiB to about 1.2 GiB. A count of requests does not bound it.
2. The overrun needs a burst of the longest windows; ordinary traffic was answered without
   error at 8 requests at once.
3. Bounding batches by cost turns the burst into refusals the client can act on, and the
   process stays inside its memory limit. Short windows still run in parallel.
4. Two long windows at once held 2.83 GiB, so the default budget of two fits 4 GiB with
   headroom; a budget of three would not.
5. The budget bounds what runs at once, not what the runtime keeps: by default every session
   holds a long window's memory after it, and served models add up past any budget. Giving the
   arena back after each run removes that at no cost in time; the service does so.

**Limitations.**

- One model, one machine, one execution provider. The unit is token pairs, a proxy: about 90
  bytes a pair at 3780 tokens here (1.19 GiB above idle over 3780² pairs), including the linear
  terms.
- The memory figures of the bursts come from a sampled `docker stats`, not from the process.
- Refused requests were not retried, so what a retrying client would experience is unmeasured.
- Readings were synthetic; no claim about the answers.
