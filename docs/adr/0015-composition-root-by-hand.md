# ADR-0015: The composition root is written by hand

- Status: accepted (settled by the third process, the campaign worker)
- Date: 2026-09-12; amended 2026-09-16, 2026-09-22, 2026-09-24
- Full text before condensation: commit `5f14447`

## Context

Use cases take dependencies through the constructor, so some place must call those constructors
with real adapters: the composition root. More processes were coming (workers, API) and would share
wiring — the argument for a DI container. The argument against: a reader asks *which adapter did
this use case get*, and a container answers indirectly.

## Decision

- **Wiring is written out** in `entrypoints/<process>/composition_root.py`. A class
  `CompositionRoot(settings, **overrides)` assembles the process in its constructor and exposes two
  frozen dataclasses: `Services` (use cases with dependencies) and `Adapters` (every port
  implementation used). No container (`dependency-injector`, `svcs`, `punq`).
- **Every adapter is an override** defaulting to the real one, so a test assembles the process over
  inert adapters and checks the wiring by public fields.
- **A sequence of use cases is a use case** (`PublishCorpus`), so the entry point calls one callable.
- **The entry point is a class**: `<Process>Cli.parse(argv)` gives an invocation; `run` builds the
  root and calls one use case. Only the `__main__` guard is module-level.
- **Lifetimes are stated by code shape**: process-scoped by default; per-request or per-task
  scopes would be a context manager in the root.

## Consequences

- Each root names every adapter it chooses; `Adapters` shows what a process is made of.
- The cost to watch is duplication between processes.

## Alternatives considered

- *A DI container*: one declaration of shared providers and scopes as a feature, but the wiring is
  no longer readable as code, tests check registrations, and every process takes the dependency.
- *Wiring inside each entry point*: a process cannot be assembled without being run.

## Revisit when

- Wiring duplicated between processes grows beyond a handful of lines.
- A per-request or per-task scope becomes bookkeeping the code does not make obvious.

## Amendments

- **2026-09-16 — the second process** (pretraining CLI, ADR-0024). The shared part was fourteen
  lines (store and engine from settings, the guard refusing either without settings), extracted to
  helper functions. Status stayed `proposed`.
- **2026-09-22 — the third process, and the decision.** The campaign worker shares the same three
  helpers, now in `entrypoints/configured.py`; the seam from stored weights to an encoder moved to
  `entrypoints/restored_backbones.py`. The expected per-task scope did not appear: scoping the
  backbone per task would download it for every cell, so the worker holds one and refuses cells of
  other weights (ADR-0035). Status → `accepted`.
- **2026-09-24 — the threshold fired.** Two workers and a declaring CLI shared the store,
  registries, queue, clock, identifiers and label drawing. That is assembled once by
  `CampaignProcess`, which the roots hold rather than inherit; each root states only its candidate
  provider. The declaring root takes only the describing half of the provider port and carries
  neither ML stack — on macOS torch and XGBoost cannot share a process, so a container resolving a
  provider by type would have crashed inside a library. The promotion CLI (ADR-0037) repeats one
  eight-line idiom; a third repetition would be extracted.
