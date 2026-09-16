# ADR-0015: The composition root is written by hand

- Status: proposed
- Date: 2026-09-12

## Context

Publishing a corpus is the first process this system has: something with an entry point, a set of
use cases and a set of adapters that have to be chosen and connected. Use cases take their
dependencies through the constructor, so somewhere those constructors have to be called with real
adapters. That place is the composition root, and where it is and who writes it is a decision.

Two more processes are coming — a Celery worker for evaluation campaigns and an HTTP API — and they
will share most of their wiring with this one. The argument for a container is exactly that: shared
registrations, resolved per process. The argument against is that the question a reader asks of a
composition root is *which adapter did this use case actually get*, and a container answers it
indirectly.

## Decision

**Wiring is written out, in `entrypoints/<process>/composition_root.py`.** A class
`CompositionRoot(settings, **overrides)` assembles the process in its constructor and exposes two
frozen dataclasses: `Services`, the use cases the process can run, each already holding its
dependencies, and `Adapters`, every port implementation those use cases were given. No container:
not `dependency-injector`, not `svcs`, not `punq`.

Every adapter is an override, defaulting to the one a real run gets, and the adapters chosen come
back with the services. That is what lets a test assemble the same process over adapters that
touch nothing and check the wiring itself — that the use cases share one repository, that
artifacts land in the store the process was given, that a run without overrides gets the bucket
the settings name — by reading public fields rather than a registration table or a private
attribute.

**A process that is a sequence of use cases gets a use case for the sequence.** Publishing is
register, freeze, tokenise; that order is an application-layer fact (`PublishCorpus`), so the entry
point parses arguments into one command and calls one callable. An entry point that sequenced use
cases itself would hold logic that only a process test could reach.

**The entry point is a class too.** `PublishCorpusCli.parse` turns arguments into a
`PublishCorpusInvocation`, the command plus where the process reads and writes, and `run` hands it
to a composition root; what the process can publish is a `KnownCorpora` registry rather than a
module-level dictionary. The only module-level code is the `__main__` guard.

Lifetimes are stated by the shape of the code rather than by configuration. This process builds
everything once and exits; a process that needs a dependency per request or per task will say so
with a context manager, in this file, where it can be read.

## Consequences

`CompositionRoot` for the publishing process is around forty lines and names every adapter it
chooses. A reader who wants to know what the process is made of reads them, or reads the
`Adapters` it exposes.

The cost is duplication: when the worker arrives it will repeat the parts it shares with this one —
building the artifact store from the settings, choosing the clock and the identifier source. That
duplication is the thing to watch, not the line count.

## Alternatives

**A DI container** (`dependency-injector`, `svcs`, `punq`). Gains: one declaration of shared
providers, scopes as a feature rather than as a convention. Costs: the wiring stops being readable
as code, the test above becomes a test of registrations, and the library joins the dependencies of
every process. At three processes and a few dozen constructor calls, the gain is small and the loss
is the thing this repository optimises for.

**Wiring inside each entry point.** No extra module, but then a process cannot be assembled without
being run, and the composition test above is impossible.

## Status

`proposed`, not `accepted`. The claim this ADR makes is about *duplication between processes*, and
there is one process. The defence of an ADR in this project is a reference implementation and a
measurement, not a declaration — so this one is settled when the second process exists and the
repetition between the two can be looked at rather than predicted. That is the evaluation
harness, the next process to be assembled.

## Revisit when

- The second and third composition roots exist and repeat the same wiring, and the repetition is
  something other than a handful of lines.
- A process needs dependencies scoped per request or per task and doing it by hand starts to mean
  bookkeeping the code does not make obvious.

## 2026-09-16 — the second process

The pretraining command line (ADR-0024) is the second process assembled by hand, in
`entrypoints/cli/pretrain/composition_root.py` beside the publishing one, which moved to
`entrypoints/cli/publish_corpus/`. What the two repeated before the repetition was removed: the
S3 store connected from the settings and the engine created from them — fourteen lines — and the
guard that refuses to build either without settings. Those became three functions in
`entrypoints/cli/configured.py`, which both roots call. Nothing else is shared: the publishing root
chooses a corpus reader and a block archive, the pretraining root chooses a runtime by whether it
accepts a result and a tracker by whether it has a tracking URI, and neither has a use for the
other's adapters. Both roots still build everything once and exit; no scope per request has been
needed. The status stays `proposed`: the third process, the evaluation worker, is the one with
per-task lifetimes, and the decision is taken there.
