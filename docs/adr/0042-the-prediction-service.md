# ADR-0042: The prediction service — open host services, a runtime routed by the form kept, and each context's API inside the context

- Status: proposed (accepted once a promoted candidate answers on arm64)
- Date: 2026-09-25

## Context

A served model is a reference to the manifest of forms a campaign kept (ADR-0037, ADR-0040).
Serving it over HTTP asks things nothing yet provides. A request of raw readings must become the
tokens the candidate was fitted on, by the Catalog's arithmetic. A classical candidate's form
expects a reading of a window that lives in Evaluation's adapters, which Serving may not import
and must not restate. A browser client asks for lists and views shaped for reading. The
process could also hand work to the queue.

## Decision

- **Two open host services, published as `Protocol` in `contracts/`.** The Catalog publishes
  `WindowTokeniser`; Evaluation publishes `KeptCandidateInference`, a classical form's answer
  from bytes the caller fetched. The provider defines and implements each; the serving process
  composes the adapter in. Neither is a port of Serving; both are listed beside the adapter
  seams. The graph's input and output names are published in Evaluation's contracts.
- **One port, `InferenceRuntime`, routed by format**: the graph where one was kept, otherwise the
  measured form where the classical service reads it, refused by name otherwise. Loaded
  candidates are kept by checksum for the life of the process; providers and threads are
  configuration.
- **What the model takes is read off the corpus manifest the kept candidate names.** A reading
  on a channel never observed in training is dropped and named; a window without a reading, too
  many windows or readings, an oversized body or a reading of the wrong kind is refused. A
  reading is one token, so length is refused before tokenising, and a body before parsing.
- **Each context owns its HTTP surface.** `<context>/api/` holds its routes, schemas and refusal
  statuses: a driving adapter beside the driven ones in `adapters/`, neither importing the other.
  Route classes build an `APIRouter` over the use cases they are given; the process includes
  the routers and owns probes, metrics, CORS, the body limit and the error shape.
- **Read models fed through listing ports.** Lists are paged by keyset; the cursor carries the
  position, so a page costs one query and needs no stored row. Campaign lists count cells
  rather than load them. Verdicts are read at request time and kept in a `VerdictMemo` port
  under campaign and revision, by a bounded least-recently-used adapter.
- **Synchronous inference, no queue**, on a thread pool sized by configuration and matched by
  the database pool; the graph runtime and the trees release the interpreter lock.
- **One error shape** (RFC 9457) for every refusal and every unnamed failure. A server error
  carries the trace it is logged under, nothing of the service's insides.
- **Telemetry outside the core.** Decorators trace and time the runtime and the tokeniser, and
  count what the use cases answered and ignored; requests, SQL and store calls are traced by
  their instrumentations; OTLP export only where a collector is named.

## Consequences

- A request names the model it asks; nothing yet answers "the model for task X".
- The API image carries the graph runtime and the baselines' libraries, and no torch.
- Tokenisation is pure Python, about two microseconds a token; it is timed on its own.
- Promotion and withdrawal stay on the command line until authenticated.

## Alternatives considered

- *A joblib adapter in Serving*: the feature reading restated in a second context.
- *Inference as a task on the queue*: a round trip through the broker and a result store for an
  answer of milliseconds, and workers that would each load the models the API already holds.
- *The verdict materialised when a campaign closes*: right once several processes read verdicts;
  today a memo per process suffices.
- *All routes and schemas in the entrypoint*: the entrypoint would carry every context's
  vocabulary and grow with each; a context's HTTP surface is its driving adapter.
- *Module-level route functions with `Depends` for the use cases*: idiomatic for the framework,
  but a service locator over process state; constructor injection keeps one composition root.

## Revisit when

- A client needs the model serving a task → a per-task aggregate (ADR-0037).
- A second process reads verdicts, or the memo misses often → materialise at closing.
- A request scores more windows than one answer should wait for → an asynchronous job resource
  (`202 Accepted`) run by a worker.
- Tokenisation shows in the latency budget → the vectorised tokeniser (ADR-0012).
- An efficiency variant is served in place of its source → the served model names the variant.
