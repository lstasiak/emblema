# ADR-0048: Authentication at the edge, authorisation in the use case — a shared identity port consumed by the driving side, a policy per context, and an allowance per caller on the open routes

- Status: accepted
- Date: 2026-10-01; accepted 2026-10-03 (a token-bearing promotion answered on the arm64 stack)

## Context

The API (ADR-0042) answered reads and windows to anyone and changed nothing: promotion and
withdrawal stayed on the command line because nothing could say who was asking. Two things
were missing, easy to conflate. *Who is asking* is a technical matter of the transport.
*Whether they may* is a rule of the project — who may put a model into service — and belongs
where the operation is performed, so that it holds for every edge and is tested without any. A
third is operational: an open inference route on a free tier disappears within a week unless
each caller is rationed.

## Decision

- **`Principal` and `Scope` in the shared kernel.** A subject and the scopes it was granted; a
  value object with validation and no behaviour; a scope is `resource:action`. The HTTP edge,
  the command line and every context's policy read the same thing, and none of them owns it.
- **`IdentityProvider` is a shared port consumed by the driving side.** `identify(credential)
  → Principal`, one refusal for unknown, malformed, expired and foreign alike, and a separate
  unavailability when the provider cannot be asked. It is the one port no use case calls: the
  HTTP adapter asks it and passes the principal inwards in the command, so a credential never
  crosses into the application. Two adapters: static tokens for a local stack and a CI run,
  and signed tokens of an external issuer verified offline against its published keys, by
  asymmetric algorithms only, so the service holds nothing it could mint with. One provider at
  a time, chosen by configuration.
- **Authorisation is a policy class in the application layer.** `PromotionPolicy` names the
  scope of each operation that changes what is served and refuses an actor without it; the
  commands carry the actor, and the use cases ask the policy before anything is looked up. The
  command line acts as the operator of the machine that keeps the registry — the shell already
  let them in — and still goes through the policy. One scope per operation.
- **Protected routes are a router, not a flag.** Promotion and withdrawal live in a router of
  their own that depends on the bearer authentication once; the handler reads the actor back
  off the request. A missing or rejected token answers `401` with the `Bearer` challenge, an
  actor the policy refuses `403`, both in the one error shape; the schema marks exactly those
  routes.
- **The open routes are rationed per caller, in the process.** A token bucket per network
  address: a minute's allowance as the bucket and the burst, refilled at a steady rate; a
  bounded number of callers remembered, the least recently seen forgotten first. Past the
  allowance a caller gets `429` and `Retry-After`. Only the routes that run a model are
  rationed.

## Consequences

- The allowance and the callers remembered have no defaults, and a process with routes to
  protect refuses to start without a provider.
- The allowance is per server process and per address: callers behind one proxy share one, and
  `workers > 1` multiplies it. Enough for a demo, not for a public deployment.
- Campaigns and corpora are still registered from the command line; when they get HTTP, they
  get a policy of their own on the same pattern.

## Alternatives considered

- *A principal per context*: the edge could not type what it hands in; three copies of one thing.
- *Checking the scope in the router*: a rule that holds for HTTP only, testable only through it.
- *An authorisation server of our own*: a portfolio that writes its own OAuth2 signals the
  opposite of what it means to. The issuer is someone else's; the service verifies.
- *Token introspection*: a round trip to the issuer on every write; offline verification costs
  one key fetch.
- *A shared counter store for the allowance*: right once several processes answer.
- *Rationing every route*: a read is one query; what needs rationing runs a model.

## Revisit when

- A browser client needs a session or a refreshed token → the issuer's login flow; the port
  stays, the adapter changes.
- The API runs behind an ingress or with several workers → the caller read from a trusted
  forwarded header, the allowance in a store the processes share.
- A second context exposes writes → its own policy class, the same command shape.
- Authenticated callers deserve another allowance → the subject as the key where a token was
  sent.
