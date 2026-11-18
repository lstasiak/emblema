# ADR-0083: A lane reports its progress transient and its result durable, and the store client waits out a burst of refusals

- Status: accepted
- Date: 2026-11-18

## Context

A machine outside this system runs an order of campaign cells and reports after every cell,
as one document holding every cell answered so far, so that a session ended without warning
can resume from the last reference in its log. This is the handoff of a pretraining run
(ADR-0024) applied to cells. Each report was stored `durable`.
A lane's document grows with every cell, so the bytes it stores grow with the square of its
length: the last reports of a long lane are close to 100 MB each, and a confirmation of 4,200
cells added about 196 GB to the remote bucket. Only the last report of a lane is ever
accepted; the others are needed only until the lane ends. Removing them took a dedicated step
in every clean-up of the bucket.

In the same run four lanes stopped on `ServiceUnavailable` from `PutObject`, after the SDK's
default of three attempts (`reached max retries: 2`). Each was resumed by hand.

## Decision

- **Two operations on the `CampaignHandoff` port.** `report_progress` stores a partial result
  and `report` stores the result a run ends with. The use case says which one it means; the
  adapter maps progress to `Retention.TRANSIENT` and the result to `Retention.DURABLE`. The
  application never names a retention class.
- **Progress expires by the bucket's lifecycle rule** for `<prefix>/transient/`, as
  checkpoints written mid-training already do. Resuming reads the newest reference in the
  lane's log, whichever class it is under. The port's `read_progress` answers `None` for
  progress no longer stored, because a log on persistent storage outlives it; the run then
  starts from its first cell instead of failing on every attempt.
- **The store client makes ten attempts** in the SDK's `standard` retry mode (capped,
  jittered exponential back-off), for every request of a client `S3ArtifactStore.connect`
  opens. This is a constant
  in the adapter, not a setting: nobody has a reason to tune it per environment.

## Consequences

- The intermediate results of a lane leave the bucket by themselves after the transient
  retention (seven days today). The clean-up chain no longer needs to find and delete them.
- A lane interrupted and not resumed within the retention loses its progress and runs again
  from its first cell. The final result of a finished lane is unaffected.
- The closing report stores the whole result once more under `durable/`, though its bytes are
  the same as the last progress report under `transient/`. The duplicate expires with the rest
  of the progress.
- A request now gives up only after about a minute or two of refusals, so a real outage takes
  longer to surface as an error. The mode's per-client retry quota can still end a long burst
  before ten attempts.
- The stored bytes still grow with the square of a lane's length while the lane runs; only
  their lifetime changes.

## Alternatives considered

- **Reporting every N cells.** Fewer bytes, but a session ended early would lose up to N − 1
  cells. In the run above a cell with its report took about 14 s at the end of a lane, against
  26–34 s at the start, so the upload did not set the pace. It would add a setting to solve a
  cost the retention already removes.
- **Reporting each cell as its own object, assembled at acceptance.** Linear bytes, but a new
  document format and a change to resuming and accepting. Worth doing only once the upload time
  matters.
- **A retention argument on `report`.** It would make the application choose a storage class,
  which is an adapter's concern.
- **Retry around the call in the use case.** It would repeat a policy the SDK already provides,
  in every caller.
