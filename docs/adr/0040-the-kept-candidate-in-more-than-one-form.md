# ADR-0040: The kept candidate in more than one form — a manifest through the store, an inference graph derived while the run still holds the candidate, and attested against the run's own answers

- Status: accepted
- Date: 2026-09-24

## Context

A neural candidate a campaign keeps exists in two forms. The state its runtime fitted — the
weights of encoder and head, the plan's parameters, the vocabulary, the target scale — is what
produced the numbers the campaign scored, what one inspects, and what a later run could continue
from. The inference graph is what the Serving context runs (ADR-0007): a graph that leaves the
training stack behind, and the base every quantised or reduced variant derives from. Both are
blobs in a content-addressed store.

The campaign's announcement names one artifact per kept competitor, by reference and checksum
(ADR-0035), and the projection Serving builds and the model it serves copy that one reference
(ADR-0037). Nothing in that chain says what the bytes are. Until now it did not have to: a kind
had one form. It no longer does. A classical candidate's trees are a joblib document; two more
classical methods, each with a serialisation of its own, are in progress; a patch model trained
from scratch would carry a graph of its own; and the neural candidate now has two forms. The
kind says neural or classical, which picks no reader.

Four things were asked of the design. Serving must find a form it can run without guessing at
bytes. What is served must be traceably derived from what was measured, and the equivalence
checked rather than assumed. The domain and the published messages must carry no technology
names. And a new form must arrive without touching the domain, the message or the schema.

What the wider practice does with this problem is consistent. MLflow's model version is a
directory with an `MLmodel` manifest listing *flavours*, and serving picks one; a Hugging Face
model repository holds the weights beside an `onnx/` export under one identity; Triton's model
repository declares a backend per model in a config file; TFX pushes the SavedModel its evaluator
blessed and validates that it loads in the serving binary before it is served. The common shape:
a logical identity, a manifest of physical forms with their digests, lineage as metadata, the
deployable form made in the pipeline stage that holds the model, and a check on what is deployed.

## Decision

**A manifest names the forms; the campaign names the manifest.** `KeptCandidateManifest` is a
message of Evaluation's published language: the kind, a reference to the published manifest of
the corpus the candidate was fitted to, the forms the candidate is stored in — each a format
name, an artifact reference and a deviation — and which form was measured. It travels through
the artifact store rather than through a call, so it has a codec in the standard library alone,
as the published corpus manifest does (ADR-0014). The announcement, the projection and the served
model are unchanged: one reference, which now points at the manifest, whose checksum pins every
form beneath it. `KeptCandidates` is the adapter every runtime keeps a candidate through: each
form into the store first, the manifest after, because a manifest can only name bytes that exist.
The in-memory stand-ins keep the same way, so a campaign over them ends in artifacts that read as
kept candidates.

**The graph is derived where the candidate is, at the moment it is kept.** The torch runtime
exports `InferenceGraph` from the live candidate after scoring it: encoder, pooling and head in
one graph with two outputs, `pooled_embedding` and `prediction`, the latter multiplied back into
the task's unit so that a reader needs no knowledge of the label ceiling. The export happens in
the worker that carries the training stack and already holds the module; a promotion stays a
metadata operation, and Serving never holds the encoder as a torch module.

**A derived form is attested against the run's answers before it is kept.** The graph is run on
the validation windows the run has just scored, and the largest absolute difference from the
run's answers, in the task's unit, is recorded in the manifest as the form's deviation. A graph
that strays by more than one part in a thousand of the label ceiling is refused
(`InferenceGraphDivergedError`) and nothing is stored. The threshold sits between the two things
it separates: two float32 graphs over different kernels differ by parts in a million, a candidate
whose measured answers came off an accelerator by parts in ten thousand, and a graph that is wrong
— an input out of order, a constant lost — by the whole ceiling.

**Format names are strings owned by the code that writes the bytes**, not a closed vocabulary of
the contract: `torch-state`, `onnx`, `xgboost-joblib`, and whatever a new runtime calls its own.
The registry of readers on the consuming side refuses a name it does not know; the message stays
as it is when a form arrives.

**The manifest carries the reference to the corpus manifest** the candidate was fitted to. The
channel vocabulary a candidate's identifiers mean is then part of the model artefact, and a
service deciding whether a channel is unknown reads it from there, without a new field in the
announcement.

Measured on the fitted candidate under all four transfer modes, over a vocabulary the task grew
and with low-rank updates in place, every constraint of ADR-0007 held: opset 20, `batch` and
`n_tokens` the only symbolic axes, agreement with eager to `1.0e-06` on the state and `1.9e-05`
on the answer over a ceiling of 125, a fully padded window finite, opset 23 still failing at
execution. An export takes 4 to 11 seconds on the x86 development machine; the arm64 leg is
recorded in the verification note when it runs.

## Consequences

- The artifact a campaign announces for a neural candidate is a manifest, and the fitted state
  is one reference further. Campaigns kept before this record point at bare torch bytes, which
  the manifest codec refuses; they were rehearsal runs, and are re-run or left unpromotable.
- Loading a served model is two reads: the manifest, then the form. The store verifies each.
- Every runtime that keeps a candidate keeps it through `KeptCandidates` and names its format. A
  runtime that stores bare bytes produces a candidate nothing can serve; the two classical
  runtimes in progress elsewhere adopt this on merge.
- The inference runtime of the prediction service picks a reader by format, not by kind. The
  kind remains the domain's fact — which budget a candidate shares, which image can run it.
- The rule that messages are composed by the application layer has one deliberate exception
  here: the adapter that holds the forms composes the manifest, because what it states is which
  bytes are in which technology, and only the adapter knows.
- The tests that have guarded the export since the stand-in moved from `tests/ml/onnx_export`
  to `tests/evaluation/adapters/onnx` and run against the fitted candidate under every mode; the
  report script moved with them. The apparatus they replaced — a test wrapper composing encoder
  and pooling — is production code now.
- Latency is where ADR-0007 left it: on the x86 machine at tier M, 114 ms through ONNX Runtime
  against 71.5 ms eager for one window of 512 tokens. That record's revision threshold — the
  target container, tuned session options, the quantised variants — stands, and the graph now
  exists to be measured there.

## Alternatives considered

- **A second reference through the whole chain**, `artifact` beside `inference_artifact` in the
  outcome, the cell result, the announcement, the projection and the served model, with columns
  in three tables. Rejected: a field per form, empty or duplicated for a classical candidate,
  the domain and the schema changed again by every new form, and still nothing that picks a
  reader for a classical candidate.
- **The graph as the kept artifact, with its source named in the graph's own metadata.**
  Rejected: the lineage sits inside a blob the registry cannot see, it assumes one true form per
  candidate, and it says nothing about the classical formats.
- **The forms as a collection in the domain**, each with a format, persisted in a table of their
  own and published as a tuple. It is the one option that makes the lineage a query in SQL.
  Rejected for now: it puts serialisation formats into the domain and the message, either as an
  enum that changes with every form or as strings in the core, and nobody asks the database that
  question today. It is where this record goes if someone does.
- **Exporting at promotion**, in the Serving context through a seam over the encoder, or
  **lazily** when a model is first loaded. Rejected: the first turns a metadata operation into a
  computation that needs the training stack in the process that promotes; the second brings that
  stack into the service, which is what the graph exists to avoid. Both convert where the model is
  not.

## Revisit when

- Someone needs to ask the database about forms across candidates — which kept candidates have
  no graph, every graph of one format. Then the forms get a table in the evaluation schema,
  filled from the manifests, and the manifest stays the document a consumer reads.
- A form has to be added to a candidate after its campaign closed, as an efficiency variant is.
  Then the variant is its own record naming the manifest it derives from; the manifest, once
  written, is not rewritten.
- The prediction service needs more from a candidate than its corpus manifest and its forms.
  Then the manifest gains a field, and the announcement does not.
- The deviation of a graph derived from an accelerator-fitted candidate approaches the threshold.
  Then the threshold is revisited with those numbers, not moved to make them pass.
