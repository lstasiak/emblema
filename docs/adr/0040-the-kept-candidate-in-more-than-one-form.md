# ADR-0040: The kept candidate in more than one form — a manifest through the store, an inference graph derived while the run still holds the candidate, and attested against the run's own answers

- Status: accepted
- Date: 2026-09-24; amended 2026-09-25 (every runtime keeps through the manifest)
- Full text before condensation: commit `8f6d13d`

## Context

A neural candidate a campaign keeps exists in two forms: the state its runtime fitted — what the
campaign scored, what one inspects, what a later run could continue from — and the inference
graph Serving runs (ADR-0007), free of the training stack and the base of every quantised variant.
The announcement names one artifact per kept competitor (ADR-0035); the projection and the served
model copy that reference (ADR-0037). Nothing in the chain says what the bytes are. While a kind
had one form that did not matter. It no longer holds: the trees are a joblib document, MiniRocket
a NumPy archive, the patch model a torch state, the neural candidate two forms. The kind picks no
reader.

Asked of the design: Serving finds a form it can run without guessing at bytes; what is served is
traceably derived from what was measured, the equivalence checked rather than assumed; domain and
messages carry no technology names; a new form arrives without touching domain, message or schema.
The wider practice agrees on the shape — MLflow's `MLmodel` flavours, a Hugging Face repository
with weights beside `onnx/`, Triton's per-model backend, TFX validating the blessed SavedModel in
the serving binary: one identity, a manifest of physical forms with digests, lineage as metadata,
the deployable form made where the model is, a check on what is deployed.

## Decision

- **A manifest names the forms; the campaign names the manifest.** `KeptCandidateManifest`, a
  message of Evaluation's published language: the kind, a reference to the published manifest of
  the corpus the candidate was fitted to, the forms — each a format name, an artifact reference
  and a deviation — and which form was measured. It travels through the store, so its codec is
  standard library alone (as ADR-0014). Announcement, projection and served model are unchanged:
  one reference, now to the manifest, whose checksum pins every form beneath it.
- **`KeptCandidates` is how every runtime keeps**: each form into the store first, the manifest
  after, because a manifest can only name bytes that exist. Torch, xgboost, MiniRocket, the patch
  model and the in-memory runtimes keep this way. The stated-errors provider, which fakes the
  whole evaluation rather than fitting anything, stores bare bytes.
- **The graph is derived where the candidate is, at the moment it is kept.** The torch runtime
  exports `InferenceGraph` from the live candidate after scoring it: encoder, pooling and head,
  two outputs `pooled_embedding` and `prediction`, the latter in the task's unit. The worker that
  carries the training stack does it; promotion stays a metadata operation; Serving never holds
  the encoder as a torch module.
- **A derived form is attested before it is kept.** The graph runs on the validation windows the
  run has just scored, and the largest absolute difference from the run's answers, in the task's
  unit, is recorded as the form's deviation. Above one part in a thousand of the label ceiling the
  graph is refused (`InferenceGraphDivergedError`) and nothing is stored. The threshold sits
  between what it separates: kernels differ by parts in a million, an accelerator-fitted candidate
  by parts in ten thousand, a wrong graph by the whole ceiling.
- **Format names belong to the code that writes the bytes**, not to a closed vocabulary of the
  contract: `torch-state`, `onnx`, `xgboost-joblib`, `minirocket-npz`, `torch-patch-state`. One
  contract test holds the names distinct and every form readable back through the manifest. The
  consumer's registry of readers refuses a name it does not know; the message does not change.
- **The manifest carries the corpus manifest reference**, so the channel vocabulary a candidate's
  identifiers mean is part of the model artifact; a service deciding whether a channel is unknown
  reads it there, without a new field in the announcement.
- **The patch model is kept as its measured form only.** The graph it would be served through
  reads a grid rather than tokens; which context lays that grid is decided with the prediction
  endpoint.

Measured on the fitted candidate under all four transfer modes, over a grown vocabulary and with
low-rank updates in place, every constraint of ADR-0007 held: opset 20, `batch` and `n_tokens`
the only symbolic axes, 1.0e-06 on the state and 1.9e-05 on the answer over a ceiling of 125, a
fully padded window finite, opset 23 still failing at execution. An export takes 4 to 11 s on
x86; the arm64 leg is recorded in the verification note when it runs.

## Consequences

- The artifact a campaign announces is a manifest; the fitted state is one reference further.
  Candidates kept before this record point at bare bytes the codec refuses: rehearsal runs, re-run
  or left unpromotable.
- Loading a served model is two reads, the manifest then the form; the store verifies each.
- The inference runtime picks a reader by format, not by kind. The kind stays the domain's fact:
  which budget a candidate shares, which image can run it.
- One deliberate exception to "messages are composed by the application layer": the adapter that
  holds the forms composes the manifest, because which bytes are in which technology is known
  only there.
- The export suite moved from `tests/ml/onnx_export` to `tests/evaluation/adapters/onnx` with its
  report; the wrapper it composed encoder and pooling with is production code now.
- Latency is where ADR-0007 left it (x86, tier M: 114 ms through ONNX Runtime, 71.5 ms eager);
  that record's revision threshold stands, and the graph now exists to be measured there.

## Alternatives considered

- *A second reference through the whole chain* (`inference_artifact` beside `artifact` in the
  outcome, the cell result, the announcement, the projection, the served model): a field per form,
  empty or duplicated for a classical candidate, domain and schema changed by every new form, and
  still nothing that picks a reader.
- *The graph as the kept artifact, its source in the graph's own metadata*: lineage inside a blob
  the registry cannot see; assumes one true form; silent on the classical formats.
- *The forms as a domain collection, in a table of their own, published as a tuple*: the one
  option that makes lineage a SQL query, and where this record goes if someone asks that question.
  Rejected now: serialisation formats in the domain and the message, as an enum that changes with
  every form or as strings in the core.
- *Exporting at promotion* through a seam over the encoder in Serving, or *lazily* on first load:
  the first needs the training stack where promotion happens; the second brings it into the
  service the graph exists to keep it out of.

## Revisit when

- Someone asks the database about forms across candidates → the forms get a table in the
  evaluation schema, filled from the manifests; the manifest stays what a consumer reads.
- A form is added after the campaign closed, as an efficiency variant is → its own record naming
  the manifest it derives from; a written manifest is not rewritten.
- The prediction service needs more than the corpus manifest and the forms → the manifest gains a
  field, the announcement does not.
- The graph of an accelerator-fitted candidate approaches the threshold → revisited with those
  numbers, not moved to make them pass.
