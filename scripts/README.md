# scripts

Everything here is run by hand, outside the package. Each script is one of three kinds:

- **Tooling** operates the local stack or fetches data. It stays.
- **Measurement reports** produce, on one machine, the numbers and figures a verification note in
  `docs/verification/` records. A report stays while a note relies on it. It is deleted once the
  package takes the same measurement; the note keeps citing the commit that produced its numbers,
  so the old code can still be checked out and rerun.
- **Transitional** scripts stand in for package code that is not written yet, and are deleted by
  the change that writes it.

A script imports the package and other scripts, never the tests
(`tests/architecture/test_scripts_import_no_tests.py`). The report of the ONNX export spike runs
the suite's own harness, so it lives with that suite in `tests/ml/onnx_export/report.py` and is
retired with it.

Logic that a verdict depends on belongs in `src/emblema`, under the architecture rules, types and
coverage, not here. That is why the rules that judge a masked-reconstruction run live in
`emblema.pretraining`, and why the loop that trains one, the state it can be resumed from and the
record of what it did followed them there. What remains below is composition — which corpus, which
experiment — and the shape of a note: its CSV, its tables, its figures.

| Script | Kind | Evidence it produces | Retired by |
| --- | --- | --- | --- |
| `smoke.sh` | tooling | run by the CI job `compose` | — |
| `bootstrap-bucket.sh` | tooling | run by the compose service `bootstrap` | — |
| `fetch_corpora.py` | tooling | archive checksums for a note | — |
| `corpus_facts.py` | measurement | counts pasted into `corpus_budget.toml` | — |
| `corpus_budget_report.py`, `corpus_budget.toml` | measurement | budget tables per tier | — |
| `budget_file.py` | shared | reads corpus facts from `corpus_budget.toml` for the other scripts | — |
| `loader_throughput_report.py` | measurement | batching against a training step | — |
| `window_sanity_report.py` | measurement | `figures/cmapss-*.png` | — |
| `synthetic_control_report.py` | measurement | `figures/synthetic-control-*.png` | — |
| `published_corpus_report.py` | measurement | artifact size and write cost | — |
| `encoder_budget_report.py` | measurement | attention cost per window length | — |
| `masked_reconstruction_report.py`, `spectral_probe.py` | measurement | [`masked-reconstruction.md`](../docs/verification/masked-reconstruction.md) | the evaluation harness, once it owns campaigns and the statistics a published result is read from; what would stay is the note's composition |
| `masked_reconstruction_assessment.py` | transitional | the CSV a run is stored and compared in | the evaluation harness, which persists a campaign rather than a directory of files |
| `masked_reconstruction_figures.py` | measurement | `figures/masked-reconstruction-*.png` | — |
| `training_loop_report.py` | measurement | [`training-loop.md`](../docs/verification/training-loop.md) | — |
| `reporting.py` | shared | the heading and table shape of every report | — |

## Known debt

- Ten scripts put the repository root on `sys.path` before their imports, and each carries a lint
  exemption (E402) for it in `pyproject.toml`.
- `masked_reconstruction_report.py` is the largest thing here and does three jobs — publishing a
  corpus, diagnosing what a run learnt, and rendering a note. Each has tests; none of them is
  domain logic. It shrinks to composition when the evaluation harness takes the assessment over.
