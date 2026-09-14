# scripts

Everything here is run by hand, outside the package. Each script is one of three kinds:

- **Tooling** operates the local stack or fetches data. It stays.
- **Measurement reports** produce, on one machine, the numbers and figures a verification note in
  `docs/verification/` records. A report stays while a note relies on it. It is deleted in the
  ticket that makes the package take the same measurement; the note keeps citing the commit that
  produced its numbers, so the old code can still be checked out and rerun.
- **Transitional** scripts stand in for package code that a named later ticket builds, and go with
  that ticket.

A script imports the package and other scripts, never the tests
(`tests/architecture/test_scripts_import_no_tests.py`). The report of the ONNX export spike
(T-0.6) runs the suite's own harness, so it lives with that suite in `tests/ml/onnx_export/report.py`
and goes with it at T-6.2.

Logic that a verdict depends on belongs in `src/emblema`, under the architecture rules, types and
coverage, not here. That is why the rules of the T-2.2 assessment live in `emblema.pretraining`
and only their storage and printing remain below.

| Script | Ticket | Kind | Evidence it produces | Goes with |
| --- | --- | --- | --- | --- |
| `smoke.sh` | T-0.5 | tooling | run by the CI job `compose` | — |
| `bootstrap-bucket.sh` | T-0.5 | tooling | run by the compose service `bootstrap` | — |
| `fetch_corpora.py` | T-1.0a | tooling | archive checksums for a note | — |
| `corpus_facts.py` | T-1.0a | measurement | counts pasted into `corpus_budget.toml` | — |
| `corpus_budget_report.py`, `corpus_budget.toml` | T-1.0a | measurement | budget tables per tier | — |
| `budget_file.py` | T-2.1 | shared | reads corpus facts from `corpus_budget.toml` for the other scripts | — |
| `loader_throughput_report.py` | T-1.3 | measurement | batching against a training step | — |
| `window_sanity_report.py` | T-1.4 | measurement | `figures/cmapss-*.png` | — |
| `synthetic_control_report.py` | T-1.5a | measurement | `figures/synthetic-control-*.png` | — |
| `published_corpus_report.py` | T-1.6 | measurement | artifact size and write cost | — |
| `encoder_budget_report.py` | T-2.1 | measurement | attention cost per window length | — |
| `masked_reconstruction_report.py`, `masked_reconstruction_assessment.py`, `masked_reconstruction_epochs.toml`, `spectral_probe.py` | T-2.2 | transitional | [`masked-reconstruction.md`](../docs/verification/masked-reconstruction.md) and `figures/masked-reconstruction-*.png` | T-2.3, T-2.4: training runtime, experiment tracker, provenance of a run |
| `reporting.py` | T-1.6 | shared | the heading and table shape of every report | — |

## Known debt

- The tickets T-0.6, T-1.0a, T-1.3, T-1.4, T-1.5a, T-1.6 and T-2.1 have no note in
  `docs/verification/` yet. Figures exist for T-1.4 and T-1.5a.
- Eight scripts put the repository root on `sys.path` before their imports, and each carries a lint
  exemption (E402) for it in `pyproject.toml`.
