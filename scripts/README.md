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
| `raw_corpora.py` | shared | finds a downloaded corpus's root under `data/raw/` for the reports that read one | — |
| `loader_throughput_report.py` | measurement | batching against a training step | — |
| `window_sanity_report.py` | measurement | `figures/<corpus>-*.png` | — |
| `synthetic_control_report.py` | measurement | `figures/synthetic-control-*.png` | — |
| `published_corpus_report.py` | measurement | artifact size and write cost | — |
| `encoder_budget_report.py` | measurement | attention cost per window length | — |
| `masked_reconstruction_report.py`, `spectral_probe.py` | measurement | [`masked-reconstruction.md`](../docs/verification/masked-reconstruction.md) | the evaluation harness, once it owns campaigns and the statistics a published result is read from; what would stay is the note's composition |
| `masked_reconstruction_assessment.py` | transitional | the CSV a run is stored and compared in | the evaluation harness, which persists a campaign rather than a directory of files |
| `masked_reconstruction_figures.py` | measurement | `figures/masked-reconstruction-*.png` | — |
| `training_loop_report.py` | measurement | [`training-loop.md`](../docs/verification/training-loop.md) | — |
| `corpus_saturation_report.py` | measurement | [`corpus-saturation.md`](../docs/verification/corpus-saturation.md): one budget of steps over a growing share of a corpus, stored run by run and resumable | the evaluation harness, once a campaign of runs is something it persists |
| `corpus_saturation_figures.py` | measurement | `figures/corpus-saturation-*.png` | — |
| `pretraining_curve_report.py` | measurement | [`manual-handoff.md`](../docs/verification/manual-handoff.md): the epochs of an accepted pretraining result as one row per corpus and epoch, and the note's table rendered from that file | the tracker, once a curve is read from it rather than from the result |
| `pretraining_curve_figures.py` | measurement | `figures/pretraining-curve-*.png` | — |
| `excursion_report.py` | measurement | [`corpus-saturation.md`](../docs/verification/corpus-saturation.md): the stored backbones scored on the windows their runs scored, with the excursions read apart from the ordinary tokens; where a side's squared magnitude lies, from the block's values | the evaluation harness, once a stored backbone is something it scores |
| `excursion_figures.py` | measurement | `figures/excursion-*.png` | — |
| `held_out_units.py` | transitional | the units a publication of a corpus whose units differ in kind states, derived once from a published version's own values | the Catalog, if a second corpus ever needs the same derivation |
| `transfer_grid.py` | shared | the grid of the label-efficiency curve as it is stored — its cells, its three CSV files with the run row as the mark that a cell is whole, its task — so the curve report reads a grid without importing torch | the evaluation harness, with the two reports that share it |
| `transfer_modes_report.py` | measurement | the pretrained backbone adapted to the turbofan task, one cell of the grid at a time — a mode at a budget under a seed — stored cell by cell as CSV, resumable, sharded by seed across accelerators and published to the bucket as one archive; the task chosen by name from `KnownTasks` — the turbofan task or the forecasting task of a synthetic pair — with its label scheme, strata and frozen test side stated there | the evaluation harness, once a campaign of runs is something it persists; the task registry moves to the Evaluation process that owns it |
| `label_curve_report.py` | measurement | [`label-efficiency-curve.md`](../docs/verification/label-efficiency-curve.md): the stored cells of the grid read as one curve — a row per cell and seed with its readings, a paired comparison against the control arm per cell with its interval over engines, the trivial predictors — and the note's tables and its one-sentence conclusion rendered from those files by the registered rules | the evaluation harness, once a campaign is scored and judged by the context that owns the statistics |
| `label_curve_figures.py` | measurement | `figures/label-efficiency-curve.png` | — |
| `backbone_comparison_report.py` | measurement | [`label-efficiency-curve.md`](../docs/verification/label-efficiency-curve.md): the rule by which one backbone replaces another — one mode under both, paired on the same engines and windows over the seeds both hold, with its interval over engines | the evaluation harness, once a campaign compares candidates by the context that owns the statistics |
| `transfer_grid_shards.sh` | tooling | the grid run as several shards at once, the seeds dealt over the accelerators and over several processes per accelerator, each shard stored and published on its own; portable bash, so it runs on the notebook platforms as it runs here | the evaluation harness, once a campaign schedules its own runs |
| `reporting.py` | shared | the heading and table shape of every report | — |

## Known debt

- Twenty-one scripts put the repository root on `sys.path` before their imports, and each carries
  a lint exemption (E402) for it in `pyproject.toml`.
- `masked_reconstruction_report.py` is the largest thing here and does three jobs — publishing a
  corpus, diagnosing what a run learnt, and rendering a note. Each has tests; none of them is
  domain logic. It shrinks to composition when the evaluation harness takes the assessment over.
