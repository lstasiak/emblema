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
(`tests/architecture/test_scripts_import_no_tests.py`). The report of the ONNX export needs the
candidates the export suite builds, so it lives with that suite in
`tests/evaluation/adapters/onnx/report.py`.

Logic that a verdict depends on belongs in `src/emblema`, under the architecture rules, types and
coverage, not here. That is why the rules that judge a masked-reconstruction run live in
`emblema.pretraining`, and why the loop that trains one, the state it can be resumed from and the
record of what it did followed them there. What remains below is composition — which corpus, which
experiment — and the shape of a note: its CSV, its tables, its figures.

| Script | Kind | Evidence it produces | Retired by |
| --- | --- | --- | --- |
| `smoke.sh` | tooling | run by the CI job `compose` | — |
| `bootstrap-bucket.sh` | tooling | run by the compose service `bootstrap` | — |
| `fetch_corpora.py`, `utsd_shards.sha256` | tooling | archive checksums for a note; the shard listing pins the collection's revision | — |
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
| `pretext_triviality_report.py` | measurement | [`pretext-variants.md`](../docs/verification/pretext-variants.md): a registered backbone against the trivial baseline of each kind of mask on the corpora it read, tallied per unit and summarised with intervals as CSV, and the note's tables rendered from those files | — |
| `held_out_units.py` | transitional | the units a publication of a corpus whose units differ in kind states, derived once from a published version's own values | the Catalog, if a second corpus ever needs the same derivation |
| `classical_baselines_report.py` | measurement | [`classical-baselines.md`](../docs/verification/classical-baselines.md): MiniRocket against aeon's where no draw differs, and fitted on a published corpus three ways — aeon's, this project's under its own draws and under aeon's — under one grid and ridge; two CSV files, printed as tables in a second step. Run under `uv run --with aeon` | — |
| `bootstrap_calibration_report.py` | measurement | [`verdict-statistics.md`](../docs/verification/verdict-statistics.md): how often the registered percentile interval covers a known reduction and how often it excludes a true zero, over many synthetic datasets at several unit counts; CSV first, table from the CSV | — |
| `frozen_representations.py` | measurement | [`head-and-representation.md`](../docs/verification/head-and-representation.md): every window of a task through the frozen backbone and through an untrained encoder of the same shape, pooled several ways (the mean, the tail of the window, one state per channel), beside the hand-made statistics the trees read; one archive and one index of windows. The only step that touches torch | the evaluation harness, once a pooling is a knob of every network and a probe is a candidate |
| `head_and_representation_report.py` | measurement | [`head-and-representation.md`](../docs/verification/head-and-representation.md): ridge and boosted trees over the stored representations, learning from the same drawn labels a campaign gives a candidate and paired over the same units by the same bootstrap; predictions and fits to CSV, comparisons and tables from those files. Runs without torch, which is what lets the trees' library run at all here | — |
| `campaign_report.py` | measurement | a finished comparison read out of the registry: a row per cell and repeat with its error — or, for a campaign read by the area under the ROC curve, its area and Brier score — a row per comparison with its interval, floor and verdict, and the sentence with a table per budget, for the note of whichever campaign it is pointed at | — |
| `campaign_curve_figures.py` | measurement | the label-efficiency figure drawn from a campaign report's files: the networks' error, the classical baselines beside the control, and the pretrained arms' reduction with its interval and floor; a report read by area is refused | — |
| `campaign_pairs_report.py` | measurement | [`head-and-representation.md`](../docs/verification/head-and-representation.md): the cells of any campaigns read out of the registry into one CSV, a row per cell and unit, and a candidate of one campaign paired against a candidate of another on the units both scored — over (repeat, unit) for selections, whose repeats hold out different units, pooled per unit for comparisons — by the harness's own bootstrap; the reading a campaign cannot make of itself, such as the same arm under a doubled floor of steps; errors per unit only, so a campaign read by area is refused | — |
| `token_view_report.py` | measurement | one window of an intensive-care stay and one of an engine tokenised as pretraining does, every token beside the reading it came from, as CSV | — |
| `token_view_figures.py` | measurement | `docs/images/token-view.png`, the README's picture of the representation | — |
| `reporting.py` | shared | the heading and table shape of every report | — |

## Known debt

- Twenty-one scripts put the repository root on `sys.path` before their imports, and each carries
  a lint exemption (E402) for it in `pyproject.toml`.
- `masked_reconstruction_report.py` is the largest thing here and does three jobs — publishing a
  corpus, diagnosing what a run learnt, and rendering a note. Each has tests; none of them is
  domain logic. It shrinks to composition when the evaluation harness takes the assessment over.
