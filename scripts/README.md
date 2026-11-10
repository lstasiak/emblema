# scripts

Everything here is run by hand, outside the package. Each script is one of three kinds:

- **Tooling** operates the local stack or fetches data. It stays.
- **Measurement reports** produce, on one machine, the numbers and figures a verification note in
  `docs/verification/` records. A report stays while a note relies on it. It is deleted once the
  package takes the same measurement; the note keeps citing the commit that produced its numbers,
  so the old code can still be checked out and rerun.
- **Transitional** scripts stand in for package code that is not written yet, and are deleted by
  the change that writes it.

A new measurement or transitional script gets a row below in the change that adds it, naming what
retires it and a date to review it by: on that date it is deleted, or its row says why it stays. The
note cites the commit, so deleting one loses no evidence.
`tests/architecture/test_scripts_are_registered.py` holds new scripts to this; older ones wait in its
list of exceptions, which can only shrink.

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
| `place_campaign_orders.py` | tooling | defines a batch of campaigns and places their orders from a clean tree | — |
| `colab_cell.py` | tooling | writes the cell or Mac script that fulfils campaign orders on Colab, Kaggle or the Mac, in lanes, resumable, every network step under a limit, and the secrets and card-return cells beside it | — |
| `pretraining_cell.py` | tooling | the same for pretraining orders, on Colab, Kaggle or the Mac: resumes from the last checkpoint a log names, one or several orders a card | — |
| `fetch_published.py` | tooling | fetches blocks and task labels from the store before a run's processes need them | — |
| `cell_parts.py` | tooling | what those cells share: the platforms, the checkout and install under limits, the rules a cell keeps | — |
| `accept_lane_results.py` | tooling | accepts a cell's pasted results, one per lane | — |
| `campaign_pairs_fanout.py` | tooling | reads a plan of pairs across processes, the tables byte for byte those of one run | — |
| `detection_side_readings.py` | measurement | a detection task's readings beside the area, per cell | — |
| `taught_fine_tuning_pilot.py` | measurement | [`taught-fine-tuning.md`](../docs/verification/taught-fine-tuning.md): fine-tuning on a few taught events under a pretrained encoder and its initialisation, read on months the encoders never read | the later of the supervised reference on telemetry and the descriptive reading on fresh pretrainings, or the change that brings this fine-tuning into the package; review by 2027-02-01 |
| `taught_fine_tuning_diagnostics.py` | measurement | [`taught-fine-tuning.md`](../docs/verification/taught-fine-tuning.md): the control, the curve of lengths, every event taught, the classes taught, and checks of those readings after the run | the reading of the supervised reference on telemetry; review by 2027-02-01 |
| `telemetry_supervised_reference.py` | measurement | [`telemetry-supervised-reference.md`](../docs/verification/telemetry-supervised-reference.md): the same network trained on every event of the first mission's fitting side, its rate and length chosen on the later fitting months, read on the validation months against the frozen direction and the trees taught the same | the change that brings a supervised detector into the package, or the end of the stage over telemetry; review by 2027-02-01 |
| `telemetry_context_report.py` | measurement | [`telemetry-context.md`](../docs/verification/telemetry-context.md): the first mission's detectors at the encoder's initialisation over a window of several hours ending at the scored hour, the cells derived from the published one-hour cells, chosen on later fitting months and read on the validation months against the one-hour control | the change that brings the context window into the package, or the end of the stage over telemetry; review by 2027-02-01 |
| `telemetry_ceiling_report.py` | measurement | [`telemetry-ceiling.md`](../docs/verification/telemetry-ceiling.md): which validation events of the first mission the departures, any other detector with no label and a taught detector find, against rotations of the scores within months, and the unfound events' annotated channels at ten minutes against the hour; CSV first, the tables from the CSV | the ceiling's measured section in the note; review by 2027-02-09 |
| `telemetry_channel_local_report.py` | measurement | [`telemetry-channel-local.md`](../docs/verification/telemetry-channel-local.md): the first mission read through each channel's departure from its own recent hours, with no model and taught, against the states' direction and the cells' bar, over the context report's cells and states; its answers are what the package's detector was accepted against (ADR-0075) | the reading of a telemetry pretext against the departures, the bar its taught arms set, or the end of the stage over telemetry; review by 2027-02-01 |
| `telemetry_pretext_input_report.py` | measurement | [`telemetry-pretext-input.md`](../docs/verification/telemetry-pretext-input.md): the hidden hour's forecast from its history by cell width and window length on the fitting months, with no label, against the median of the five hours before; the yearly structure of the channels' errors, the held-out block, and the small shape's cost per layout | the telemetry pretext's reading, or the end of the stage over telemetry; review by 2027-02-01 |
| `telemetry_history_window_check.py` | measurement | [`telemetry-history-window.md`](../docs/verification/telemetry-history-window.md): one month of a mission cut into 48-hour windows whole against the same hours' windows concatenated and their cells joined, and the host's windows a second either way | the telemetry pretext's reading, or the end of the stage over telemetry; review by 2027-02-01 |
| `history_pretext_answers_report.py` | measurement | [`telemetry-history-pretext.md`](../docs/verification/telemetry-history-pretext.md): a forecasting pretext's answer for each channel's hidden last hour against the last reading, the median of the hours before and the hour one period back, per time axis, paired and resampled by window | the telemetry pretext's reading in the package, or the end of the stage over telemetry; review by 2027-02-01 |
| `history_pretext_reading_pilot.py` | measurement | [`telemetry-history-pretext.md`](../docs/verification/telemetry-history-pretext.md): a forecasting pretext's error as a detector, read by the package's own runtimes on the fitting side's later months before any card, over every scored month and over those the pretraining never read | the pretext's reading on the validation months, or the end of the stage over telemetry; review by 2027-02-01 |
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
| `cell_pretext_audit.py` | measurement | [`cell-pretext.md`](../docs/verification/cell-pretext.md): for each corpus a backbone read, the hidden cells of a pretext over cells answered without a model — carried, interpolated, from the other channels, from the schedule alone and stacked — as sums per kind, target and answer in CSV, and the note's tables rendered from them | — |
| `pretraining_budget_report.py` | measurement | [`cell-pretraining.md`](../docs/verification/cell-pretraining.md): how many epochs a recipe trains, read off one run's log — each term's validation cost over the trivial predictor's per epoch in CSV, the epoch each term stops learning by a declared plateau rule, and the epoch the run keeps beside it | — |
| `pretraining_screen_report.py` | measurement | [`cell-pretraining.md`](../docs/verification/cell-pretraining.md): the declared reading of a screen of pretraining recipes over the pairs `campaign_pairs_report.py` compared — fine-tuning's spread between pretrainings, each gain's threshold and level, the contrasts between recipes and whether they move the recipe, and the null pairs of the network from nothing | — |
| `rate_choice_report.py` | measurement | [`cell-pretraining.md`](../docs/verification/cell-pretraining.md): the rate an arm keeps on a fixed half — each rate's mean area over every repeat against the rate in force, its standard error from one draw of stays shared by every budget, seed and backbone | — |
| `transduction_report.py` | measurement | [`cell-pretraining.md`](../docs/verification/cell-pretraining.md): fine-tuning's gain over the network from nothing on the scored stays a backbone read in pretraining and on those it never read, per backbone and as a mean with its interval | — |
| `held_out_units.py` | transitional | the units a publication of a corpus whose units differ in kind states, derived once from a published version's own values | the Catalog, if a second corpus ever needs the same derivation |
| `classical_baselines_report.py` | measurement | [`classical-baselines.md`](../docs/verification/classical-baselines.md): MiniRocket against aeon's where no draw differs, and fitted on a published corpus three ways — aeon's, this project's under its own draws and under aeon's — under one grid and ridge; two CSV files, printed as tables in a second step. Run under `uv run --with aeon` | — |
| `bootstrap_calibration_report.py` | measurement | [`verdict-statistics.md`](../docs/verification/verdict-statistics.md): how often the registered percentile interval covers a known reduction and how often it excludes a true zero, over many synthetic datasets at several unit counts; CSV first, table from the CSV | — |
| `frozen_representations.py` | measurement | [`head-and-representation.md`](../docs/verification/head-and-representation.md): every window of a task through the frozen backbone and through an untrained encoder of the same shape, pooled several ways (the mean, the tail of the window, one state per channel), beside the hand-made statistics the trees read; one archive and one index of windows. The only step that touches torch | the evaluation harness, once a pooling is a knob of every network and a probe is a candidate |
| `head_and_representation_report.py` | measurement | [`head-and-representation.md`](../docs/verification/head-and-representation.md): ridge and boosted trees over the stored representations, learning from the same drawn labels a campaign gives a candidate and paired over the same units by the same bootstrap; predictions and fits to CSV, comparisons and tables from those files. Runs without torch, which is what lets the trees' library run at all here | — |
| `campaign_report.py` | measurement | a finished comparison read out of the registry: a row per cell and repeat with its error — or, for a campaign read by the area under the ROC curve, its area and Brier score — a row per comparison with its interval, floor and verdict, and the sentence with a table per budget, for the note of whichever campaign it is pointed at | — |
| `campaign_curve_figures.py` | measurement | the label-efficiency figure drawn from a campaign report's files: the networks' error, the classical baselines beside the control, and the pretrained arms' reduction with its interval and floor; a report read by area is refused | — |
| `campaign_pairs_report.py` | measurement | [`head-and-representation.md`](../docs/verification/head-and-representation.md), [`layer-readout.md`](../docs/verification/layer-readout.md): the cells of any campaigns read out of the registry into CSV — a row per cell and unit, every window's answer for a campaign read by area, and every cell scored whole in its measure — and a candidate of one campaign paired against a candidate of another on the units both scored — over (repeat, unit) for selections, whose repeats hold out different units, pooled per unit for comparisons, by area where a campaign is read so — by the harness's own bootstrap; the reading a campaign cannot make of itself, such as the same arm under a doubled floor of steps or under another backbone | — |
| `detection_power_report.py` | measurement | [`anomaly-tasks.md`](../docs/verification/anomaly-tasks.md): how often the detection rule confirms a known gain in event-weighted area, from each task's validation events and windows read off the raw annotations (`structure`) and a model of scores over them (`simulate`); [`telemetry-ceiling.md`](../docs/verification/telemetry-ceiling.md): the same for the events found at a share of false alarms, over models (`recall`, `jitter`) and over real pairs with an injected gain (`injected`); CSV first, the table from the CSV | — |
| `comparison_power_report.py` | measurement | [`verdict-statistics.md`](../docs/verification/verdict-statistics.md): how often the replacement rule replaces a known gain, for named pairs of exported answers at every budget they share, from the parts of the variance a seed and the stays scored carry; CSV first, tables from the CSV | — |
| `stage_reader_figures.py` | measurement | `figures/stage-reader-*.png`: the recipes' curve over the budgets of labels from the scored cells, and the rule's power over the budgets from the power file | — |
| `token_view_report.py` | measurement | one window of an intensive-care stay and one of an engine tokenised as pretraining does, every token beside the reading it came from, as CSV | — |
| `token_view_figures.py` | measurement | `docs/images/token-view.png`, the README's picture of the representation | — |
| `reporting.py` | shared | the heading and table shape of every report | — |

## Known debt

- Thirty-six scripts put the repository root on `sys.path` before their imports, and each carries
  a lint exemption (E402) for it in `pyproject.toml`.
- `masked_reconstruction_report.py` is the largest thing here and does three jobs — publishing a
  corpus, diagnosing what a run learnt, and rendering a note. Each has tests; none of them is
  domain logic. It shrinks to composition when the evaluation harness takes the assessment over.
