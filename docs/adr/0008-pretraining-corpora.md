# ADR-0008: Pretraining corpora — which enter the mix, which only host a task, and the budget that bounds the model

Status: **accepted** (2026-09-11), on counts measured from the raw files on the development Mac.
The parameter budget is corrected once more by the corpus-saturation measurement that precedes the
first full pretraining run.

## Context

Five corpora were candidates for pretraining a variable-channel encoder, spread over two axes of
heterogeneity: channel count (8 to 176) and sampling regime (regular machine telemetry against
sparse, irregular clinical and satellite streams). The whole programme — pretraining, a transfer
matrix and label-budget campaigns — has to fit the free GPU quota of a Kaggle T4, of the order of
150 GPU-hours in total.

Three things had to be settled before a tokeniser existed, because each one shapes it:

- **Whether each corpus is usable for pretraining at all**: enough independent units to split on,
  enough information to teach a model of the intended size, and a licence that allows the results, a
  sample in the repository and — separately — a redistributed tokenised derivative.
- **How much each realistic windowing costs**, so the later choice of attention mechanism and window
  length picks from priced scenarios instead of forcing a recount.
- **How large the model may be**, given the data that actually exists.

Two silent failure modes drive the counting rules. Overlapping windows inflate the sample count
without adding information: a hundred thousand windows from ten machines are still ten machines,
and a value seen by ten overlapping windows is still one value. So independent units are counted
next to windows, and *unique observed values* are counted next to *tokens processed*. And splitting
after windowing leaks neighbouring windows across the boundary, so every verdict names the split at
unit level and the official test portions are frozen before anything is counted.

## Decision

### The arithmetic is code; the facts are data

Four scripts under `scripts/` carry the spike, so the numbers can be regenerated and corrected:

| Script | Role |
|---|---|
| `fetch_corpora.py` | downloads each corpus from its source of record (never a mirror), pins GitHub-hosted corpora to a commit, records SHA-256, verifies the publisher's MD5 where one exists, resumes interrupted transfers, unpacks Deflate64 members that the standard library refuses |
| `corpus_facts.py` | counts units, channels, observed values, and the exact windows and tokens per declared window variant; prints a `[corpora.<key>.measured]` TOML block |
| `corpus_budget.toml` | every number: cost model, thresholds, tiers, corpora with licence facts, literature estimates and measured facts, window variants, the backbone programme, the campaigns, the parameter sweep |
| `corpus_budget_report.py` | the arithmetic, printed as markdown for the verification note; known-answer tests in `tests/scripts` |

The report marks each row `measured` or `estimate`, so the note never hides which numbers came from
a paper and which from the files. The five candidates are measured; the two substitutes are priced
from the literature.

### Cost model

FLOPs per window = `6 · P · n + 12 · n² · d · L`, with `P = 12 · d² · L` for a transformer encoder
of width `d` and `L` layers. Both terms count a multiply-add as two operations and the backward pass
as twice the forward one: 2 × 3 = 6 per parameter per token in the dense layers; attention scores
and their weighted sum are 2 × 2 × n² · d per layer forward, hence 12 · n² · d · L. Sustained
throughput: 10 TFLOP/s for a T4 in fp16, 1.25 TFLOP/s for an M1 Pro through MPS in fp32,
50 TFLOP/s for an A100 in bf16 (the ratio at which the paid platform prices it against the T4).
Every GPU-hour is quoted with a ±3× band.

Where the quadratic term bites is itself a result: at the default windows attention is about 41 % of
the FLOPs for C-MAPSS (1,050 tokens) and about 71 % for SMD (3,800 tokens). Window length is
therefore the first scale knob, and SMD is where it matters.

One known bias, accepted: the quadratic term is evaluated at the *mean* tokens per window. For
irregular corpora the windows differ in density, and the mean of n² exceeds the square of the mean,
so their attention cost is understated — by roughly 15 % for a threefold density spread. That sits
well inside the ±3× band; the measured window counts, not this term, are what the saturation
measurement corrects.

### Eligibility thresholds

Two criteria, both data in the TOML:

1. **≥ 20 independent units on the training side**, so a unit-level train/validation split leaves
   something on each side and the count of units — not windows — bounds generalisation.
2. **Unique observed values ≥ 20 × P** at the reference tier (d = 256, L = 6, P = 4.7M, so 94M
   values). A value is one channel at one time; overlapping windows do not multiply it. The ratio
   is Chinchilla's (Hoffmann et al. 2022), used as a reference point rather than a law: it was
   fitted to language models trained for one epoch. Because these corpora are trained for many
   epochs, the report also prints an *effective* count that credits up to four epochs of repetition
   as fresh data — beyond that repeated data adds little (Muennighoff et al. 2023) — and that is the
   column the saturation measurement will calibrate.

A corpus that fails the units criterion never becomes a single-corpus backbone. A corpus that fails
the value criterion is data-starved alone; whether it stays in the mix is decided per corpus below.

### Verdicts (measured 2026-09-11)

| Corpus | Units | Unique values (pretraining side) | Units ok | Values ok | Effective per P | Verdict |
|---|---|---|---|---|---|---|
| C-MAPSS FD001–FD004 (train) | 709 engines | 3.37M | yes | no | 3 | pretraining — primary corpus, hosts the RUL task |
| SKAB | 35 experiments | 0.37M | yes | no | 0 | pretraining as a small ingredient; anomaly task corpus; its own backbone is a data-starved control |
| SMD (train halves) | 28 machines | 26.9M | yes | no | 23 | pretraining — four fifths of the classical values; data licence unstated |
| PhysioNet 2012 (set-a) | 4,000 stays | 1.73M | yes | no | 1 | pretraining as the irregular ingredient; classification task |
| ESA Anomaly Dataset | 2 missions (3 measured) | 64.3M of 87.1M in the lightweight channels after 30-second binning | no | no | 55 | **ingredient**: enters the mix, never a single-corpus backbone |
| PSM (substitute) | 1 pooled unit | 3.3M | no | no | 3 | candidate, priced from the literature |
| Tennessee Eastman (substitute) | 10,500 simulated runs | 273M | yes | yes | 231 | candidate, priced from the literature; simulated trajectories of one process, not independent machines |

Three verdict classes, encoded in the budget file: `pretraining` corpora enter the mix and get a
backbone of their own for the transfer matrix; an `ingredient` enters the mix only; `downstream-only`
corpora would host a task and never be pretrained on (none today). The measured counts confirmed the
literature estimates within a few percent for the classical corpora (largest deviation: SKAB, +21 %,
because the anomaly-free file is eight times longer than an experiment) and exceeded them for ESA
(+29 %).

The headline: **the classical corpora are small, and the satellite corpus is what makes the
reference model legitimate.** The four classical corpora hold 32.4M unique values together — enough
for a 1.6M-parameter model by the strict criterion. With ESA-AD in the mix the count is 96.7M,
which clears the strict criterion for the reference model with nothing to spare (20 values per
parameter) and gives 82 per parameter with four epochs of repetition credited. Hence:

- **Corpora are mixed from the first pretraining run**, not only in the transfer stage, and the mix
  includes ESA-AD from the start. This moves the multi-source evidence earlier rather than
  weakening it.
- **Parameter budget**: d = 256, L = 6 (4.7M) as the reference tier, justified strictly by the
  five-corpus mix; d = 192, L = 4 (1.8M) as the fallback should ESA-AD drop out or the mix saturate
  early. Growing to d = 320, L = 8 (9.8M) fails the strict criterion (10 per parameter) and needs
  the saturation measurement to show the mix is not exhausted at 4.7M.
- **Tokens processed per epoch** (230M for the mix at the default windows, 5.0G over the configured
  epochs) drive the compute budget, not the eligibility: the two numbers are reported side by side
  and never substituted for one another.

### ESA Anomaly Dataset: subsampling strategy

Version 2 of the Zenodo record has three missions and 11.6 GB compressed (the benchmark paper
describes two missions and 7.3 GB; Mission3 is published but excluded from the benchmark for
communication gaps, invalid segments and trivial anomalies). The corpus is measured whole, but the
pretraining strategy is:

- **Channels**: the benchmark's lightweight subsets — Mission1 channels 41–46 (subsystem 5),
  Mission2 channels 18–28 (subsystem 1); Mission3 by target channels if it is ever included, and it
  is measured but not counted by default.
- **Period**: the first half of each mission by time, which is also the benchmark's training
  portion; the second half is the frozen test set.
- **Decimation**: at most one observation per channel per 30-second bin, keeping the native
  timestamps of the kept observations. This bounds the volume without resampling onto a grid — a
  grid would erase the very irregularity that makes the corpus worth having on the second axis.
- **Windows**: 1-, 3- and 6-hour windows priced; 1 hour (839 tokens on average) is the default. A
  window that falls into a communication gap holds nothing and is not counted.

What the files showed: Mission1's lightweight channels run at a median spacing of 30 s, so the
binning removes only 5 % of their 46.3M training-half observations; Mission2's run at 18 s and are
halved (40.8M to 20.2M). Mission3's 24 target channels hold 184M observations over 4 training-half
years (92M after binning) — a reserve of the same order as the whole counted mix, should the
benchmark authors' reservations about it ever be outweighed. Two or three missions are two or three
independent units, so ESA-AD is an ingredient of the mix, never a standalone backbone; the transfer
matrix treats it as a target corpus.

### Licences

All five candidates permit publishing results and keeping a miniature sample in the public
repository. Redistribution of a tokenised derivative differs:

| Corpus | Terms as read on 2026-09-11 | Derivatives |
|---|---|---|
| C-MAPSS | no licence text; a NASA work of the U.S. Government (not subject to copyright in the U.S.); acknowledgement requested | unclear |
| SKAB | GPL-3.0 at the repository root; data files inside the same repository | yes, copyleft |
| SMD | MIT at the repository root; the data folder sits inside it, with no statement about the data, "collected from a large Internet company" | unclear |
| PhysioNet 2012 | Open Data Commons Attribution 1.0, open access | yes |
| ESA-AD | CC BY 3.0 IGO | yes |
| PSM (substitute) | CC BY 4.0 for the data files, BSD-3 for the code | yes |
| Tennessee Eastman (substitute) | public-domain dedication on Harvard Dataverse | yes |

A corpus without a licence text is not rejected when the legal basis is clear without one: C-MAPSS
is a work of the U.S. Government, and SMD's files are covered by the MIT licence of the repository
that ships them. What stays unclear in both cases is the right to redistribute a derivative, so
tokenised C-MAPSS and SMD remain in the private artifact store and the public distribution of a
processed corpus is limited to the corpora marked `yes`. SMD stays in pretraining on that basis;
the two substitutes remain priced should its licence ever be judged insufficient.

### Split rule

Official test portions are frozen before any counting and never enter pretraining: the C-MAPSS test
files, PhysioNet set-c (set-b is validation), the second half of each ESA mission, the SMD test
halves. Where no official split exists (SKAB), units are held out by the task definition.

### Budget against the cap

At the reference tier the programme prices at about 98 T4-hours: 46 for nine backbones (four
single-corpus, mixed, two leave-one-corpus-out, mixed with channel dropout), 22 for 405 fine-tuning
runs, and a 30-hour allowance for ablations and distillation. The midpoint is inside the 150-hour
cap; the upper edge of the ±3× band (295 hours) is not. The knob identified now is the SMD window:
`w50s10` cuts its attention term by three quarters at the same number of tokens per epoch.

The smallest programme that still yields a publishable preliminary result — one C-MAPSS backbone
and one label-efficiency curve of five budgets and three seeds — costs about 2 T4-hours, or one
evening on the laptop's MPS at the small tier.

## Consequences

- The tokeniser is built for a mixed corpus of four regular and two irregular sources from its
  first version; single-corpus pretraining is the exception, not the rule.
- The encoder architecture picks its window length from the priced variants; the report is rerun,
  not recomputed, when it does.
- The catalog's corpus versions carry these licence terms and the derivative flag; the public
  distribution step consults the flag.
- The ESA-AD reader will need pandas: channels ship as pickled DataFrames inside zip files, written
  by a 2.x pandas, and the mission archives hold Deflate64 members (a metadata file per mission)
  that the standard library's `zipfile` refuses — the fetch script decompresses them with
  `inflate64`. Both dependencies — and the pickle-compatibility risk — are decided when the reader
  is written, not here.
- Raw data lives under `data/raw/<corpus>/` on the development Mac, outside version control; the
  sources of record, commits and checksums are in `fetch_corpora.py` and the verification note.
- The budget table is compared with actual GPU consumption at the end of the project.

## Alternatives considered

- **Tokens processed per epoch as the eligibility criterion.** Rejected: it counts a value once per
  overlapping window, so C-MAPSS at a stride of 5 looks eight times richer than it is — the very
  trap the counting rules exist to avoid. Kept as the compute figure it is.
- **A hand-filled table in the notes.** Rejected: the saturation measurement corrects these numbers
  and the end-of-project comparison needs the formulas, not the digits.
- **Tokens seen over all epochs as the eligibility criterion.** Rejected as the primary criterion,
  because repetition has diminishing returns beyond a few epochs; kept as the capped *effective*
  column so the saturation measurement can promote it if the mix turns out not to saturate.
- **Units only.** Rejected: it would admit PhysioNet's 1.7M values as a standalone pretraining
  corpus for a 4.7M-parameter model.
- **ESA-AD as a candidate until the transfer stage.** Rejected by the measurement: without its 64M
  values the reference model fails the strict criterion threefold, and the corpus is the second
  point on the irregular axis the thesis needs.
- **Resampling ESA-AD onto a 30-second grid**, as the benchmark's own preprocessing does. Rejected
  for pretraining: it destroys the irregularity; binning with native timestamps bounds the volume
  and keeps it.
- **Estimating ESA-AD from metadata without downloading it.** Rejected, and the measurement proved
  the point: the estimate was 29 % low because per-channel native rates differ by mission.
- **Rejecting corpora without a licence text outright.** Rejected: the primary corpus would fall
  with them, while the legal basis for results and samples is clear; the restriction lands where
  the uncertainty is, on derivatives.
- **Counting empty time windows.** Rejected: a window in a communication gap costs no compute and
  teaches nothing; counting it would inflate window counts for ESA-AD and shrink the mean tokens per
  window, hiding the real window length from the architecture decision.

## Revisit when

- The saturation measurement shows the mixed corpus not saturating at 4.7M parameters: move to the
  next shape in the sweep, adding Mission3 if the value criterion demands it — or saturating early:
  fall back to d = 192, L = 4.
- Measured facts and actual training consumption differ beyond the ±3× band.
- SMD's data licence is clarified either way, or a substitute is chosen.
- A corpus is added: it is a new entry in the TOML, not a change to the arithmetic.

## Sources

- Saxena, A. and Goebel, K. (2008). Turbofan Engine Degradation Simulation Data Set. NASA
  Prognostics Data Repository.
- Katser, I. D. and Kozitsin, V. O. (2020). Skoltech Anomaly Benchmark (SKAB).
- Su, Y. et al. (2019). Robust Anomaly Detection for Multivariate Time Series through Stochastic
  Recurrent Neural Network. KDD.
- Silva, I. et al. (2012). Predicting In-Hospital Mortality of ICU Patients: The
  PhysioNet/Computing in Cardiology Challenge 2012.
- Kotowski, K. et al. (2024, v2 2025). European Space Agency Benchmark for Anomaly Detection in
  Satellite Telemetry. arXiv:2406.17826. Data: doi:10.5281/zenodo.15237121.
- Abdulaal, A. et al. (2021). Practical Approach to Asynchronous Multivariate Time Series Anomaly
  Detection and Localization. KDD.
- Rieth, C. A. et al. (2017). Additional Tennessee Eastman Process Simulation Data for Anomaly
  Detection Evaluation. Harvard Dataverse, doi:10.7910/DVN/6C3JR1.
- Hoffmann, J. et al. (2022). Training Compute-Optimal Large Language Models. arXiv:2203.15556.
- Muennighoff, N. et al. (2023). Scaling Data-Constrained Language Models. arXiv:2305.16264.
