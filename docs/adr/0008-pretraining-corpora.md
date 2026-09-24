# ADR-0008: Pretraining corpora — which enter the mix, which only host a task, and the budget that bounds the model

- Status: accepted
- Date: 2026-09-11; amended 2026-09-17, 2026-09-18, 2026-09-19 (see Amendments)
- Full text before condensation: commit `5f14447`

## Context

Five corpora were candidates for pretraining a variable-channel encoder, spread over channel
count (8 to 176) and sampling regime (regular machine telemetry against sparse, irregular clinical
and satellite streams). The programme has to fit about 150 free Kaggle T4 GPU-hours. Before a
tokeniser existed we had to settle: whether each corpus is usable (independent units, information,
licence), what each windowing costs, and how large the model may be.

Overlapping windows inflate counts without adding information, so independent units are counted
next to windows, and *unique observed values* next to *tokens processed*. Official test portions
are frozen before anything is counted.

## Decision

- **Arithmetic in code, facts in data.** `scripts/fetch_corpora.py` (sources of record, checksums),
  `corpus_facts.py` (measured counts), `corpus_budget.toml` (every number), `corpus_budget_report.py`
  (the arithmetic, tested). Rows are marked `measured` or `estimate`.
- **Cost model.** FLOPs per window = `6·P·n + 12·n²·d·L`; ±3× band on every GPU-hour. Attention
  is ~41 % of FLOPs on C-MAPSS and ~71 % on SMD, so window length is the first scale knob.
- **Eligibility**: ≥ 20 independent units on the training side, and unique values ≥ 20 × P at the
  reference tier (Hoffmann et al., 2022, as a reference point). An *effective* column credits up
  to four epochs of repetition (Muennighoff et al., 2023).
- **Verdicts (2026-09-11)**:

  | Corpus | Units | Unique values | Verdict |
  |---|---|---|---|
  | C-MAPSS FD001–FD004 | 709 engines | 3.37M | pretraining; hosts the RUL task |
  | SKAB | 35 experiments | 0.37M | small ingredient; anomaly task |
  | SMD (train halves) | 28 machines | 26.9M | pretraining |
  | PhysioNet 2012 set-a | 4,000 stays | 1.73M | irregular ingredient; classification task |
  | ESA-AD | 2 missions | 64.3M (after 30 s binning) | ingredient only, never a single-corpus backbone |

- **The classical corpora are small; the satellite corpus makes the reference model legitimate.**
  Classical corpora hold 32.4M values; with ESA-AD, 96.7M. So corpora are **mixed from the first
  pretraining run**, ESA-AD included. Reference shape d = 256, L = 6 (4.7M parameters); fallback
  d = 192, L = 4 (1.8M).
- **ESA-AD subsampling**: the benchmark's lightweight channels, the first half of each mission
  (the second is the frozen test), at most one observation per channel per 30-second bin with native
  timestamps kept — no grid, which would erase the irregularity. Default window 1 hour. Mission3 is
  measured, not counted.
- **Licences.** All five allow publishing results and a sample. Redistributing a tokenised
  derivative is clear for SKAB (GPL-3.0), PhysioNet (ODC-By 1.0) and ESA-AD (CC BY 3.0 IGO), and
  unclear for C-MAPSS (US Government work, no licence text) and SMD (MIT on the repository, nothing
  on the data). Those two stay in the private store.
- **Split rule**: C-MAPSS test files, PhysioNet set-c, the second half of each ESA mission and SMD
  test halves never enter pretraining.
- **Budget**: about 98 T4-hours at the reference tier; the midpoint fits the cap, the upper band
  does not.

## Consequences

- The tokeniser handles a mix of regular and irregular sources from its first version.
- Corpus versions carry licence terms and a derivative flag, which distribution consults.
- The ESA-AD reader needs pandas and `inflate64`; raw data lives outside version control.

## Alternatives considered

- *Tokens per epoch as eligibility*: counts a value once per overlapping window.
- *Units only*: would admit PhysioNet's 1.7M values for a 4.7M model.
- *ESA-AD on a grid*: destroys the irregularity.

## Amendments

- **2026-09-17 — corrected by the saturation measurement** (ADR-0027,
  `docs/verification/corpus-saturation.md`). At 1.8M parameters C-MAPSS saturates from a quarter of
  its engines, SMD and SKAB are data-limited, ESA-AD's held-out months were not learnt. The sweep
  past the reference shape is closed; Mission3 is not added; the first mixed run needs validation
  stopping, validation per corpus and dropout as a stated parameter.
- **2026-09-18 — every backbone uses the reference shape.** ESA-AD is learnt once the loss is
  bounded (ADR-0028). On the T4 the reference shape costs 0.31 s a step against 0.18 s and ends 11 %
  lower on C-MAPSS held-out; comparing backbones at different shapes would read capacity into every
  cell. The small shape remains the laptop tier's. The sweep reopens only if the mixed run's curve
  is still falling.
- **2026-09-19 — the mixed run does not run out of budget.** Mean relative validation 0.2295 at
  epoch 7, 0.2308 at epoch 8 (ADR-0029); neither budget nor shape grows. The next knob is a weight
  per corpus.
