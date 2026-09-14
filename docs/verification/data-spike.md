# Data spike: corpus arithmetic and licences

Purpose: before a tokenizer exists, establish for each candidate corpus how many independent units
and tokens it offers under realistic windowing, what its licence permits, and how many GPU-hours
the pretraining and evaluation programme costs at each compute tier. The decisions live in
ADR-0008; this note records what was read, what was counted, where, and with which versions.

Method, on the machine that holds the data (the development Mac):

```sh
uv run scripts/fetch_corpora.py                       # ~11.9 GB, mostly satellite telemetry; resumable
uv run scripts/corpus_facts.py                        # classical corpora: units, windows, tokens as TOML
uv run --with "pandas<3" scripts/corpus_facts.py esa_ad   # the channels are pickles from a 2.x pandas
uv run scripts/corpus_budget_report.py                # the tables, from scripts/corpus_budget.toml
```

The `[corpora.<key>.measured]` blocks printed by the second and third command are pasted into
`scripts/corpus_budget.toml`; the fourth command then prints the tables below with `measured` in the
basis column instead of `estimate`. The licence check needs no data and was done from the sources
of record.

## 2026-09-11 — licences, read at the sources of record (Windows, network only)

| Corpus | Read | What it says | Conclusion |
|---|---|---|---|
| C-MAPSS | data.nasa.gov dataset page; PCoE repository page | "License not specified"; access level public; PCoE: "Users employ the data at their own risk", publications "are requested to acknowledge" the repository | results and a sample: yes (U.S. Government work); derivatives: unclear — no licence text to point at |
| SKAB | `LICENSE` at the repository root (GPL-3.0 text); README badge "GPL v3.0" | one licence for the whole repository, data folder included; no separate data statement | results, sample, derivatives: yes, under GPL-3.0 (copyleft) |
| SMD | `LICENSE` at the repository root (MIT, "Copyright (c) 2021 NetManAIOps-OmniAnomaly"); README | MIT for the repository, data folder inside it; README: "We collected it from a large Internet company"; no data statement | results and a sample: yes; derivatives: unclear — MIT covers the files, the authors' right to license the data is unverifiable |
| PhysioNet 2012 | project page and licence page | Open Data Commons Attribution License v1.0; open access; share, adapt and derivative databases with attribution | yes on all three |
| ESA Anomaly Dataset | Zenodo record 15237121 (v2, 2025-04-17) and its API | CC BY 3.0 IGO; three files, 11.6 GB, MD5 per file | yes on all three |
| PSM (substitute) | `README` and `data/LICENSE` of eBay/RANSynCoders | data files CC BY 4.0, code BSD-3 | yes on all three |
| Tennessee Eastman (substitute) | Harvard Dataverse API for doi:10.7910/DVN/6C3JR1 | "Public Domain Dedication with disclaimer"; four RData files, 1.4 GB | yes on all three |

Sources of record and versions pinned by `fetch_corpora.py`: NASA archive
`phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`
(12,429,152 bytes on 2026-09-11; the `data.nasa.gov/docs/legacy/CMAPSSData.zip` link redirects to a
presigned URL that rejects `HEAD`); SKAB at commit `b2c0d46c` (master, 2024-08-11); OmniAnomaly at
commit `7fb0e0ac` (master, 2021-02-12); PhysioNet `set-a.zip` 7,938,449 bytes, `set-b.zip`
7,958,979 bytes, `set-c.tar.gz` 6,600,293 bytes (no `set-c.zip` exists); Zenodo v2 files
3,776,246,073 / 4,098,539,932 / 3,734,403,444 bytes with MD5 in the script.

Two facts the published descriptions do not state: the Zenodo record has **three** missions in 11.6 GB (the benchmark
paper describes two, 7.3 GB; Mission3 is excluded from the benchmark for gaps, invalid segments and
trivial anomalies), and the channels are stored as **pickled pandas DataFrames inside zip files**,
one per channel, with `channels.csv`, `labels.csv`, `anomaly_types.csv` and `telecommands.csv`
alongside — so the reader will need pandas.

## 2026-09-11 — estimate-only report (Windows, before any download)

Before anything was downloaded the same script priced every corpus from the literature: figures
from the papers and dataset descriptions cited in `corpus_budget.toml`, every row `estimate`. Those
tables are not kept here. They cannot be regenerated once the `measured` blocks sit in the file,
and the measured section below repeats all but one of them almost unchanged; the per-corpus error
is quoted there in a line. What is kept is where the prediction would have led somewhere else.

Two differences carried a decision. The GPU-hour totals barely moved — tier M 94.6 h to 98.4 h, S
431.5 h to 432.6 h, L 23.6 h to 25.5 h, all far inside the ±3× band and under the 150 h cap. The
parameter budget flipped: on the estimated mix of 32.4M unique values no shape passed the strict
threshold, the 4.72M reference model sat at 7 unique values per parameter and could be defended
only on the effective count (27 per P), with d=192, L=4 as the fallback. The measured 96.7M values
carry the same model over the strict threshold at 20 per P, on the nose. Both moves come from one
place: the estimate put ESA-AD at ~50M values after subsampling, the count found 64.3M.

The table below is the one that changed a verdict. ESA-AD stood at `candidate` pending the measured
count; two independent units kept it out of a backbone of its own either way, but the values it
brought promoted it to `ingredient` of the mix.

### Eligibility (estimate)

Thresholds: ≥ 20 training units; unique observed values ≥ 20 × P at tier M (= 94.4M). Overlapping windows do not multiply the count. The last column credits up to 4 epochs of repetition as fresh data.

| Corpus | Units | Units ok | Unique values | Unique ok | Effective per P | Verdict | Decision |
|---|---|---|---|---|---|---|---|
| C-MAPSS FD001–FD004 | 709 | yes | 3.36M | **no** | 3 | pretraining | Tentative: primary corpus, enters the mix and hosts the RUL task; 3.4M unique values make it data-starved on its own. |
| SKAB | 35 | yes | 308k | **no** | 0 | pretraining | Tentative: enough units, far too few tokens; anomaly task corpus and a small ingredient of the mix. |
| SMD | 28 | yes | 26.9M | **no** | 23 | pretraining | Tentative: the largest classical source, four fifths of the mix's unique values; the licence of the data itself is the open question. |
| PhysioNet 2012 | 4,000 | yes | 1.8M | **no** | 2 | pretraining | Tentative: the only classical corpus on the irregular axis, so it stays in the mix despite very few tokens. |
| ESA Anomaly Dataset | 2 | **no** | 50M | **no** | 42 | candidate | Tentative: two independent units, so an ingredient at most; its ~50M values after subsampling would more than double the mix — decision after the measured count. |
| PSM | 1 | **no** | 3.31M | **no** | 3 | candidate | Clear licence and the same role as SMD, but one pooled unit: it can only ever be a mixing ingredient. |
| Tennessee Eastman (Rieth et al.) | 10,500 | yes | 273M | yes | 231 | candidate | Public domain and an order of magnitude more tokens than SMD; simulated like C-MAPSS, so its runs are trajectories of one process model, not independent machines. |

## 2026-09-11 — measured on the development Mac (macOS arm64, M1)

Method as above, all five corpora in one pass of `uv run --with "pandas<3" scripts/corpus_facts.py`
after `fetch_corpora.py` had brought the files in. Two things broke on first contact and were fixed
on the same branch before the counts were taken: the ESA mission archives hold Deflate64 members
(compression method 9, one metadata file per mission) that `zipfile` refuses in the middle of
`extractall`, so the fetch script now decompresses those members itself with `inflate64`; and a
re-run re-hashed 11.6 GB, so checksums are now remembered in `.digests.json` beside the archives
(`--recheck` distrusts the note).

Archive sizes on disk, from the measured blocks:

| Corpus | Bytes |
|---|---:|
| C-MAPSS | 12,429,152 |
| SKAB | 5,419,930 |
| SMD | 105,339,616 |
| PhysioNet 2012 | 22,735,280 |
| ESA-AD (three missions) | 11,609,189,449 |

Checksums as printed by the second run of `uv run scripts/fetch_corpora.py` (data root
`data/raw`; every archive already downloaded and unpacked; the three Zenodo files match the
publisher's MD5):

| Corpus | File | Bytes | SHA-256 | Publisher MD5 |
|---|---|---:|---|---|
| cmapss | turbofan-engine-degradation-simulation.zip | 12,429,152 | `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2` | none published |
| skab | SKAB-b2c0d46c2971.zip | 5,419,930 | `45ac11b460e495ba2c1301c3f8e871b688b5ecfa6cd0770b4225594fe45efc80` | none published |
| smd | OmniAnomaly-7fb0e0acf89e.zip | 105,339,616 | `5bcce77823a9ed2ea872c733895ccb6a7edb40476306ab8626b49481803b831e` | none published |
| physionet2012 | set-a.zip | 7,938,449 | `1433638192fda622e8bb11a4559b8af2dbb1c69f0e1c3b5b606cc12aabd0a4c5` | none published |
| physionet2012 | set-b.zip | 7,958,979 | `54094c39631797af554ad4233164e004a699521351bd2f2bed50d6564ebc56dc` | none published |
| physionet2012 | set-c.tar.gz | 6,600,293 | `a4a56b95bcee4d50a3874fe298bf2998f2ed0dd98a676579573dc10419329ee1` | none published |
| physionet2012 | Outcomes-a.txt | 79,219 | `2613ea60ccda29f87571a7d6b09ad130858a8ccd66325bd073365925026883c2` | none published |
| physionet2012 | Outcomes-b.txt | 79,149 | `1305f29734da374a809483229a99f33a9230aa8d4fa74b1a2d6ad788136423d5` | none published |
| physionet2012 | Outcomes-c.txt | 79,191 | `0b32dcb7753c0f18f2f5fde498a28618b8879c5e527ce0dae8ad120dd7e205c9` | none published |
| esa_ad | ESA-Mission1.zip | 3,776,246,073 | `ba28f761b1deab4dbba4728793bff139fea39dbf9cf0d9c559d619ffe75d5a72` | ok |
| esa_ad | ESA-Mission2.zip | 4,098,539,932 | `e8a89be1917b6754a10bd323441e87a82c8cf2e84ed162442c2dcf72ecc346d5` | ok |
| esa_ad | ESA-Mission3.zip | 3,734,403,444 | `f426c4bb9857299c586e3c50f35ddb58b469a3e5565f586d4095bcb6cf532404` | ok |

These digests are the provenance the catalog's corpus versions will record; the Zenodo byte counts
match the record's file listing exactly.

Facts counted, pasted into `scripts/corpus_budget.toml` as `[corpora.<key>.measured]`:

| Corpus | Units | Channels | Observed values (pretraining side) | What the files showed |
|---|---:|---:|---:|---|
| C-MAPSS | 709 | 21 | 3,367,539 | 100 / 260 / 100 / 249 training engines in FD001–FD004; 128 / 207 / 543 cycles per engine (min / median / max) |
| SKAB | 35 | 8 | 374,448 | 745 / 1,141 / 9,405 rows per file; the anomaly-free file is eight times an experiment; 1 s spacing |
| SMD | 28 | 38 | 26,919,390 | groups of 8 / 9 / 11 machines; 23,687 / 23,702 / 28,743 rows per machine |
| PhysioNet 2012 | 4,000 | 37 | 1,733,980 | 0 / 422 / 1,496 observations per stay — three stays in set-a carry descriptors only; 5 timeless descriptors per stay; last observation at 48.0 h |
| ESA-AD | 2 (a third measured) | 17 | 64,328,525 of 87,112,696 before binning | Mission1: 6 lightweight channels, median spacing 30 s, 46.3M → 44.1M; Mission2: 11 channels, 18 s, 40.8M → 20.2M; Mission3 (not counted): 24 target channels, 15 s, 183.8M → 92.1M |

Against the literature estimates: C-MAPSS +0.1 %, SMD ±0 %, PhysioNet −4 %, SKAB +21 %, ESA-AD
+29 % (the estimate assumed Mission1's channels ran slower than the binning; they run at it).

Result: **pass**, and the one open verdict is settled by the numbers — ESA-AD enters the mix as an
ingredient (two units, 64M values), which is what carries the 4.7M-parameter reference model over
the strict value threshold (96.7M unique values, 20 per parameter). ADR-0008 is accepted on these
counts. The regenerated tables:

### Assumptions

FLOPs per window = 6 · P · n + 12 · n² · d · L; P = 12 · d² · L. Every GPU-hour below is ±3×.

| Tier | Device | TFLOP/s | d × L | P | Corpus fraction | Window |
|---|---|---|---|---|---|---|
| S | M1 Pro, MPS, fp32 | 1.25 | 192 × 4 | 1.77M | 0.1 | default |
| M | T4, fp16 | 10 | 256 × 6 | 4.72M | 1 | default |
| L | A100 40 GB, bf16 | 50 | 256 × 6 | 4.72M | 1 | longest |

### Corpora

| Corpus | Role | Regime | Download | Units | Observed values | After subsampling | Pretraining side | Basis |
|---|---|---|---|---|---|---|---|---|
| C-MAPSS FD001–FD004 | primary: machine telemetry, regular, remaining-useful-life task | regular | 12 MiB | 709 | 3.37M | same | the four train_FD00x files; the official test files with true RUL are the frozen test set | measured |
| SKAB | second system: testbed telemetry with anomalies, regular, anomaly-detection task | regular | 5 MiB | 35 | 374k | same | all 35 experiment files; there is no official split, so units are held out by the task definition | measured |
| SMD | third system: server telemetry, regular, multivariate anomaly-detection task | regular | 100 MiB | 28 | 26.9M | same | the 28 train files (first half of each machine); the test halves with labels stay out | measured |
| PhysioNet 2012 | irregular: sparse clinical stream, binary classification task | irregular | 22 MiB | 4,000 | 1.73M | same | set-a (4,000 stays); set-b is validation, set-c the frozen test set | measured |
| ESA Anomaly Dataset | second irregular system: satellite telemetry with varying rates per channel, anomaly-detection task | irregular | 10.8 GiB | 2 | 87.1M | 64.3M | first half of each mission by time, as the benchmark itself splits; the second half is the frozen test set | measured |
| PSM | substitute for the server-telemetry role: pooled server metrics, regular | regular | 107 MB (train.csv, test.csv, test_label.csv) | 1 | 3.31M | same | train.csv (13 weeks); test.csv with labels stays out | estimate |
| Tennessee Eastman (Rieth et al.) | substitute for the multivariate-telemetry role: simulated chemical process, regular, 52 variables | regular | 1.4 GB (four RData files) | 10,500 | 273M | same | the training files (fault-free and faulty, 500 runs per condition); the testing files stay out | estimate |

### Window variants

| Corpus | Variant | Units | Windows | Tokens/window | Tokens/epoch | Epochs | GPU-h at M | Basis |
|---|---|---|---|---|---|---|---|---|
| C-MAPSS FD001–FD004 | w30s5 | 709 | 28,231 | 630 | 17.8M | 50 | 59 min | measured |
| C-MAPSS FD001–FD004 | w50s5 (default) | 709 | 25,395 | 1,050 | 26.7M | 50 | 1.8 h | measured |
| C-MAPSS FD001–FD004 | w100s10 | 709 | 9,334 | 2,100 | 19.6M | 50 | 1.8 h | measured |
| SKAB | w50s10 | 35 | 4,525 | 400 | 1.81M | 50 | 5 min | measured |
| SKAB | w100s10 (default) | 35 | 4,350 | 800 | 3.48M | 50 | 12 min | measured |
| SKAB | w200s20 | 35 | 2,011 | 1,600 | 3.22M | 50 | 15 min | measured |
| SMD | w50s10 | 28 | 70,715 | 1,900 | 134M | 20 | 4.7 h | measured |
| SMD | w100s20 (default) | 28 | 35,296 | 3,800 | 134M | 20 | 7.3 h | measured |
| SMD | w200s40 | 28 | 17,586 | 7,600 | 134M | 20 | 12.5 h | measured |
| PhysioNet 2012 | h48 (default) | 4,000 | 3,997 | 439 | 1.75M | 100 | 11 min | measured |
| PhysioNet 2012 | h24s12 | 4,000 | 11,978 | 219 | 2.62M | 100 | 14 min | measured |
| PhysioNet 2012 | h12s6 | 4,000 | 27,863 | 113 | 3.14M | 100 | 16 min | measured |
| ESA Anomaly Dataset | h1s1 (default) | 2 | 76,685 | 839 | 64.3M | 10 | 47 min | measured |
| ESA Anomaly Dataset | h3s3 | 2 | 25,561 | 2,516 | 64.3M | 10 | 1.3 h | measured |
| ESA Anomaly Dataset | h6s6 | 2 | 12,780 | 5,033 | 64.3M | 10 | 2.2 h | measured |
| PSM | w50s10 | 1 | 13,244 | 1,250 | 16.6M | 20 | 28 min | estimate |
| PSM | w100s20 (default) | 1 | 6,620 | 2,500 | 16.6M | 20 | 41 min | estimate |
| PSM | w200s40 | 1 | 3,308 | 5,000 | 16.5M | 20 | 1.1 h | estimate |
| Tennessee Eastman (Rieth et al.) | w50s10 (default) | 10,500 | 483,000 | 2,600 | 1.26G | 5 | 13.3 h | estimate |
| Tennessee Eastman (Rieth et al.) | w100s20 | 10,500 | 220,500 | 5,200 | 1.15G | 5 | 19.8 h | estimate |
| Tennessee Eastman (Rieth et al.) | w200s40 | 10,500 | 84,000 | 10,400 | 874M | 5 | 26.7 h | estimate |

### Eligibility

Thresholds: ≥ 20 training units; unique observed values ≥ 20 × P at tier M (= 94.4M). Overlapping windows do not multiply the count. The last column credits up to 4 epochs of repetition as fresh data.

| Corpus | Units | Units ok | Unique values | Unique ok | Effective per P | Verdict | Decision |
|---|---|---|---|---|---|---|---|
| C-MAPSS FD001–FD004 | 709 | yes | 3.37M | **no** | 3 | pretraining | Primary corpus: enters the mix and hosts the RUL task; 3.4M unique values make it data-starved on its own. |
| SKAB | 35 | yes | 374k | **no** | 0 | pretraining | Enough units, far too few values: the anomaly task corpus and a small ingredient of the mix; its own backbone is a data-starved control. |
| SMD | 28 | yes | 26.9M | **no** | 23 | pretraining | The largest classical source, four fifths of the classical values; stays in pretraining, its tokenised derivative stays private because the data's own licence is unstated. |
| PhysioNet 2012 | 4,000 | yes | 1.73M | **no** | 1 | pretraining | The only classical corpus on the irregular axis: stays in the mix despite very few values, and hosts the classification task. |
| ESA Anomaly Dataset | 2 | **no** | 64.3M | **no** | 55 | ingredient | Enters the mix, never a single-corpus backbone: two independent units, but 64M measured values that triple the mix and carry the reference model over the value threshold. |
| PSM | 1 | **no** | 3.31M | **no** | 3 | candidate | Clear licence and the same role as SMD, but one pooled unit: it can only ever be a mixing ingredient. |
| Tennessee Eastman (Rieth et al.) | 10,500 | yes | 273M | yes | 231 | candidate | Public domain and an order of magnitude more tokens than SMD; simulated like C-MAPSS, so its runs are trajectories of one process model, not independent machines. |

### Licences

| Corpus | Publisher | Terms | Results | Sample in repo | Derivatives | Checked |
|---|---|---|---|---|---|---|
| C-MAPSS FD001–FD004 | NASA Prognostics Center of Excellence | No licence stated (data.nasa.gov: 'License not specified'). A NASA work of the U.S. Government, not subject to copyright in the United States; the PCoE repository asks publications to acknowledge it and disclaims liability. | yes | yes | unclear | 2026-09-11 |
| SKAB | Skoltech (Katser and Kozitsin) | GPL-3.0 (LICENSE at the repository root); no separate statement for the data files, which sit in the same repository. Derivatives are allowed under the same copyleft licence. | yes | yes | yes | 2026-09-11 |
| SMD | NetManAIOps (Su et al.) | The repository is MIT-licensed with the data folder inside it; there is no separate statement for the data, which the authors 'collected from a large Internet company'. | yes | yes | unclear | 2026-09-11 |
| PhysioNet 2012 | PhysioNet (Silva et al.) | Open Data Commons Attribution License v1.0: share, adapt and build derivative databases with attribution; open access, no credentialing. | yes | yes | yes | 2026-09-11 |
| ESA Anomaly Dataset | European Space Agency (De Canio, Kotowski, Haskamp) | CC BY 3.0 IGO (Zenodo record, version 2 of 2025-04-17): attribution required, derivatives and redistribution allowed. | yes | yes | yes | 2026-09-11 |
| PSM | eBay (Abdulaal et al.) | Data files released under CC BY 4.0 (data/LICENSE); the code under BSD-3. | yes | yes | yes | 2026-09-11 |
| Tennessee Eastman (Rieth et al.) | Harvard Dataverse (Rieth, Amsel, Tran, Cook) | Public domain dedication with disclaimer (CC0-equivalent) on Harvard Dataverse. | yes | yes | yes | 2026-09-11 |

### Pretraining programme

In the mix: cmapss, skab, smd, physionet2012, esa_ad; with a backbone of their own: cmapss, skab, smd, physionet2012.

| Backbone | Runs | Corpora | S (M1 Pro, MPS, fp32) | M (T4, fp16) | L (A100 40 GB, bf16) |
|---|---|---|---|---|---|
| single corpus | 4 | cmapss, physionet2012, skab, smd | 3.4 h | 9.5 h | 3.0 h |
| mixed | 1 | cmapss, esa_ad, physionet2012, skab, smd | 3.7 h | 10.3 h | 3.4 h |
| leave-one-corpus-out | 2 | cmapss, esa_ad, physionet2012, skab, smd | 5.9 h | 16.4 h | 5.4 h |
| mixed with channel dropout | 1 | cmapss, esa_ad, physionet2012, skab, smd | 3.7 h | 10.3 h | 3.4 h |
| **Total** |  |  | **16.8 h** | **46.4 h** | **15.1 h** |

### Fine-tuning campaigns

Fine-tuning always uses the tier-M backbone (4.72M parameters); the columns differ only by device.

| Campaign | Runs | Budget | Epochs | Per run on T4, fp16 | Total on M1 Pro, MPS, fp32 | Total on T4, fp16 | Total on A100 40 GB, bf16 |
|---|---|---|---|---|---|---|---|
| first-result grid | 80 | 1000 | 30 | 2.5 min | 26.7 h | 3.3 h | 40 min |
| transfer matrix (8 backbones × 4 tasks × 2 budgets × 5 seeds) | 320 | 1000 | 30 | 2.5 min | 106.8 h | 13.3 h | 2.7 h |
| budget 'all' on C-MAPSS, 5 seeds | 5 | all | 30 | 63.5 min | 42.4 h | 5.3 h | 1.1 h |
| **Total** |  |  |  |  | **175.8 h** | **22.0 h** | **4.4 h** |

### Totals

Cap: 150 h of T4, fp16 for the whole project.

| Tier | Pretraining | Campaigns | Fixed | Total | Band ±3× |
|---|---|---|---|---|---|
| S | 16.8 h | 175.8 h | 240.0 h | **432.6 h** | 144.2 h to 1297.9 h |
| M | 46.4 h | 22.0 h | 30.0 h | **98.4 h** | 32.8 h to 295.2 h |
| L | 15.1 h | 4.4 h | 6.0 h | **25.5 h** | 8.5 h to 76.6 h |

Fixed allowances (T4-class):

| Item | GPU-h | Basis |
|---|---|---|
| ablations, channel dropout, distillation | 30.0 h | allowance of 20–40 h, midpoint; priced when the runs are designed |

### Parameter budget

Mixed corpus (every eligible corpus, default windows): 96.7M unique observed values; 387M effective with up to 4 repetitions credited; 230M tokens processed per epoch and 5.01G over the configured epochs (compute, not information).

| d × L | P | Unique per P | ≥ 20 | Effective per P | ≥ 20 | Seen per P | Mixed pretraining on T4, fp16 |
|---|---|---|---|---|---|---|---|
| 192 × 4 | 1.77M | 55 | yes | 219 | yes | 2830 | 4.6 h |
| 256 × 6 | 4.72M | 20 | yes | 82 | yes | 1061 | 10.3 h |
| 320 × 8 | 9.83M | 10 | **no** | 39 | yes | 509 | 18.7 h |
| 384 × 10 | 17.7M | 5 | **no** | 22 | yes | 283 | 30.6 h |
| 512 × 8 | 25.2M | 4 | **no** | 15 | **no** | 199 | 37.9 h |
