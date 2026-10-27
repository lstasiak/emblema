# Sources in the tasks' regime

## 2026-10-26 — declared before the audit: which source of another domain joins the pretraining

**Question.** No pretraining so far has mixed the intensive-care stays with another domain sampled
the way the stays are: every corpus of another domain was a grid, so domain and regime were never
told apart. One real, multichannel source of another domain, sampled asynchronously, is to join the
pretraining of the stays in one arm. This section fixes, before any candidate is measured,
how that source is chosen and when none is.

**Conditions.** Commit `8d2181d7`, M1 Pro, CPU. The reference regimes come from the coverage report
over corpora that already have readers (100 units, seed 1); the candidates will be measured by the
raw audit, which applies the report's unit measures to a long table read straight from the
publisher's files, and agrees with the report row for row on 300 drawn stays of 2012
(`tests/scripts/test_raw_regime_audit.py`).

    uv run scripts/regime_coverage_report.py measure --out data/report/regime-reference \
        --corpus physionet2012 physionet2019 esa_ad cmapss

**Reference regimes** (median [10th; 90th percentile] over units).

| Corpus | Reader declares | Co-observed | Gap variation | Rate spread | Tokens / window |
|---|---|---|---|---|---|
| Stays 2012 | irregular | 0.22 [0.18; 0.34] | 0.46 [0.29; 0.66] | 1.38 [1.06; 1.89] | 417 [284; 616] |
| Stays 2019 | irregular | 0.32 [0.27; 0.38] | 0.33 [0.00; 0.64] | 1.15 [0.60; 1.40] | 164 [135; 207] |
| ESA-AD, 30 s bins | irregular | 1.00 [1.00; 1.00] | 0.08 [0.08; 0.29] | 0.00 [0.00; 0.00] | 720 [720; 1,320] |
| C-MAPSS per condition | regular | 0.17 [0.17; 1.00] | 0.85 [0.00; 0.93] | 0.30 [0.00; 0.48] | 1,050 [1,050; 1,050] |

C-MAPSS read per condition looks asynchronous on two measures of three, because the reader splits
one grid into a channel per operating condition; only the spread of rates gives it away. A rule on
one measure would admit such an artefact, so the rule below asks for all three.

**Candidates, in order.** The Zwicky Transient Facility's public light curves (DR24), the US Water
Quality Portal (WQX 2.2), the Mars Express Power Challenge (Zenodo 6327379). All three are real
measurements. Their terms as found so far: IRSA's public data without usage restrictions, public
domain according to the re3data registry, and CC BY 4.0 on the Zenodo record; only the last states
on the publisher's own page that a derived block may be published, so rule 4 below asks it of the
other two.

**Frames, fixed before any file is read.** A reading is a row with a numeric value at a stated
instant; a unit enters the frame with at least 50 readings; 100 units are drawn by seed 1 with the
project's seeded ranking, all of them where fewer exist.

| Candidate | Unit | Channels | Readings | Frame |
|---|---|---|---|---|
| ZTF DR24 | a source matched across the g, r and i light curves of one field quadrant within 1.5″ | the bands, at most 3 | epochs with `catflags` 0 and positive `magerr` | 5 field quadrants with all three bands in the release listing, drawn by seed 1 |
| Water Quality Portal | a monitoring location | `CharacteristicName` as published, not harmonised | rows with a numeric `ResultMeasureValue` between 2010-01-01 and 2023-12-31, from the organisation that reports most of the location's results; the instant is the activity's start date and time in UTC, a date without a time at its midnight | stream, lake and reservoir locations with at least 100 results over the same years in the Station listing |
| Mars Express | a calendar month (UTC) of the three training Martian years, 2008-08-22 to 2014-04-14 | the 33 power lines and the 4 solar aspect angles | non-empty numeric cells | every month of the three years |

Names as published can overstate asynchrony: one quantity reported under two names at different
instants reads as two channels never measured together. Synonyms arise mostly between the
organisations that report to the portal, so a location is read through the one that reports most
of its results; the share of its results this leaves out is reported. Command and event files of
Mars Express are not measurements and enter neither the audit nor a corpus.

**Rule.** A candidate passes when all of the following hold; the first to pass in the order above
joins the pretraining, and if none passes the arm is not run and this note says why.

1. *Regime, from the raw audit.* Each median lies at least halfway from a grid (1, 0 and 0) to
   the less irregular corpus of stays (2019: 0.322, 0.325 and 1.154): co-observed ≤ 0.661, gap
   variation ≥ 0.163, rate spread ≥ 0.577.
2. *Channels.* At least 3 channels per unit at the median, since the stays' presence is read
   mostly from the other channels measured in the same bin (`cell-pretext.md`), which a pair of
   channels hardly carries.
3. *Volume.* The source holds at least 12 million readings, about what the pretraining of the
   stays draws from (half the 3.48 million readings of 2012 and the 10.49 million of 2019);
   declared volume suffices where the publisher states one, otherwise the readings are counted.
4. *Licence, before a reader is written.* The publisher's own pages allow a derived block to be
   published with attribution.
5. *Shortcut, after a reader and a publication exist.* Windows are cut so that the median tokens
   per window lie within the stays' range, from the 10th percentile of 2019 to the 90th of 2012
   (135 to 616). Under the masks the stays' values were read with
   (`experiments/backbone-cells-physionet2012-s.toml`), the per-corpus audit
   (`scripts/cell_pretext_audit.py`) answers the hidden cells' presence no better than on the
   stays of 2019 (`cell-pretext.md`): `schedule` ≤ 0.351 and `stacked` ≤ 0.653 in log loss
   removed. A candidate that fails here is set aside and the next one is measured; a survey's
   cadence that acts as a clock fails here.

**Checked on the corpora above, which are not candidates.** The stays of 2012 pass rules 1 and 2,
and the stays of 2019 pass rule 1 by construction; ESA-AD in 30 s bins fails rule 1 on
co-observation, and C-MAPSS per condition on the spread of rates.

**Not decided here.** What the arm is compared with, its prediction and how it is read are
declared with the pretraining that runs it. The source enters only pretraining, never a task.

## 2026-10-26 — M1 Pro, CPU: the Water Quality Portal passes the rules read from raw files, ZTF and Mars Express do not

**Question.** Which candidate, in the declared order, passes the rules that the raw files decide
(regime, channels, volume) and the licence, before a reader is written?

**Conditions.** Commit `23d22403`, M1 Pro, CPU, seed 1, 100 units (all 69 months of Mars Express).
ZTF: the five quadrants drawn lie in fields 567, 875, 711, 591 and 629, their 15 files checked
against the release's MD5 list. Water Quality Portal: the Station listing of 2026-10-26 (114,340
locations); results fetched one location at a time. Every location kept was reported by one
organisation alone, so reading through the main one left no result out. Mars Express: Zenodo record
6327379, MD5 checked.

    uv run scripts/ztf_light_curve_table.py table --root data/raw/ztf \
        --checksums data/raw/ztf/checksum.md5 --out data/report/raw-regime-audit/ztf
    uv run scripts/wqp_result_table.py --stations data/raw/wqp/station.csv \
        --cache data/raw/wqp/results --out data/report/raw-regime-audit/wqp [--timed-only]
    uv run scripts/raw_regime_audit.py --name <source> --root data/report/raw-regime-audit/<source> \
        --pattern readings.csv --unit unit --channel channel --time time
    uv run scripts/mex_power_audit.py --root data/raw/mex/mars-express-power-3years \
        --out data/report/raw-regime-audit

**Rules 1 to 3** (median [10th; 90th percentile] over units; a rule failed in bold).

| Candidate | Co-observed ≤ 0.661 | Gap variation ≥ 0.163 | Rate spread ≥ 0.577 | Channels ≥ 3 | Readings / unit | Volume ≥ 12 M |
|---|---|---|---|---|---|---|
| ZTF DR24 | 0.33 [0.33; 0.50] | 2.85 [1.72; 3.60] | **0.30** [0.19; 2.18] | 3 | 1,040 | 5.3 billion curves |
| Water Quality Portal | 0.54 [0.30; 0.96] | 0.67 [0.23; 2.10] | 0.68 [0.03; 2.54] | 29 | 288 | over 265 million results |
| Mars Express | 0.61 [0.49; 0.68] | **0.08** [0.01; 3.07] | **0.19** [0.03; 0.27] | 37 | 2,882,160 | about 190 million readings |

**The portal without results dated alone.** A quarter of the portal's readings (24.5 %) carry a
date and no time, which the frame places at midnight. The second row applies the frame to the
timed readings alone (124 locations fetched, 100 kept, 85 shared with the first draw):

| Water Quality Portal | Co-observed | Gap variation | Rate spread | Channels |
|---|---|---|---|---|
| dated results at midnight (the declared frame) | 0.54 [0.30; 0.96] | 0.67 [0.23; 2.10] | 0.68 [0.03; 2.54] | 29 |
| timed results only | 0.57 [0.31; 0.95] | 0.67 [0.24; 1.94] | 0.65 [0.03; 2.26] | 30 |

**Licence (rule 4).** The portal's record in the federal catalogue (catalog.data.gov) gives the
U.S. Public Domain licence and public access; the portal's user guide asks for a citation with
DOI 10.5066/P9QRKUVJ; EPA's data licence covers what EPA produces and leaves data of others to
their own terms, which the portal does not state. Organisations submit their data to the portal
to make it publicly accessible. Read as passing, with the citation and the list of reporting
organisations carried by the published block.

**Conclusions.**

1. **The Water Quality Portal passes rules 1 to 4** and is the source a reader is written for;
   rule 5 is read on its publication.
2. **ZTF fails on the spread of rates alone.** Its bands are taken in separate exposures and
   unevenly, but at similar rates (about twice apart at the median, 29 % of sources past the
   threshold against 91 % and 97 % of the stays), where C-MAPSS per condition also sits.
3. **Mars Express is a grid with a second packet.** The 33 power lines share their instants at a
   near constant spacing; the solar angles run apart, which keeps co-observation under one.
4. **Midnight stamps do not decide the portal's result.** Dropping the results dated alone moves the
   median of each regime measure by at most 0.03; the reader is to drop them, since a block of
   midnights would give a pretext a clock to read.

**Limitations.** The portal passes co-observation and the spread of rates by 0.07 to 0.12: its
regime lies near the middle between a grid and the less irregular stays, not at the stays. Its
channels are names as published; one quantity under two names within one organisation still reads as
two channels. ZTF's draw took 82 of its 100 units from the quadrant in field 591, the densest, since
the units were drawn from the five quadrants pooled. Whether data the portal holds from states and
other organisations may be republished is nowhere stated outright. If the portal fails rule 5, no
candidate is left and the arm is not run.

## 2026-10-26 — declared before the publication: how the portal is read, windowed and checked

**Question.** The portal passed the rules its raw files decide. How is it read into a corpus, how
are its windows cut, and what must the published corpus still show before the shortcut audit
(rule 5) is read on it?

**The reader** (`src/emblema/catalog/adapters/readers/wqp.py`, ADR-0065). A unit is a location over
the fetched years, 2010-01-01 to 2024-01-01 UTC, the same extent for every location. A reading is a
result taken in water by the location's main organisation, not a quality-control sample or a
statistic of other results, with a finite number and an instant of its own: results dated alone are
dropped (the timed-only row above), and so are those of an activity that reports one characteristic,
fraction and depth more than once, a logger's readings all stamped with its start. Of the readings
of one channel at one instant one is kept, the water as taken, then the shallowest, then the first.
A characteristic is read in its most frequent unit, codes compared regardless of case. The 64
characteristics measured at the most locations are kept, about 84 % of the audit's readings; a
location enters with at least 50 readings on them. The corpus is the prefix of the frame's seeded
order, fetched by `scripts/wqp_fetch.py` a hundred locations at a time and lengthened by whole
batches until the reader counts at least 12 million readings. These rules drop about half of what
the audit's frame read, so the check below is made on what the reader keeps.

**Windows.** One window length for every location, the shortest of 14, 7 and 3.5 years (122,712,
61,356 and 30,678 hours, stride equal to length) whose median tokens per window in the
publication lie within the stays' range, 135 to 616. A longer window holds more tokens and a
coarser cell; the shortest that keeps the token range is the one nearest the stays' resolution.
The length is chosen on the publication's token counts alone, before the shortcut audit runs.

**The publication is measured again.** Filtering and the cap on channels change what the audit
measured, so the published corpus is measured by the coverage report through its reader (100
units, seed 1) and must still pass rules 1 and 2: co-observed ≤ 0.661, gap variation ≥ 0.163,
rate spread ≥ 0.577 and at least 3 channels per unit, at the median. A failure here ends the arm,
as a failure of rule 5 does; the check can only exclude the portal.

**Predicted from the audit's 100 locations under the cap of 64**, before the reader's further rules
were applied to them. Windows of 7 years held a median of about 170 tokens (14 years: about 225; 3.5
years: about 130, below the range); the reader keeps about half of those readings, so the choice may
fall to 14 years. The regime of the timed-only row: co-observed about 0.57, gap variation about
0.67, rate spread about 0.65. Dropping repeats at one instant leaves the instants a channel is read
at, so the gap variation and the spread of rates should move little; dropping the loggers removes
instants of their own, in a direction not measured. The margins were 0.07 to 0.12, so the check is
not foregone; the regime of the reader's output was not measured before this section was
committed.

## 2026-10-26 — M1 Pro, CPU: the portal as read fails the regime check; the arm goes on as a question of data, not of regime

**Question.** Does the corpus the reader keeps still pass rules 1 and 2, how long are its windows,
and does it hold 12 million readings?

**Conditions.** Commit `829d9572`, M1 Pro, CPU; the batches of the first 37,000 locations of the
frame. The check is the declared command; the decomposition re-reads the same 100 locations with
the reader's rules added one at a time.

    uv run scripts/regime_coverage_report.py measure --corpus wqp --out data/report/wqp-regime-gate

**The check** (median [10th; 90th percentile] over 100 locations, seed 1).

| Co-observed ≤ 0.661 | Gap variation ≥ 0.163 | Rate spread ≥ 0.577 | Channels ≥ 3 |
|---|---|---|---|
| **0.68** [0.33; 0.95] | 0.62 [0.14; 2.33] | **0.36** [0.00; 1.63] | 15 |

**The same locations, rule by rule** (medians).

| Read as | Co-observed | Gap variation | Rate spread | Channels |
|---|---|---|---|---|
| the audit's timed-only frame | 0.63 | 0.78 | 0.48 | 22 |
| water only, no quality-control samples | 0.66 | 0.71 | 0.47 | 22 |
| no statistics of other results | 0.66 | 0.71 | 0.47 | 22 |
| no repeats within an activity | 0.67 | 0.71 | 0.45 | 22 |
| one reading per channel and instant | 0.67 | 0.71 | 0.45 | 22 |
| one unit per characteristic, 64 channels (the reader) | 0.68 | 0.62 | 0.36 | 15 |

**Volume and windows.** 22,281 locations kept, 11,520,684 readings, 64 channels: 4 % short of 12
million, so the prefix is lengthened by whole batches. Median tokens per window: 174 at 14 years
(p10 69, p90 1,094), 133 at 7 years, 96 at 3.5 years; the window is 14 years.

**Conclusions.**

1. **The portal as read fails rule 1** on co-observation and the spread of rates; by the rule
   declared above, the arm as a test of another domain in the stays' regime ends.
2. **No setting of the reader would pass it.** Without the cap on channels the same locations read
   0.67 and 0.45, both past the thresholds, and a cap only removes channels; even the audit's
   frame reads a spread of rates of 0.48 here. The audit's 0.65 to 0.68 came from another draw of
   100 locations, and the spread between locations (0.03 to 2.5) makes a median of 100 move by
   that much: the margin it showed was the draw's.
3. **A typical location is a visit at which many characteristics are measured together at a
   similar rhythm**: between a grid and the stays, nearer the grid on the spread of rates.
4. **The arm goes on as a question of data.** The regime rule decided what the arm could claim, not
   whether its reading is sound: whether adding the portal helps the stays is read on the stays'
   held-out side by the arm itself, under the readout declared with the pretraining. The arm now
   asks whether a large, irregular source of another domain, sampled between a grid and the
   stays, helps the intensive-care tasks. The shortcut audit (rule 5) and the reader's cleaning
   stay, since they protect what the model learns; no corpus of another domain that pretraining
   has read so far was other than a grid.

**Limitations.** A median over 100 units moves by more than the margins the rule asked for, so the
regime rule decided on the draw as much as on the source; a rule of this kind needs more units or
an interval. A result of the arm within its threshold says only that the portal, at this capacity,
moves the stays by less than about 0.02; it does not settle whether another domain in the stays'
regime would.

## 2026-10-26 — M1 Pro, CPU: the portal's cells hold no clock; its values are hard to answer without a model

**Question.** Does the published portal pass the shortcut audit (rule 5), and what volume and
windows does its publication hold?

**Conditions.** Commit `0b1a6102`, M1 Pro, CPU. The batches of the first 38,700 locations, read to
the minute (ADR-0065) and published to the local stack's bucket with windows of 14 years (122,712
hours), stride equal, a seeded split of a fifth: manifest `sha256:d854f2cf…`. The audit reads the
publication by its manifest under the masks the stays' values were read with, 2,000 training and
2,000 validation windows.

    uv run python -m emblema.entrypoints.cli.publish_corpus --corpus wqp \
        --window 122712 --stride 122712
    uv run scripts/cell_pretext_audit.py --manifest <key> <checksum> --corpus wqp \
        --masks experiments/backbone-cells-physionet2012-s.toml

**The publication.** 23,293 locations, every one a window; 11,674,718 readings over 64 channels,
2.7 % short of 12 million once readings within one minute were merged; cells held 0.41 of the time
(the stays: 0.52 and 0.62).

**Presence of a hidden cell** (log loss removed against the constant share), against the stays
under the same masks.

| Answer | Portal | Rule 5 | Stays 2012 | Stays 2019 |
|---|---|---|---|---|
| `schedule` | 0.027 | ≤ 0.351 | 0.238 | 0.351 |
| `cross` | 0.481 | — | 0.537 | 0.623 |
| `stacked` | 0.543 | ≤ 0.653 | 0.593 | 0.653 |

**Values** (R² against the channel's mean, every hidden cell): mean −0.249 / −0.036 / 0.041 /
−0.000 / −0.120 for carry, interpolation, cross, schedule and the stack; maximum −0.932 for the
stack; the count 0.904 from the carry and 0.914 from the stack, as on the stays.

**Conclusions.**

1. **The portal passes rule 5.** Its schedule answers presence at 0.027, a tenth of the stays',
   so the programmes' calendars give no clock; presence is read from the other channels of the
   same bin, as on the stays.
2. **Its values are almost unanswerable without a model**, and the stack does worse than the
   mean: raw values over units and sites with heavy tails, which the value terms of the pretext
   will weigh. Robust normalisation of values, parked so far, bears on this corpus.
3. **The count is a shortcut here as on the stays** (0.90 to 0.91), so the count term carries
   little the corpus could teach.

**Limitations.** The publication used for the audit is local; the one the pretraining reads is
published later, last in the chain of vocabularies, from a prefix lengthened past 12 million
readings. The audit scores answers without a model; whether the portal helps the stays is read by
the arm.

## 2026-10-27 — M1 Pro, CPU: the eICU demo read as 2012 frames a stay keeps the stays' regime and holds no clock

**Question.** Which tables of the eICU Collaborative Research Database demo (version 2.0.1) are read
so that its stays join the pretraining in the regime of the 2012 stays, and does its publication
pass the shortcut audit the portal passed (rule 5)?

**Conditions.** Commit `d6784854`, M1 Pro, CPU. The variants were measured before the reader was
written: long tables of the first 48 hours of every unit stay (numeric values, offsets in hours),
built from the demo's tables outside the repository and read by the raw audit (100 stays, seed 1),
which reproduces the 2012 reference row from 2012's raw files to the hundredth. The reader
(ADR-0066) was then measured by the coverage report, and its publication to the local stack's
bucket (windows of 48 hours, stride equal, a seeded split of a fifth; manifest `sha256:75a73b1b…`)
by the shortcut audit under the masks the stays' values were read with.

    uv run scripts/raw_regime_audit.py --name eicu-<variant> --root <long table> \
        --pattern long.csv --unit unit --channel channel --time time --skip _stay
    uv run scripts/regime_coverage_report.py measure --corpus eicu_demo physionet2012 \
        --out data/report/l32b4/eicu-reader
    uv run python -m emblema.entrypoints.cli.publish_corpus --corpus eicu_demo \
        --window 48 --stride 48
    uv run scripts/cell_pretext_audit.py --manifest <key> <checksum> --corpus eicu_demo \
        --masks experiments/backbone-cells-physionet2012-s.toml

**Regime** (median [10th; 90th percentile] over 100 stays, seed 1).

| Read as | Co-observed | Gap variation | Rate spread | Readings / stay |
|---|---|---|---|---|
| Stays 2012 (reference) | 0.22 [0.18; 0.34] | 0.46 [0.29; 0.66] | 1.38 [1.06; 1.89] | 417 |
| monitor (`vitalPeriodic`) and laboratory | 0.14 [0.08; 0.51] | 0.21 [0.00; 0.65] | 2.45 [0.00; 2.51] | 1,609 |
| charting, `vitalAperiodic` and laboratory, every label | 0.13 [0.09; 0.32] | 0.42 [0.17; 0.68] | 1.42 [0.65; 2.02] | 343 |
| charting, laboratory and urine in the 2012 variables | 0.24 [0.18; 0.52] | 0.43 [0.12; 0.90] | 1.07 [0.44; 1.97] | 134 |
| the reader: stays of 48 hours on the unit | 0.21 [0.15; 0.29] | 0.42 [0.17; 0.74] | 1.37 [0.61; 1.95] | 310 tokens a window |

**Presence of a hidden cell** (log loss removed against the constant share), against the stays
under the same masks. The first column is the publication of every stay at commit `2a7b3e03`, the
shorter ones leaving the end of their window empty; the reader keeps stays of 48 hours on the unit.

| Answer | Every stay | The reader | Rule 5 | Stays 2012 | Stays 2019 |
|---|---|---|---|---|---|
| `schedule` | 0.251 | 0.257 | ≤ 0.351 | 0.238 | 0.351 |
| `cross` | 0.627 | 0.589 | — | 0.537 | 0.623 |
| `stacked` | **0.671** | 0.635 | ≤ 0.653 | 0.593 | 0.653 |
| cells held | 0.41 | 0.52 | — | 0.52 | 0.62 |

**The publication.** 932 stays of 803 patients in 181 of the demo's 186 hospitals; 930 hold a
window (two hold no reading), 308,105 tokens, 186 stays on the validation side. Values (R² against
the channel's mean, the stack): 0.033 over every hidden cell, 0.446 under block masks and 0.631
under token masks, where neighbouring readings remain; the count 0.931 from the stack.

**Conclusions.**

1. **The nurse's charting, in the 2012 variables, is the stays' regime**; the five-minute monitor
   is another (gap variation 0.21, rate spread 2.45, four times the readings), and is not read.
2. **Through the reader the demo matches 2012 on all three measures** (0.21, 0.42, 1.37 against
   0.22, 0.46, 1.38), so it adds stays of a second hospital system without adding a regime.
3. **Short stays fail the shortcut audit and stays of 48 hours pass it.** Read whole, a stay that
   leaves the unit early leaves whole bins empty, which the other channels of the bin answer
   without a model (stack 0.671); stays of 48 hours on the unit, the 2012 challenge's own rule,
   answer at 0.635 and hold cells as often as the 2012 stays.
4. **The schedule gives no clock** in either reading (0.251, 0.257).

**Limitations.** The stays kept hold 60 % of the demo's readings in the first 48 hours; a pipeline
whose units may be shorter than a window would keep the rest. The validation side holds 186 stays,
so the audit's figures move by more than on the 2012 stays. The publication is local; the one the
pretraining reads is published to the shared bucket before the run.
