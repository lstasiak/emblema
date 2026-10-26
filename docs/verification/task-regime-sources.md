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
