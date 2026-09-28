# Classical baselines on the turbofan task

## 2026-09-24 — M1 Pro, 32 GB: MiniRocket against aeon, the selection of knobs, and the comparison

**Question.** Does this project's MiniRocket compute what aeon's does? Which setting of each
classical baseline does the declared selection choose? Where do the baselines stand against the
network trained from scratch? All numbers are **validation**, tier S.

**Conditions.**

| | |
| --- | --- |
| Corpus | `cmapss` in force: per operating condition, four subsets, window 50, stride 5, manifest `sha256:d63f8e1b…`, republished here from the raw files to the same bytes |
| Task | FD001, 79 tuning / 21 validation / 100 frozen engines |
| Classical cells | `worker-general` image at `eb62475`, CPU, 4 threads |
| Network cells | host MPS, fp32, order at `4058a62` run by `campaign_run` and accepted back |
| Defaults | XGBoost library defaults; MiniRocket 9,996 features, ridge penalty 10⁻³…10³ by LOO, one grid step per cycle |
| Selection | `campaigns/selection-classical-fd001.toml` (`305c2e9`), 10 repeats, 16 of 79 tuning engines held out each, one-SE rule with Nadeau–Bengio correction |
| Comparison | `campaigns/baselines-fd001.toml` (`4058a62`), 3 seeds, control = network from scratch under the configuration in force |
| aeon check | `scripts/classical_baselines_report.py`, aeon 1.6.0, 20 seeds |

### MiniRocket against aeon

Where no draw differs, features agree (`agreement.csv`): largest difference 2.6e-08 in four of six
cases; 5 cells in a million differ in the other two, where a convolution lands exactly on a bias
in aeon's float32.

On 21 channels, grid of 64 steps, 2,000 held-out windows (`multivariate.csv`):

| budget | this project | aeon | paired gap | lower in | under aeon's draws |
| --- | --- | --- | --- | --- | --- |
| 200 | 0.2189 ± 0.0035 | 0.2175 ± 0.0038 | +0.0014 ± 0.0053 | 8 of 20 | ≤ 1.8e-03 from aeon |
| 1000 | 0.2006 ± 0.0041 | 0.1970 ± 0.0034 | +0.0035 ± 0.0050 | 6 of 20 | ≤ 7.5e-04 from aeon |

### Selection `3856e705…`

RMSE on held-out tuning engines, mean ± SD over 10 repeats; **bold** = chosen.

| candidate | default | depth 3 | depth 9 | rate 0.1 × 300 |
| --- | --- | --- | --- | --- |
| trees per channel, 50 | **20.83 ± 2.69** | 19.95 ± 3.19 | 20.81 ± 2.71 | 21.32 ± 3.64 |
| trees per channel, 200 | 15.64 ± 0.88 | **14.48 ± 0.67** | 16.02 ± 1.24 | 15.16 ± 0.92 |
| spectrum, 50 | 27.08 ± 2.24 | **24.95 ± 1.98** | 27.08 ± 2.25 | 27.45 ± 2.49 |
| spectrum, 200 | 20.15 ± 1.17 | **18.99 ± 1.54** | 20.70 ± 1.30 | 19.68 ± 1.09 |
| across channels, 50 | 17.66 ± 1.45 | **16.90 ± 1.21** | 17.69 ± 1.50 | 17.83 ± 1.33 |
| across channels, 200 | 14.79 ± 0.79 | **14.09 ± 0.90** | 15.08 ± 0.91 | 14.30 ± 0.77 |

| MiniRocket grid | × 0.5 | × 1 (default) | × 2 | × 4 |
| --- | --- | --- | --- | --- |
| 50 | 19.03 ± 0.93 | **17.40 ± 0.78** | 17.93 ± 1.44 | 18.08 ± 1.32 |
| 200 | 18.42 ± 0.92 | **16.24 ± 1.26** | 16.86 ± 0.88 | 17.16 ± 0.95 |

### Comparison `ee69d456…`

RMSE on the 21 validation engines, mean ± SD over 3 seeds.

| candidate | 50 | 200 |
| --- | --- | --- |
| trees per channel | 18.25 ± 1.47 | **13.94 ± 0.71** |
| trees across channels | **16.59 ± 1.36** | 14.32 ± 0.92 |
| MiniRocket | 16.81 ± 0.79 | 15.24 ± 0.15 |
| trees over the spectrum | 21.61 ± 1.21 | 18.47 ± 0.25 |
| network from scratch (control) | 22.51 ± 1.61 | 18.86 ± 0.23 |

Against the control, paired over engines (bootstrap, 10,000 resamples):

| candidate | budget | reduction | 95 % interval | floor | verdict |
| --- | --- | --- | --- | --- | --- |
| **trees per channel** | **200** | **+26.0 %** | **[+2.86, +6.88]** | 0.38 | **confirmed** |
| trees per channel | 50 | +18.9 % | [+0.82, +7.46] | 1.61 | indistinguishable |
| trees across channels | 200 / 50 | +24.0 % / +26.3 % | [+3.02, +6.13] / [+3.40, +8.25] | | distinguishable |
| MiniRocket | 200 / 50 | +19.2 % / +25.4 % | [+1.83, +5.40] / [+2.91, +8.35] | | distinguishable |
| trees over the spectrum | 200 / 50 | +2.1 % / +4.1 % | both include 0 | | indistinguishable |

### Cost per cell (fit and answer, corpus at hand)

| trees per channel | spectrum | across channels | MiniRocket | network (MPS) |
| --- | --- | --- | --- | --- |
| 0.3–0.4 s | 0.7–1.0 s | 0.9–1.1 s | 2.7–3.3 s | 812–979 s |

### Conclusions

1. This MiniRocket matches aeon's arithmetic; the remaining gap (0.6–1.8 %) is the random stream,
   not the computation.
2. The trees prefer depth 3; MiniRocket prefers the series as recorded.
3. Every baseline except the spectrum beats the network trained from scratch; the trees per
   channel by 26 % at 200 labels (confirmed).
4. Across campaigns, the trees per channel (13.94) are about 13 % below the best pretrained arm in
   `label-efficiency-curve.md` (low-rank updates, 16.10). This is not a paired comparison.
5. Runs repeat exactly: the 16 classical cells shared by two campaigns, and every tree cell across
   the two runs, agree bit for bit.

### Limitations and what was superseded

- 3 seeds, tier S, validation only. The baselines were tuned per budget; the arms at 200 only.
- The network drifts by up to 1.1 RMSE per seed between identical MPS runs.
- A first run (selection `77e4f5f8…`, comparisons `62fb1f09…`, `b1e8712f…`) is superseded for
  every MiniRocket cell. Its grid binned single-precision times one step early (fixed in
  `afe5199`) and spanned all 126 channels, 105 of which FD001 never reports (fixed in `2032804`).
  MiniRocket then scored 17.08 / 18.29 and its selection chose a grid of × 2. Tree and network
  cells stand.

## 2026-09-25 — M1 Pro, 32 GB: the patch model trained from scratch

**Question.** Where does a patch model trained from nothing on the grid (in the style of
PatchTST, [ADR-0039](../adr/0039-the-patch-baseline.md)) stand against the set encoder trained
from nothing and against the tuned trees, paired on the same engines? All numbers are
**validation**, tier S.

**Conditions.**

| | |
| --- | --- |
| Code | `f1050c6` |
| Corpus, task | as in the section above: `cmapss` `sha256:d63f8e1b…`, FD001, 21 validation engines |
| Campaign | `campaigns/patch-fd001.toml`, `add35a93…`; 50 and 200 labels × seeds 1–3; control = network from scratch, endpoint = patch model at 200 |
| Schedule (both networks) | 30 epochs, at least 2,000 steps, batch 16, peak 1e-3 (the control's), warm-up 0.1, cosine to 1 %, no weight decay |
| Network from scratch | the shape of the backbone in force (`sha256:6283c210…`, tier M): 4.78 M weights |
| Patch model | patch 8, stride 4 (12 tokens per channel), width 192, 3 heads, 4 layers, feed-forward 768, dropout 0.2, one grid step per cycle; 21 channels read, 1.79 M weights; not tuned |
| Trees | per channel, at the variants selection `3856e705…` chose; `worker-general` image at `f1050c6`, CPU, 4 threads |
| Network cells | host MPS, fp32, run by a Celery worker started on the host at `f1050c6` (not by an order, so the commit was not checked by the run itself) |

**RMSE**, mean ± SD over 3 seeds:

| candidate | 50 | 200 | seconds per cell |
| --- | --- | --- | --- |
| trees per channel | **18.25 ± 1.47** | **13.94 ± 0.71** | 0.3–0.9 |
| patch model from scratch | 21.10 ± 1.99 | 18.40 ± 1.01 | 82–96 |
| network from scratch (control) | 22.62 ± 1.68 | 18.75 ± 0.28 | 808–1,032 |

**Paired over engines** (bootstrap, 10,000 resamples; reduction of RMSE relative to the rival):

| comparison | budget | reduction | 95 % interval | floor | verdict |
| --- | --- | --- | --- | --- | --- |
| **patch model vs network from scratch** | **200** | **+1.8 %** | **[−1.22, +1.82]** | 0.38 | **indistinguishable** |
| patch model vs network from scratch | 50 | +6.6 % | [+0.29, +2.65] | 1.68 | distinguishable, practically nil |
| trees vs network from scratch | 200 / 50 | +25.6 % / +19.3 % | [+2.74, +6.79] / [+1.01, +7.50] | | distinguishable |
| patch model vs trees (outside the design) | 200 | −32.0 % | [−5.80, −3.01] | | trees better |
| patch model vs trees (outside the design) | 50 | −15.7 % | [−5.90, +0.37] | | includes 0 |

The last two rows pair the same cells by the same bootstrap; the design registered neither, so
they carry no verdict of the campaign's rules.

**Reproducibility.** The 126 per-engine errors of the trees equal those of campaign `ee69d456…`
bit for bit. The network from scratch does not repeat on MPS: 18.75 here against 18.86 there at
200, up to 6.7 RMSE apart on a single engine.

**The kept model.** The patch model of the endpoint cell (200 labels, seed 1) was stored and read
back by its checksum: 21 channels × 50 steps, 1,789,441 weights, target scale 125.

### Conclusions

1. Two different architectures trained from nothing, under one schedule, land at the same error
   on this task at 200 labels (18.40 and 18.75, interval across zero); the patch model does it
   with 37 % of the weights and a tenth of the time per cell.
2. Both stay about a quarter to a third behind the tuned trees at 200 labels. The gap is not
   specific to the set encoder.
3. Both networks pool over the whole window before a linear head, and both run under the same
   untuned schedule. The result is consistent with the head or the schedule limiting both, not
   with an architectural cause; it does not show which (diagnostics in the next ticket).

### Limitations

- 3 seeds, tier S, validation only. The patch model is not tuned: it learns at the control's peak
  rate with PatchTST's default dropout.
- The two networks differ in size (1.79 M against 4.78 M weights).
- The network cells ran through a Celery worker on the host, not through an order. The broker
  closed the worker's connection once (missed heartbeats during a long cell); every result had
  been stored before, and the redelivered cells answered from the registry without running again.

*2026-09-25:* the diagnostics conclusion 3 defers to are in
[`head-and-representation.md`](head-and-representation.md).

## 2026-09-27 — M1 Pro, 32 GB: the selection of knobs at 1,000 labels and at every label

**Question.** The selection of 2026-09-24 chose a variant per baseline at 50 and 200 labels. The
repeated curve also stands at 1,000 labelled windows and at all of them, and a variant chosen
where labels are scarce need not be the one for where they are not. Which variant does the same
rule choose there?

**Conditions.** Code `d3c98928`; campaign `2b81fda7…` (`campaigns/selection-classical-large-fd001.toml`):
the grid of 2026-09-24 — the trees shallower and deeper and at a third of the rate with three
times the rounds, MiniRocket on grids of half, twice and four times the corpus's step — at 1,000
labels and at all, ten repeats each holding 16 of the 79 tuning engines out, one-standard-error
rule with the Nadeau–Bengio correction, ties towards the setting published. Run natively on the
CPU through an order (`campaign_run`, 320 cells, 15 min; trees 1–4 s a cell, MiniRocket 6–24 s),
accepted back into the registry, read with `campaign select`. RMSE on the held-out engines, mean
± SD over the ten repeats; **bold** = chosen.

| candidate | default | depth 3 | depth 9 | rate 0.1 × 300 |
| --- | --- | --- | --- | --- |
| trees per channel, 1,000 | 13.29 ± 0.81 | 13.37 ± 0.84 | 13.87 ± 1.02 | **12.79 ± 0.82** |
| trees per channel, all | 12.68 ± 0.87 | 12.91 ± 0.89 | 13.04 ± 0.84 | **12.12 ± 0.83** |
| spectrum, 1,000 | 17.60 ± 1.00 | 17.01 ± 0.94 | 18.28 ± 0.97 | **16.41 ± 0.68** |
| spectrum, all | 16.59 ± 0.99 | 16.61 ± 1.27 | 17.64 ± 0.83 | **15.68 ± 0.94** |
| across channels, 1,000 | 13.62 ± 0.97 | **13.31 ± 0.78** | 14.03 ± 0.90 | 13.06 ± 0.86 |
| across channels, all | 13.08 ± 0.81 | **12.90 ± 0.66** | 13.39 ± 0.69 | 12.63 ± 0.68 |

| MiniRocket grid | × 0.5 | × 1 (default) | × 2 | × 4 |
| --- | --- | --- | --- | --- |
| 1,000 | 17.88 ± 1.06 | **16.52 ± 1.16** | 17.22 ± 1.17 | 16.78 ± 1.05 |
| all | 18.66 ± 1.11 | **16.66 ± 1.03** | 17.80 ± 1.21 | 17.56 ± 0.98 |

**Conclusions.**

- With labels plentiful the trees want a finer ensemble, not a shallower one: the per-channel
  and spectral trees move from depth 3 at 200 to a third of the rate with three times the rounds
  at 1,000 and at all. Across channels the finer ensemble is best at both budgets too, but depth
  3 lies within one standard error and turns one knob to its two, so the rule keeps depth 3.
- MiniRocket keeps the grid of the corpus's own step at every budget; the finer grids never help
  and the coarser one costs a cycle or more.
- The baselines are now tuned at every budget of the repeated curve by the one protocol the
  arms are tuned by.

**Limitations.** Ten repeats of 16 engines; the intervals of the trees' variants overlap at every
budget and the rule reads them as ties broken towards the default. Nothing here touches the
validation side.

## 2026-09-28 — declared before the run: MiniRocket's penalty over outcomes on the intensive-care task

**Question.** Over outcomes MiniRocket fits an L2-penalised logistic regression, its penalty
chosen by the log-loss of five folds within the drawn labels among ten strengths from 0.001 to
1,000, the list the regression task's ridge uses. A smoke campaign at the small tier chose 215 at
50 stays, and a fit outside the campaign chose 1,000 at 200 — the strongest listed. Does the list
stop too soon at the budgets the task's grid will run, and what does a fit cost?

**Conditions.** Commit of this section; MacBook Pro M1 Pro, CPU, eight threads of linear
algebra. Task `physionet2012-in-hospital-death` over manifest `cef44de2…`; budgets 50, 200,
1,000 and every stay of the tuning side, each under seeds 1, 2 and 3, drawn by the task's own
draw in proportion to the outcomes, as a campaign cell draws them; 9,996 features; the fit is the
candidate's own code, offered the ten strengths in force and four more steps of the same factor
(4,640, 21,500, 100,000, 464,000). The choice is made within the drawn labels; no window of the
validation side is read and nothing is scored.

    uv run scripts/convolution_penalty_report.py --manifest durable/sha256/cef44de2… sha256:cef44de2… --out data/report/t42c/penalties

**Prediction.** At 200, 1,000 and every stay, at least two seeds of three choose 1,000 or a
stronger penalty; no fit chooses the strongest of the longer list, so four steps are enough. The
fit over every stay takes about ten minutes, most of it at the weakest penalties.

**Reading, declared beforehand.** If any fit chooses 1,000 or stronger, the convolution baseline
on outcomes gets a list of strengths of its own, apart from the regression's, which stays as it
is: the list in force followed by every longer step up to one beyond the strongest chosen here,
registered as configuration before any selection on the task runs. If no fit does, the list in
force stands. The weak end of the list stays whatever it costs; dropping it would be a separate
decision on a measured cost.

## 2026-09-28 — MiniRocket's penalty over outcomes measured

Under the declaration above, at `63b99be5`, on the MacBook Pro M1 Pro: the choices in 33 minutes
on eight threads of linear algebra, the fits over every stay taking 465 to 671 s each.

| Budget | Seed | Stays | Deaths | Penalty chosen | At or past 1,000 |
| --- | --- | --- | --- | --- | --- |
| 50 | 1 | 50 | 7 | 215 | no |
| 50 | 2 | 50 | 7 | 464,000 | yes |
| 50 | 3 | 50 | 7 | 464,000 | yes |
| 200 | 1–3 | 200 | 28 | 4,640 | yes |
| 1,000 | 1–3 | 1,000 | 139 | 4,640 | yes |
| every stay | 1–3 | 3,997 | 554 | 4,640 | yes |

**Not a prediction: the choices checked.** While reading the table above, the fit was compared
with the kept candidate of a smoke campaign on the same draw (50 stays, seed 1) and found
identical to the bit — penalty, rows of the grid, scale of every feature, every weight — and the
folds' log-loss was traced by a throwaway script on six draws. The trace was then committed as a
mode of the same script and run at `19f7752c` on 2026-09-29, four threads, 34 minutes: every strength fitted
independently of the candidate's own cross-validation on the same five folds, at the solver's
tolerance and at 1e-8.

    M=cef44de241af45ebb9f99da55679445a72632ada9f8b982dc9651e8554e51a78
    uv run scripts/convolution_penalty_report.py --manifest durable/sha256/$M sha256:$M \
        --out data/report/t42c/curves --curves

| Budget | Seed | Chosen by the fit | Folds' best, tolerance 1e-4 | Folds' best, tolerance 1e-8 | Prevalence alone |
| --- | --- | --- | --- | --- | --- |
| 50 | 1 | 215 | 215 (0.2780) | 215 (0.2783) | 0.4107 |
| 50 | 2 | 464,000 | 464,000 (0.4107) | 464,000 (0.4107) | 0.4107 |
| 50 | 3 | 464,000 | 464,000 (0.4109) | 464,000 (0.4109) | 0.4107 |
| 200 | 1 | 4,640 | 4,640 (0.3718) | 4,640 (0.3719) | 0.4053 |
| 200 | 2 | 4,640 | 4,640 (0.3739) | 4,640 (0.3741) | 0.4053 |
| 200 | 3 | 4,640 | 4,640 (0.3715) | 4,640 (0.3714) | 0.4053 |
| 1,000 | 1 | 4,640 | 4,640 (0.3506) | 4,640 (0.3506) | 0.4032 |
| 1,000 | 2 | 4,640 | 4,640 (0.3503) | 4,640 (0.3503) | 0.4032 |
| 1,000 | 3 | 4,640 | 4,640 (0.3430) | 4,640 (0.3430) | 0.4032 |

**Conclusions.**

1. The first part of the prediction held: at 200, 1,000 and every stay all nine fits chose 4,640,
   past the strongest penalty in force. At 200 and 1,000 the choice is an interior minimum of the
   folds' log-loss, well below the prevalence alone, and the same under both tolerances.
2. The second part failed: at 50 stays two seeds of three chose the strongest step of the longer
   grid. With seven deaths the folds find nothing that predicts better than the prevalence: the
   log-loss falls towards the prevalence's as the penalty grows and never goes below it. The
   choice is the limit of an infinite penalty, and a longer grid would not change it.
3. The cost was as predicted: about ten minutes a fit over every stay.
4. By the reading declared beforehand, the convolution baseline on outcomes gets a list of its
   own: the published grid followed by 4,640, 21,500, 100,000, 464,000 and 2,150,000, one step
   beyond the strongest chosen. The regression task keeps the published grid.

**Limitations.** A selection draws its budget from four fifths of the tuning side, so its largest
budget is about 3,200 stays, not 3,997; the choice was the same at 200, 1,000 and 3,997, and is
expected to be the same there. The choices ran on eight threads, where a campaign runs on four,
and a fit's answer depends on the order its sums were split in; the same fits repeated on four
threads for the trace chose the same strength on every draw it covers. The curve over every
stay was not traced, for its cost.
