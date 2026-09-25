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
