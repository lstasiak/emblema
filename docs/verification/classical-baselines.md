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
