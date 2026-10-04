# Tennessee Eastman process simulation sample

Two runs from each training file of Rieth et al.'s additional Tennessee Eastman process
simulation data, cut to their first twenty samples: runs 1 and 2 of the fault-free file and run 1
of faults 1 and 2 of the faulty file, 40 rows per file of the publisher's 55 columns
(`faultNumber`, `simulationRun`, `sample`, `xmeas_1`…`xmeas_41`, `xmv_1`…`xmv_11`). The rows are
real values at their real sample numbers, written back as gzip-compressed RData by the `rdata`
package the project locks, each file holding one data frame under the publisher's variable name
(`fault_free_training`, `faulty_training`). Derived rather than byte for byte: the smaller file is
25 MB and the larger 494 MB.

Source of record: Harvard Dataverse, doi:10.7910/DVN/6C3JR1, files `TEP_FaultFree_Training.RData`
(24,678,017 bytes, MD5 `ec126484534331f85001d8c4ebce6d17`) and `TEP_Faulty_Training.RData`
(494,063,194 bytes, MD5 `c5f594d54c47e620ff877feb58407fda`); the testing files are not sampled, as
they are the benchmark's evaluation data. Reference: C. A. Rieth, B. D. Amsel, R. Tran and
M. B. Cook, "Additional Tennessee Eastman Process Simulation Data for Anomaly Detection
Evaluation", Harvard Dataverse, 2017. The data is dedicated to the public domain (a CC0-equivalent
dedication with disclaimer), which permits a redistributed derivative.

This is test data for the corpus reader. Keep the files as they are: the reader's checksum is
computed over the raw bytes, and a regenerated file may differ in its compression stream alone.

Regenerate from the downloaded files, with the project's corpora extra (`uv sync --all-extras`):

```sh
uv run python - <<'PY'
from pathlib import Path

import pandas as pd
import rdata

RAW = Path("data/raw/tep")
OUT = Path("tests/data/tep")
for name, runs in (
    ("TEP_FaultFree_Training", ((0, 1), (0, 2))),
    ("TEP_Faulty_Training", ((1, 1), (2, 1))),
):
    ((variable, frame),) = rdata.read_rda(RAW / f"{name}.RData").items()
    kept = pd.concat(
        frame[
            (frame["faultNumber"] == fault)
            & (frame["simulationRun"] == run)
            & (frame["sample"] <= 20)
        ]
        for fault, run in runs
    ).reset_index(drop=True)
    kept.index = pd.RangeIndex(1, len(kept) + 1)
    rdata.write_rda(OUT / f"{name}.RData", {variable: kept}, compression="gzip")
PY
```
