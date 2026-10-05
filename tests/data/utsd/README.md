# UTSD sample

Two datasets of UTSD, the Unified Time Series Dataset (Tsinghua, THUML), as two shards of the
collection's layout under `UTSD-12G/`: Arrow IPC stream files of record batches with the columns
`item_id` (string), `start`, `end`, `freq` (strings, empty in the collection) and `target` (list of
single-precision floats), one record per univariate series named `<dataset>_<a>_<b>`.

- `Health_SelfRegulationSCP1`: three six-variate series (139, 219 and 458, the first three records'
  series of the 1G volume), each variate cut to its first 32 values. A multivariate dataset: `a`
  numbers the series and `b` its variate. Series 139 lies across both shards, three variates in
  each, so a reader has to gather a unit from more than one file.
- `Nature_temperature_rain_dataset_without_missing_values`: the first five records of shard 69 of
  the 12G volume (`a` = 0, `b` = 971–975), each cut to its first 20 values. A collection of
  separate series: every record is a unit, named by both numbers.

The values are real values in their real order; the sample is derived, not byte for byte, since
the smallest shard of the collection is 11 MB. The first shard holds the records of series 458
and 219 in one batch and the first three variates of 139 in another; the second holds the other
three variates of 139 with the first two rainfall records, then the other three rainfall records.
Shards carry no `state.json` or `dataset_info.json`: the reader finds records by the names they
carry and reads nothing else.

Source of record: Hugging Face dataset `thuml/UTSD`, revision `7326ff5f4578da73d843fd675d760c6c6054017f`
(2025-06-19), directory `UTSD-12G/` (80 shards, 3,892,126,910 bytes, 289,560 records). Reference:
Y. Liu, H. Zhang, C. Li, X. Huang, J. Wang and M. Long, "Timer: Generative Pre-trained Transformers
Are Large Time Series Models", ICML 2024. The collection is published under the Apache License 2.0,
which permits a redistributed derivative.

This is test data for the corpus reader. Keep the files as they are: the reader's checksum is
computed over the records' names and values, and a regenerated shard may differ in its framing.

Regenerate from the downloaded shards and the 1G volume's first records, with the project's
corpora extra (`uv sync --all-extras`):

```sh
uv run python - <<'PY'
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

RAW = Path("data/raw/utsd")
OUT = Path("tests/data/utsd/UTSD-12G")
SCP1 = "Health_SelfRegulationSCP1"
RAIN = "Nature_temperature_rain_dataset_without_missing_values"


def records(volume, dataset, wanted, cut):
    kept = {}
    for shard in sorted((RAW / volume).glob("*.arrow")):
        table = ipc.open_stream(pa.memory_map(str(shard))).read_all()
        for item, values in zip(table.column("item_id").to_pylist(), table.column("target")):
            if item in wanted:
                kept[item] = np.asarray(values.as_py()[:cut], dtype=np.float32)
    return [(item, kept[item]) for item in wanted]


scp1 = records(
    "UTSD-1G", SCP1, [f"{SCP1}_{s}_{v}" for s in (458, 219, 139) for v in range(6)], 32
)
rain = records("UTSD-12G", RAIN, [f"{RAIN}_0_{v}" for v in range(971, 976)], 20)
schema = pa.schema(
    [
        ("item_id", pa.string()),
        ("start", pa.string()),
        ("end", pa.string()),
        ("freq", pa.string()),
        ("target", pa.list_(pa.float32())),
    ]
)


def batch(rows):
    return pa.record_batch(
        [
            pa.array([item for item, _ in rows]),
            *(pa.array([""] * len(rows)) for _ in range(3)),
            pa.array([values.tolist() for _, values in rows], type=pa.list_(pa.float32())),
        ],
        schema=schema,
    )


shards = {
    "data-00000-of-00002.arrow": [scp1[:12], scp1[12:15]],
    "data-00001-of-00002.arrow": [scp1[15:] + rain[:2], rain[2:]],
}
for name, batches in shards.items():
    with pa.OSFile(str(OUT / name), "wb") as sink, ipc.new_stream(sink, schema) as writer:
        for rows in batches:
            writer.write_batch(batch(rows))
PY
```
