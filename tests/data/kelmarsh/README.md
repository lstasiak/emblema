# Kelmarsh wind farm sample

Two turbine tables and one status log of the first yearly archive of the Kelmarsh wind farm data,
`Kelmarsh_SCADA_2016_3082.zip`: turbines 1 and 2, each table's preamble and header copied byte for
byte and sixty of its ten-minute rows, one in every four days from 2016-05-03 09:40, the first row
in which the turbines' temperatures are read. Thinned rather than truncated so that a turbine
spans eight calendar months, which is what a split on units needs. The status log of turbine 1
keeps its preamble, header and first five events; the reader does not read it, and the archive
holds it so that a test shows as much. The rows are real values at their real instants, every one
of the table's 299 columns kept.

Source of record: Zenodo record 16807551 (doi:10.5281/zenodo.16807551),
`Kelmarsh_SCADA_2016_3082.zip` (97,915,553 bytes, MD5 `ca8efbb6dbcfae3516583937eb5b5402`, SHA-256
`4e8a73aeb823aeb21a2c7829f14434e44eea523f1d7f9be0177ff1b76e3ded27`). Reference: C. Plumley and
R. Takeuchi, "Kelmarsh wind farm data", Zenodo, 2025. The data is published by Cubico Sustainable
Investments Ltd under CC BY 4.0, which permits a redistributed derivative with attribution.

This is test data for the corpus reader. Keep the archive byte-exact: the reader's checksum is
computed over the raw bytes.

Regenerate both farms' samples from the downloaded archives (the members carry one fixed date, so
the same rows give the same bytes):

```sh
uv run python - <<'EOF'
import zipfile
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("tests/data")
# One row in every four days of a table, 576 of its ten-minute rows, from the row named; a
# status log keeps its preamble, header and first five events.
STEP, ROWS, HEAD = 576, 60, 10
CUTS = {
    "kelmarsh/Kelmarsh_SCADA_2016_3082.zip": {
        "Turbine_Data_Kelmarsh_1_2016-01-03_-_2017-01-01_228.csv": "2016-05-03 09:40:00",
        "Turbine_Data_Kelmarsh_2_2016-01-03_-_2017-01-01_229.csv": "2016-05-03 09:40:00",
        "Status_Kelmarsh_1_2016-01-03_-_2017-01-01_228.csv": None,
    },
    "penmanshiel/Penmanshiel_SCADA_2016_WT01-10_3107.zip": {
        "Turbine_Data_Penmanshiel_01_2016-06-06_-_2017-01-01_1042.csv": "2016-06-06 18:10:00",
        "Status_Penmanshiel_01_2016-06-06_-_2017-01-01_1042.csv": None,
    },
    "penmanshiel/Penmanshiel_SCADA_2016_WT11-15_3107.zip": {
        "Turbine_Data_Penmanshiel_15_2016-07-14_-_2017-01-01_1056.csv": "2016-07-14 20:00:00",
    },
}
for archive, members in CUTS.items():
    target = OUT / archive
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW / archive) as source, zipfile.ZipFile(target, "w") as sample:
        for member, first in members.items():
            lines = source.read(member).split(b"\n")
            if first is None:
                kept = lines[: HEAD + 5]
            else:
                start = next(i for i, line in enumerate(lines) if line.startswith(first.encode()))
                kept = lines[:HEAD] + lines[start::STEP][:ROWS]
            info = zipfile.ZipInfo(member, date_time=(2026, 11, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            sample.writestr(info, b"\n".join(kept) + b"\n")
EOF
```
