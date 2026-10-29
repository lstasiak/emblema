# ESA Anomaly Dataset sample

Every channel the two readers of the ESA Anomaly Dataset, version 2, select, thinned to one row
in every 48 hours and cut to the first 365 rows: the lightweight channels `ESA-Mission1/channels/
channel_41.zip` to `channel_46.zip` and `ESA-Mission2/channels/channel_18.zip` to `channel_28.zip`,
which both readers read, and the further Mission1 channels of the raw reader's policy,
`channel_12.zip` to `channel_40.zip`, `channel_47.zip` to `channel_52.zip` and `channel_70.zip` to
`channel_76.zip`; fifty-nine archives of about 4 KB each. The lightweight channels of a mission are
sampled at the same rows, as the published channels of a packet share their instants; the further
channels keep their own packets' instants, one row in two days of each. The rows are real values at their real
instants, re-pickled by the pandas the project locks, in the layout the publisher uses: one zip per
channel holding one member of the channel's name, a data frame of one column indexed by instants.
Thinned rather than truncated so that a mission's training half spans a year of calendar months —
twelve for Mission1, ten for Mission2 — which is what a split on units needs.

Source of record: Zenodo record 15237121 (doi:10.5281/zenodo.15237121), the three mission archives
`ESA-Mission1.zip` (3,776,246,073 bytes, SHA-256
`ba28f761b1deab4dbba4728793bff139fea39dbf9cf0d9c559d619ffe75d5a72`), `ESA-Mission2.zip`
(4,098,539,932 bytes, SHA-256 `e8a89be1917b6754a10bd323441e87a82c8cf2e84ed162442c2dcf72ecc346d5`)
and `ESA-Mission3.zip` (3,734,403,444 bytes, SHA-256
`f426c4bb9857299c586e3c50f35ddb58b469a3e5565f586d4095bcb6cf532404`); the third mission is not
sampled, as the benchmark leaves it out. Reference: K. Kotowski, C. Haskamp, J. Andrzejewski,
B. Ruszczak, J. Nalepa, D. Lakey, P. Collins, A. Kolmas, M. Bartesaghi, J. Martinez-Heras and
G. De Canio, "European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry",
arXiv:2406.17826, 2024. The data is published under CC BY 3.0 IGO, which permits a redistributed
derivative with attribution.

Beside the channels, each mission holds `labels.csv` and `anomaly_types.csv` cut to the first
three events that touch the lightweight channels, rows copied verbatim from the published files:
test data for the reader of the telemetry's events.

This is test data for the corpus reader. Keep the archives byte-exact: the reader's checksum is
computed over the raw bytes, and a regenerated zip differs in its member timestamps alone.

Regenerate from the unpacked archives, with the project's pandas (`uv sync --all-extras`):

```sh
uv run python - <<'EOF'
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/esa_ad")
OUT = Path("tests/data/esa_ad")
# 48 hours of a channel's own rows: 5,760 of Mission1's 30-second rows, 1,920 of its 90-second
# rows, 192 of its 900-second rows, 9,600 of Mission2's 18-second rows.
for mission, numbers in (
    ("ESA-Mission1", [*range(12, 53), *range(70, 77)]),
    ("ESA-Mission2", range(18, 29)),
):
    for number in numbers:
        frame = pd.read_pickle(RAW / mission / "channels" / f"channel_{number}.zip")
        out = OUT / mission / "channels" / f"channel_{number}.zip"
        out.parent.mkdir(parents=True, exist_ok=True)
        median = float(frame.index.to_series().diff().dt.total_seconds().median())
        step = {30.0: 5760, 90.0: 1920, 900.0: 192, 18.0: 9600}[median]
        frame.iloc[::step].head(365).to_pickle(
            out, compression={"method": "zip", "archive_name": f"channel_{number}"}
        )
EOF
```
