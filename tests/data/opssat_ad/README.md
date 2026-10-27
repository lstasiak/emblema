# OPSSAT-AD sample

Seven fragments of `segments.csv`, the telemetry file of OPSSAT-AD, the OPS-SAT anomaly detection
benchmark: the header and, for each chosen fragment, its first twelve rows, copied byte for byte.
The fragments were chosen so that the reader meets everything the file varies: both parts of the
authors' split (`train` fragments 1, 6, 219, 333 and 1069; `test` fragments 5 and 13), both labels
(1 and 13 anomalous), both sampling periods (1 s and 5 s), four of the nine channels, and the
shortest fragment of the file (1069, eight rows). Rows of a fragment are contiguous in the file,
as they are in the source.

Source of record: Zenodo record 12588359 (doi:10.5281/zenodo.12588359), `segments.csv`
(17,970,946 bytes, MD5 `1f4b6ffd0359efac96a6698ff557e0f9`, SHA-256
`dab2e01e873c93590692b50a7cc312d9cce4d5ba7e69ce8015603c47a8ffb97b`) and `dataset.csv`
(507,568 bytes, MD5 `c12fb83863afa4463dec07303a20bb14`, the features the authors computed per
fragment; not sampled). Reference: B. Ruszczak, K. Kotowski, D. Evans and J. Nalepa, "The OPS-SAT
benchmark for detecting anomalies in satellite telemetry", Scientific Data 12, 2025,
doi:10.1038/s41597-025-05035-3. The data is published under CC BY 4.0, which permits a
redistributed derivative with attribution.

This is test data for the corpus reader. Keep the file byte-exact: the reader's checksum is
computed over the raw bytes.

Regenerate from the downloaded file:

```sh
uv run python - <<'EOF'
from pathlib import Path
chosen = {1, 5, 6, 13, 219, 333, 1069}
kept: dict[int, int] = {}
lines = Path("data/raw/opssat_ad/segments.csv").read_bytes().split(b"\n")
out = [lines[0]]
for line in lines[1:]:
    if line:
        segment = int(line.split(b",")[6])
        if segment in chosen and kept.get(segment, 0) < 12:
            out.append(line)
            kept[segment] = kept.get(segment, 0) + 1
Path("tests/data/opssat_ad/segments.csv").write_bytes(b"\n".join(out) + b"\n")
EOF
```
