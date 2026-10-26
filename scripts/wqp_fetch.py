"""Fetch the Water Quality Portal results of the first locations of the seeded frame, in batches.

The corpus is a prefix of the frame the regime audit drew from: the Station listing's locations in
the order the project's seeded ranking gives their identifiers. Their results over the years read
are fetched a batch of locations at a time, one zip per batch, so an interrupted fetch resumes
where it stopped and a longer prefix only adds batches.

    uv run scripts/wqp_fetch.py --stations data/raw/wqp/station.csv \
        --out data/raw/wqp/batches --locations 16000

Writes ``batch-<n>.zip`` (the portal's CSV of the batch's results) and ``locations.csv`` (each
location fetched, its batch). The narrow result profile carries every column the reader uses.
"""

import argparse
import csv
import json
import sys
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.wqp_result_table import FIRST, LAST, ranked

RESULTS = (
    "https://www.waterqualitydata.us/data/Result/search"
    "?mimeType=csv&zip=yes&dataProfile=narrowResult"
)
# The portal answers a large batch slowly rather than not at all; silence this long is a failure.
SILENCE = 900


def batches(locations: Sequence[str], size: int) -> list[tuple[str, ...]]:
    """The locations cut into consecutive batches of ``size``, the last one possibly shorter."""
    return [tuple(locations[at : at + size]) for at in range(0, len(locations), size)]


def batch_path(out: Path, number: int) -> Path:
    return out / f"batch-{number:05d}.zip"


def query(locations: Sequence[str]) -> bytes:
    return json.dumps(
        {"siteid": list(locations), "startDateLo": FIRST, "startDateHi": LAST}
    ).encode()


def fetch(locations: Sequence[str], attempts: int = 5) -> bytes:  # pragma: no cover - the network
    for attempt in range(attempts):
        request = urllib.request.Request(
            RESULTS, data=query(locations), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=SILENCE) as response:
                content: bytes = response.read()
            return content
        except OSError as error:
            if attempt == attempts - 1:
                raise
            print(f"retrying after {error}", flush=True)
            time.sleep(60 * (attempt + 1))
    raise AssertionError("unreachable")


def write_listing(out: Path, planned: Sequence[tuple[str, ...]]) -> None:
    with (out / "locations.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("location", "batch"))
        for number, batch in enumerate(planned):
            writer.writerows((location, batch_path(out, number).name) for location in batch)


def main() -> None:  # pragma: no cover - the command line and the network
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stations", type=Path, required=True, help="the Station listing, CSV")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--locations", type=int, required=True, help="the prefix fetched")
    parser.add_argument("--batch", type=int, default=100, help="locations per request")
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    planned = batches(
        ranked(arguments.stations, arguments.seed)[: arguments.locations], arguments.batch
    )
    write_listing(arguments.out, planned)
    for number, batch in enumerate(planned):
        path = batch_path(arguments.out, number)
        if path.exists():
            continue
        partial = path.with_suffix(".part")
        partial.write_bytes(fetch(batch))
        partial.rename(path)
        print(f"{path.name}: {len(batch)} locations, {path.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
