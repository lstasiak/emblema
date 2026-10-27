"""Cut the miniature sample from the fetched batches: named locations, their first records."""

import csv
import sys
import zipfile
from pathlib import Path

LOCATIONS = {
    "batch-00000.zip": (
        "USGS-01387610",
        "USGS-423412082471600",
        "11113300-BCHFSPFRNLF",
        "CHESAPEAKEMONITORINGCOOP-CMC.WSR.WASPRU002",
    ),
    "batch-00001.zip": ("21VASWCB-9-SNC008.04", "21VASWCB-1ALIV016.50"),
}
ROWS = 120
LOCATION_COLUMN = "MonitoringLocationIdentifier"


def main() -> None:
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for name, wanted in LOCATIONS.items():
        with zipfile.ZipFile(raw / name) as archive:
            lines = archive.read("result.csv").decode("utf-8").splitlines(keepends=True)
        records = csv.reader(lines)
        column = next(records).index(LOCATION_COLUMN)
        kept, taken, start = [], dict.fromkeys(wanted, 0), 1
        # A quoted field may hold a line break, so a record is cut back out of the lines the
        # parser consumed, byte for byte.
        for record in records:
            chunk = lines[start : records.line_num]
            start = records.line_num
            location = record[column]
            if location in taken and taken[location] < ROWS:
                taken[location] += 1
                kept.extend(chunk)
        info = zipfile.ZipInfo("result.csv", date_time=(2026, 10, 26, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(out / name, "w") as archive:
            archive.writestr(info, lines[0] + "".join(kept))
        print(name, taken)


if __name__ == "__main__":
    main()
