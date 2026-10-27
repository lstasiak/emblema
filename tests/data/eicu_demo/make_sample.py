"""Cut the miniature sample from the downloaded tables: every row of five named stays."""

import csv
import gzip
import io
import sys
from pathlib import Path

STAYS = ("2858139", "2848323", "1843889", "376041", "1817131")
TABLES = ("patient", "nurseCharting", "lab", "intakeOutput")
STAY_COLUMN = "patientunitstayid"


def main() -> None:
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        name = f"{table}.csv.gz"
        text = gzip.decompress((raw / name).read_bytes()).decode("utf-8")
        # Only a line feed ends a line: ``splitlines`` would also break on separators that free
        # text may hold.
        lines = io.StringIO(text, newline="\n").readlines()
        records = csv.reader(lines)
        column = next(records).index(STAY_COLUMN)
        kept: list[str] = []
        start = 1
        # A quoted field may hold a line break, so a record is cut back out of the lines the
        # parser consumed, byte for byte.
        for record in records:
            chunk = lines[start : records.line_num]
            start = records.line_num
            if record[column] in STAYS:
                kept.extend(chunk)
        # No timestamp in the header, so a rerun on the same tables writes the same bytes.
        data = (lines[0] + "".join(kept)).encode("utf-8")
        (out / name).write_bytes(gzip.compress(data, mtime=0))
        print(name, len(kept), "lines")


if __name__ == "__main__":
    main()
