"""What every report script prints the same way.

The reports these scripts produce are pasted into the verification notes, so they share a shape:
a dated heading saying which machine produced the numbers, then tables. Keeping that here means a
note written a year apart still reads as one document, and a change to the shape is one change.
"""

import platform
from collections.abc import Iterable, Sequence
from datetime import datetime


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [" | ".join(header), " | ".join("---" for _ in header)]
    lines += [" | ".join(row) for row in rows]
    return "\n".join(f"| {line} |" for line in lines)


def dated_heading() -> str:
    """The heading a note's section starts with: today, and the machine that measured."""
    return f"## {datetime.now():%Y-%m-%d} — {platform.system()} {platform.machine()}"


def machine() -> str:
    return f"{platform.platform()}, {platform.processor() or 'unknown CPU'}"
