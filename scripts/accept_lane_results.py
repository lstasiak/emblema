"""Accept the results of a platform cell into the local registry, one per lane.

A cell prints a line per finished lane, ``[label] exit 0 in 73 s: durable/sha256/<d> sha256:<d>``.
This script takes that block as it was pasted back, the last line of each label winning since a
cell run again appends, and accepts each result through the campaign CLI. A label whose last line
did not exit cleanly is named and not accepted. Prints one line a result and a count; exits 1 if
anything was left out or refused.

    uv run scripts/accept_lane_results.py --results campaign-results.txt --log accept.log
"""

import argparse
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

RESULT = re.compile(
    r"^\[(?P<label>[^\]]+)\] exit (?P<status>\d+) in \d+ s:"
    r"(?: durable/sha256/(?P<key>[0-9a-f]{64}) sha256:(?P<digest>[0-9a-f]{64}))?.*$"
)


@dataclass(frozen=True, kw_only=True)
class LaneResult:
    """What a lane last reported.

    Attributes:
        label: The lane's label in the cell.
        digest: The digest of its result, or ``None`` if it did not end cleanly.
    """

    label: str
    digest: str | None


def read_results(text: str) -> list[LaneResult]:
    """The last report of each label, in the order the labels first appear.

    A line that is no lane's report is ignored, so the block can be pasted with what surrounds it.
    """
    last: dict[str, LaneResult] = {}
    for line in text.splitlines():
        found = RESULT.match(line.strip())
        if found is None:
            continue
        clean = (
            found["status"] == "0" and found["key"] is not None and found["key"] == found["digest"]
        )
        last[found["label"]] = LaneResult(
            label=found["label"], digest=found["key"] if clean else None
        )
    return list(last.values())


def accept_through_cli(digest: str) -> tuple[int, str]:  # pragma: no cover
    """Accept one result by the campaign CLI against the remote store: its status and last line."""
    done = subprocess.run(
        [
            "uv",
            "run",
            "--env-file",
            ".env.r2",
            "python",
            "-m",
            "emblema.entrypoints.cli.campaign",
            "accept",
            "--result",
            f"durable/sha256/{digest}",
            f"sha256:{digest}",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = (
        (done.stdout if done.returncode == 0 else done.stderr or done.stdout).strip().splitlines()
    )
    return done.returncode, lines[-1] if lines else ""


def accept_all(
    results: Sequence[LaneResult], accept: Callable[[str], tuple[int, str]]
) -> tuple[list[str], bool]:
    """Accept every clean result: a line per result, and whether all went through."""
    lines: list[str] = []
    every = True
    for result in results:
        if result.digest is None:
            lines.append(f"[{result.label}] not accepted: the lane did not end cleanly")
            every = False
            continue
        status, last = accept(result.digest)
        lines.append(f"[{result.label}] exit {status}: {last}")
        every = every and status == 0
    return lines, every


def main(
    argv: Sequence[str] | None = None,
    *,
    accept: Callable[[str], tuple[int, str]] = accept_through_cli,
) -> None:
    """Read the pasted block, accept what ended cleanly, and write the log.

    Args:
        argv: The command line.
        accept: What accepts one digest; the campaign CLI unless given.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    arguments = parser.parse_args(argv)
    results = read_results(arguments.results.read_text())
    if not results:
        raise SystemExit(f"no lane's report in {arguments.results}")
    lines, every = accept_all(results, accept)
    arguments.log.write_text("".join(f"{line}\n" for line in lines))
    print("\n".join(lines))
    print(f"accepted: {sum(' exit 0:' in line for line in lines)} of {len(results)}")
    if not every:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
