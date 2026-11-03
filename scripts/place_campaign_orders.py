"""Define a batch of campaigns under the registered worker settings and place one order for each.

Run from the repository's root, on a clean tree, since an order carries the commit it was placed
at. Resumable: a campaign already defined or ordered is skipped, so a refusal half way through
loses nothing. Writes what a platform cell reads (``colab_cell.py``) and prints identifiers only.

The plan names the settings and one table per campaign:

    worker_env = "worker-env.sh"     # relative to the plan; sourced with B set to the backbone
    pool = "ml"                      # which workers the orders are for
    [[campaign]]
    key = "m1-taught-base1"          # the label the cell and the reading know it by
    file = "campaigns/x.toml"
    task = "<task id>"
    weights = "<backbone digest>"

    uv run scripts/place_campaign_orders.py --plan plan.toml --campaigns campaigns.txt
        --orders orders.txt
"""

import argparse
import re
import shlex
import subprocess
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

CAMPAIGN = re.compile(r"^[0-9a-f-]{36}$")
ORDER = re.compile(r"^durable/sha256/(?P<digest>[0-9a-f]{64}) sha256:(?P=digest)$")

type Command = Callable[[str, Sequence[str]], str]
"""Runs a campaign CLI command under a backbone: its digest, the arguments; the last line back."""


@dataclass(frozen=True, kw_only=True)
class PlannedCampaign:
    """A campaign to define and order.

    Attributes:
        key: Its label.
        file: Its campaign file.
        task: The task it runs on.
        weights: The digest of the backbone it runs under.
    """

    key: str
    file: str
    task: str
    weights: str


def read_plan(path: Path) -> tuple[Path, str, list[PlannedCampaign]]:
    """The settings' path, relative to the plan, the workers' pool and the campaigns of a plan."""
    plan = tomllib.loads(path.read_text())
    campaigns = [
        PlannedCampaign(key=c["key"], file=c["file"], task=c["task"], weights=c["weights"])
        for c in plan["campaign"]
    ]
    return path.parent / plan["worker_env"], plan["pool"], campaigns


def _rows(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    rows = (line.split() for line in path.read_text().splitlines() if not line.startswith("#"))
    return {fields[0]: fields for fields in rows if fields}


def place(planned: PlannedCampaign, pool: str, campaigns: Path, orders: Path, run: Command) -> None:
    """Define the campaign unless it was, then order it unless it was.

    Raises:
        SystemExit: If the CLI refuses, naming the campaign's key and its answer.
    """
    known = _rows(campaigns).get(planned.key)
    if known is None:
        identifier = run(
            planned.weights, ("define", "--file", planned.file, "--task", planned.task)
        )
        if CAMPAIGN.match(identifier) is None:
            raise SystemExit(f"[{planned.key}] define refused: {identifier}")
        with campaigns.open("a") as handle:
            handle.write(f"{planned.key} {identifier} {planned.weights}\n")
    else:
        identifier = known[1]
    if planned.key in _rows(orders):
        return
    answer = run(planned.weights, ("order", "--campaign", identifier, "--pool", pool))
    ordered = ORDER.match(answer)
    if ordered is None:
        raise SystemExit(f"[{planned.key}] order refused: {answer}")
    with orders.open("a") as handle:
        handle.write(f"{planned.key} {identifier} {ordered['digest']}\n")


def _answer_of(status: int, stdout: str, stderr: str) -> str:
    """What a CLI answered: its last printed line, or why it refused."""
    lines = (stdout if status == 0 else stderr or stdout).strip().splitlines()
    return lines[-1] if lines else ""


def campaign_command(worker_env: Path) -> Command:  # pragma: no cover
    """A command running the campaign CLI under the settings; an order reaches the remote store."""

    def run(weights: str, arguments: Sequence[str]) -> str:
        remote = ["--env-file", ".env.r2"] if arguments[0] == "order" else []
        call = shlex.join(
            ["uv", "run", *remote, "python", "-m", "emblema.entrypoints.cli.campaign", *arguments]
        )
        script = f"B={shlex.quote(weights)}; export B; . {shlex.quote(str(worker_env))} && {call}"
        done = subprocess.run(
            ["/bin/bash", "-c", script],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        return _answer_of(done.returncode, done.stdout, done.stderr)

    return run


def tree_state() -> tuple[str, bool]:  # pragma: no cover
    """The commit the tree stands at, and whether it is clean."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return head, not changes.strip()


def main(
    argv: Sequence[str] | None = None,
    *,
    run: Command | None = None,
    state: Callable[[], tuple[str, bool]] = tree_state,
) -> None:
    """Place every campaign of the plan.

    Args:
        argv: The command line.
        run: What runs a campaign CLI command; the CLI itself unless given.
        state: The tree's commit and whether it is clean; asked of git unless given.

    Raises:
        SystemExit: If the tree is dirty, or the orders already name another commit.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--campaigns", type=Path, required=True)
    parser.add_argument("--orders", type=Path, required=True)
    arguments = parser.parse_args(argv)
    head, clean = state()
    if not clean:
        raise SystemExit("the tree is not clean: an order carries its commit")
    orders: Path = arguments.orders
    if orders.exists():
        found = re.search(r"^# from ([0-9a-f]+)$", orders.read_text(), re.MULTILINE)
        if found is None:
            raise SystemExit(f"{orders} names no commit in a '# from <commit>' line")
        if found[1] != head:
            raise SystemExit(f"the orders were placed at {found[1]}, the tree is at {head}")
    else:
        orders.write_text(f"# from {head}\n")
    worker_env, pool, campaigns = read_plan(arguments.plan)
    command = run or campaign_command(worker_env)
    for planned in campaigns:
        place(planned, pool, arguments.campaigns, orders, command)
    print(f"{len(_rows(orders))} orders in {orders}")


if __name__ == "__main__":  # pragma: no cover
    main()
