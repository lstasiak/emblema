"""Write the cell or script that fulfils pretraining orders on Colab, Kaggle or the Mac.

A pretraining is one long process. By default a card holds one order at a time; a small shape,
which keeps a card a few per cent busy because the host limits it, runs several to a card under
CUDA MPS with ``per_card``. On Kaggle a running order is stopped before the batch
session would be killed and take its output with it. An order interrupted resumes from the last
checkpoint its log names, tried twice in the cell and again if the cell is run again; a checkpoint
from another session or platform can be named in the plan. Every network step runs under a limit
and ends the cell cleanly, so a stalled server does not hold a paid GPU, and the cell never exits
non-zero.

The plan:

    name = "emblema-pretraining"   # the directory of results
    orders = "orders.txt"          # "# from <commit>", then lines "label ... <digest>"
    blocks = ["<manifest digest>"] # optional: publications to fetch once, before the workers
    cards = 1                      # cards the workers are spread over
    per_card = 1                   # orders a card holds at once (one on the Mac)
    threads = 2                    # optional: threads per process when a card is shared
    num_workers = 0                # optional: batch collating processes
    extras = "ml,tracking"         # the pretraining command line builds its tracker either way
    progress_every = 100           # steps between reports
    pieces = ["seed1", "seed2"]    # order labels, in the order they are claimed

    [resume]                       # optional: where an order continues from another session
    seed1 = "transient/sha256/<digest> sha256:<digest>"

    uv run scripts/pretraining_cell.py --plan plan.toml --out cell.sh [--platform kaggle|mac]
"""

import argparse
import re
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cell_parts import (
    COLAB,
    LIMITED,
    MPS,
    PLATFORMS,
    Platform,
    blocks_step,
    cutoff_line,
    head,
    heredoc,
    monitors,
    write_cells,
)

CHECKPOINT = re.compile(r"^transient/sha256/([0-9a-f]+) sha256:\1$")


@dataclass(frozen=True, kw_only=True)
class PretrainingPlan:
    """All a cell states.

    Attributes:
        name: The directory the cell keeps its results in.
        commit: The commit the orders were placed at.
        orders: The digest of each order, by label.
        pieces: The labels, in the order they are claimed.
        cards: How many cards the workers are spread over.
        per_card: How many orders a card holds at once.
        threads: Threads each process keeps to when a card is shared, or nought to leave it.
        num_workers: Processes collating batches for each run, or nought.
        extras: What the cell installs.
        progress_every: Steps between the run's reports.
        blocks: The publications to fetch before the workers start.
        resume: The checkpoint each named order continues from.
    """

    name: str
    commit: str
    orders: Mapping[str, str]
    pieces: tuple[str, ...]
    cards: int
    extras: str
    progress_every: int
    per_card: int = 1
    threads: int = 0
    num_workers: int = 0
    blocks: tuple[str, ...] = ()
    resume: Mapping[str, str] = field(default_factory=dict)


def read_plan(path: Path) -> PretrainingPlan:
    """The plan at ``path``, its files read relative to it.

    Raises:
        SystemExit: If the orders name no commit, a piece has no order, a checkpoint to resume from
            is malformed or belongs to no piece, or the plan asks for no card.
    """
    plan = tomllib.loads(path.read_text())
    text = (path.parent / plan["orders"]).read_text()
    found = re.search(r"^# from ([0-9a-f]{40})$", text, re.MULTILINE)
    if found is None:
        raise SystemExit(f"{plan['orders']} names no commit in a '# from <commit>' line")
    orders = _digests(text)
    pieces = tuple(plan["pieces"])
    missing = [label for label in pieces if label not in orders]
    if missing:
        raise SystemExit(f"no order for {', '.join(missing)}")
    resume = dict(plan.get("resume", {}))
    for label, reference in resume.items():
        if label not in pieces:
            raise SystemExit(f"a checkpoint names {label}, which is no piece of this plan")
        if CHECKPOINT.fullmatch(reference) is None:
            raise SystemExit(f"{label}: {reference} is not a checkpoint's key and digest")
    cards = plan.get("cards", 1)
    per_card = plan.get("per_card", 1)
    if cards < 1 or per_card < 1:
        raise SystemExit("a pretraining needs a card and at least one order on it")
    if per_card > 1 and plan.get("threads", 0) < 1:
        raise SystemExit("orders sharing a card each need a limit on threads: the plan names none")
    return PretrainingPlan(
        name=plan["name"],
        commit=found[1],
        orders={label: orders[label] for label in pieces},
        pieces=pieces,
        cards=cards,
        extras=plan.get("extras", "ml,tracking"),
        progress_every=plan.get("progress_every", 100),
        per_card=per_card,
        threads=plan.get("threads", 0),
        num_workers=plan.get("num_workers", 0),
        blocks=tuple(plan.get("blocks", ())),
        resume=resume,
    )


def _digests(text: str) -> dict[str, str]:
    """Each order's digest by its label: the last field of a line, with or without its prefix."""
    rows = (line.split() for line in text.splitlines() if line and not line.startswith("#"))
    return {fields[0]: fields[-1].removeprefix("sha256:") for fields in rows if len(fields) > 1}


def cell_of(plan: PretrainingPlan, platform: Platform = COLAB) -> str:
    """The cell's text, ``%%bash`` first.

    Raises:
        SystemExit: If a plan shares one accelerator on a machine where that is not done.
    """
    if platform.hosted and plan.cards > 1:
        raise SystemExit("the Mac has one accelerator: cards must be 1")
    if plan.per_card > 1 and not platform.shares_by_mps:
        raise SystemExit(f"{platform.name} does not share a card under MPS: per_card must be 1")
    title = f"{plan.name}: pretraining orders."
    start, stop, utilisation = monitors(platform, every=10)
    threads = (
        f"export OMP_NUM_THREADS={plan.threads} MKL_NUM_THREADS={plan.threads}"
        if plan.threads
        else ""
    )
    workers_arg = f" --num-workers {plan.num_workers}" if plan.num_workers else ""
    parts = [
        head(platform, title=title, name=plan.name, commit=plan.commit, extras=plan.extras),
        blocks_step(plan.blocks) if plan.blocks else "",
        heredoc("pieces", plan.pieces),
        heredoc("orders", [f"{label} {plan.orders[label]}" for label in plan.pieces]),
        heredoc("resume", [f"{label} {ref}" for label, ref in plan.resume.items()]),
        LIMITED,
        MPS if plan.per_card > 1 else "",
        _WORK.replace("@CUTOFF@", cutoff_line(platform))
        .replace("@CARDS@", str(plan.cards))
        .replace("@PER_CARD@", str(plan.per_card))
        .replace("@THREADS@", threads)
        .replace("@DEVICE@", platform.device)
        .replace("@WORKERS_ARG@", workers_arg)
        .replace("@MONITOR@", start.strip("\n"))
        .replace("@STOP@", stop)
        .replace("@UTILISATION@", utilisation.strip("\n"))
        .replace("@EVERY@", str(plan.progress_every)),
    ]
    return "\n".join(part.strip("\n") for part in parts if part) + "\n"


def main(argv: Sequence[str] | None = None, *, head_commit: str | None = None) -> None:
    """Write the plan's cell, if the tree stands at the orders' commit and the cell is sound.

    Raises:
        SystemExit: If the tree is at another commit or the cell breaks its own rules.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--platform", choices=sorted(PLATFORMS), default=COLAB.name)
    arguments = parser.parse_args(argv)
    plan = read_plan(arguments.plan)
    at = head_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if at != plan.commit:
        raise SystemExit(f"the tree is at {at}, the orders at {plan.commit}")
    platform = PLATFORMS[arguments.platform]
    cell = cell_of(plan, platform)
    written = write_cells(arguments.out, cell, platform)
    names = ", ".join(map(str, written))
    print(f"{names}: {len(cell.splitlines())} lines, {len(plan.pieces)} orders")


_WORK = r"""
export MLFLOW_DISABLE_AGENT_HINT=1
CARDS=@CARDS@
PER_CARD=@PER_CARD@
@CUTOFF@
@THREADS@
@MONITOR@
PAT="checkpoint transient/sha256/[0-9a-f]+ sha256:[0-9a-f]+"
finished() { grep -q "^\[$1\] exit 0" $W/results.txt 2>/dev/null; }
seen() {
  cat "$W/$1".log.* "$W/$1".log 2>/dev/null | grep -oE "$PAT" | tail -1 | cut -d' ' -f2-
}
named() { awk -v k="$2" '$1 == k { print $2, $3 }' "$W/$1.txt"; }
piece() {  # label order
  local ref start status args said
  for attempt in 1 2; do
    args=(--order "durable/sha256/$2" "sha256:$2" --device @DEVICE@)
    args+=(--progress-every @EVERY@ --commit "$C" --workspace $B@WORKERS_ARG@)
    ref=$(seen "$1")
    [ -n "$ref" ] || ref=$(named resume "$1")
    # shellcheck disable=SC2206
    [ -n "${ref// /}" ] && args+=(--resume-from $ref)
    # An earlier attempt's log is kept beside the new one, whether it failed or dropped.
    [ -f "$W/$1.log" ] && mv "$W/$1.log" "$W/$1.log.$(date +%s)$attempt"
    echo "$1 from ${ref:-the start}"
    start=$(date +%s)
    limited python -m emblema.entrypoints.cli.pretrain run "${args[@]}" \
      > "$W/$1.out" 2> "$W/$1.log"
    status=$?
    said=$(tr '\n' ' ' < "$W/$1.out")
    echo "[$1] exit $status in $(( $(date +%s) - start )) s: $said" >> $W/results.txt
    [ $status -eq 0 ] && return
    past_cutoff && return
  done
}
worker() {  # card, or nothing to share one
  local label
  [ -n "${1:-}" ] && export CUDA_VISIBLE_DEVICES=$1
  while read -r label; do
    finished "$label" && continue
    # The shell's exclusive create: uutils mkdir (Ubuntu 26.04) lets two through.
    ( set -C; : > $T/claimed/$label ) 2>/dev/null || continue
    if past_cutoff; then
      echo "[$label] not claimed: past the cutoff" >> $W/results.txt; continue
    fi
    piece "$label" "$(awk -v k="$label" '$1 == k { print $2 }' $W/orders.txt)"
  done < $W/pieces.txt
}
mkdir -p $T/claimed
for w in $(seq 0 $(( CARDS * PER_CARD - 1 ))); do
  if [ "$CARDS" -gt 1 ]; then worker $(( w % CARDS )) & else worker & fi
done
wait
@STOP@
if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then echo quit | nvidia-cuda-mps-control; fi
echo "=== results (paste back)"; cat $W/results.txt
echo "=== failed"; grep -v "exit 0" $W/results.txt || echo none
echo "=== not finished, with the checkpoint to continue from"
while read -r label; do
  finished "$label" || echo "$label $(seen "$label")"
done < $W/pieces.txt
@UTILISATION@
echo "took $(( $(date +%s) - START )) s"
exit 0
"""


if __name__ == "__main__":  # pragma: no cover
    main()
