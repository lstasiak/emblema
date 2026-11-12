"""Write the cell that fulfils a batch of campaign orders on Colab, Kaggle or the Mac.

On Colab one GPU, many processes under CUDA MPS; on Kaggle each worker keeps to one of the cards,
and no piece is claimed once the batch session nears its end. Workers claim the next piece of
work (one order's lane, ``campaign_run --lane i/N``) from a list kept in the plan's order, so the
plan puts the longest
first. Blocks a task reads are fetched once before any worker starts. Run again, a cell skips the
pieces that ended and resumes a cut one from its last answer on Drive. Every network step runs
under a limit and ends the cell cleanly when it runs out, so a stalled server does not hold a paid
GPU. The cell never exits non-zero, since a notebook would raise on it.

The plan names the commit's orders and what each needs:

    name = "emblema-screen"           # the directory on Drive
    orders = "orders.txt"             # "# from <commit>", then lines "label id digest"
    weights = "campaigns.txt"         # lines "label id digest-of-the-backbone"
    worker_env = "worker-env.sh"      # the registered worker settings, as `export` lines
    blocks = ["<manifest digest>"]    # publications to fetch once, before the workers
    labels = "esa-labels.txt"         # optional: "key sha256:digest" of a tar of task labels
    workers = 24                      # processes on the card
    threads = 2                       # threads per process
    extras = "ml"                     # what the cell installs
    fetch = [["skab", "opssat_ad"]]   # optional: arguments of each corpora fetch
    answers = ["skab", "opssat_ad"]   # the corpora whose answers the tasks read, or []
    cards = 2                         # Kaggle: the cards the workers are spread over
    pieces = [["label", 24], ...]     # order label and its lanes, longest first

The cells before and after this one (the store's secrets, the card given back) are written beside
it. This refuses to write a cell from any commit but the orders', a cell that breaks its own
rules, and one whose tasks read answers that neither a fetch nor the labels bring: no block holds
a task's answers, so such a cell would start every worker only to fail each order at once.

    uv run scripts/colab_cell.py --plan plan.toml --out cell.sh [--platform kaggle|mac]
"""

import argparse
import re
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
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
    fitted,
    head,
    heredoc,
    monitors,
    write_cells,
)

# Set by the cell itself, or too long for a line a notebook shows whole.
LEFT_TO_THE_CELL = ("BACKBONE", "WORKSPACE", "CORPORA", "DEVICE")

# The corpora a worker reads answers of, by the name a task gives them, and the corpora fetch
# that brings each one's files.
FETCHED_AS = {
    "cmapss": "cmapss",
    "physionet2012": "physionet2012",
    "physionet2019": "physionet2019",
    "esa_ad_raw": "esa_ad",
    "skab": "skab",
    "opssat_ad": "opssat_ad",
}
# The corpora whose answers the labels archive carries, so a cell need not fetch the whole corpus.
IN_THE_LABELS = frozenset({"esa_ad_raw"})


@dataclass(frozen=True, kw_only=True)
class Piece:
    """A lane of an order.

    Attributes:
        label: The order's label.
        lane: Which lane, counted from zero.
        lanes: How many lanes the order is cut into.
    """

    label: str
    lane: int
    lanes: int


@dataclass(frozen=True, kw_only=True)
class CellPlan:
    """All a cell states.

    Attributes:
        name: The directory the cell keeps its results in on Drive.
        commit: The commit the orders were placed at.
        orders: The digest of each order, by label.
        weights: The digest of the backbone each order runs under, by label.
        worker_exports: The ``export`` lines of the registered worker settings.
        blocks: The publications to fetch before the workers start.
        labels: The digest of a tar of task labels, or ``None``.
        workers: How many processes share the card.
        threads: How many threads each process keeps to.
        extras: What the cell installs.
        pieces: The work, in the order it is claimed.
        answers: The corpora whose answers the tasks read.
        fetches: The arguments of each corpora fetch.
        cards: The cards the workers are spread over, or nought to share one under MPS.
    """

    name: str
    commit: str
    orders: dict[str, str]
    weights: dict[str, str]
    worker_exports: tuple[str, ...]
    blocks: tuple[str, ...]
    labels: str | None
    workers: int
    threads: int
    extras: str
    pieces: tuple[Piece, ...]
    answers: tuple[str, ...]
    fetches: tuple[tuple[str, ...], ...] = ()
    cards: int = 0


def read_plan(path: Path) -> CellPlan:
    """The plan at ``path``, its files read relative to it.

    Raises:
        SystemExit: If the orders do not name their commit, a piece has no order or no weights,
            the labels' key does not match their digest, or the plan does not say whose answers
            its tasks read.
    """
    plan = tomllib.loads(path.read_text())
    if "answers" not in plan:
        raise SystemExit("the plan names no answers: list the corpora its tasks read answers of")
    base = path.parent
    orders_text = (base / plan["orders"]).read_text()
    found = re.search(r"^# from ([0-9a-f]{40})$", orders_text, re.MULTILINE)
    if found is None:
        raise SystemExit(f"{plan['orders']} names no commit in a '# from <commit>' line")
    orders = _column(orders_text, 2)
    weights = _column((base / plan["weights"]).read_text(), 2)
    pieces: list[Piece] = []
    for label, lanes in plan["pieces"]:
        if label not in orders or label not in weights:
            raise SystemExit(f"no order or no weights for {label}")
        pieces += [Piece(label=label, lane=lane, lanes=lanes) for lane in range(lanes)]
    return CellPlan(
        name=plan["name"],
        commit=found[1],
        orders=orders,
        weights=weights,
        worker_exports=_worker_exports((base / plan["worker_env"]).read_text()),
        blocks=tuple(plan.get("blocks", ())),
        labels=_labels_digest((base / plan["labels"]).read_text()) if "labels" in plan else None,
        workers=plan["workers"],
        threads=plan["threads"],
        extras=plan["extras"],
        pieces=tuple(pieces),
        answers=tuple(plan["answers"]),
        fetches=tuple(tuple(each) for each in plan.get("fetch", ())),
        cards=plan.get("cards", 0),
    )


def _column(text: str, column: int) -> dict[str, str]:
    """The given column of each non-comment line, by its first."""
    rows = (line.split() for line in text.splitlines() if line and not line.startswith("#"))
    return {fields[0]: fields[column] for fields in rows if len(fields) > column}


def _worker_exports(text: str) -> tuple[str, ...]:
    """The settings the workers run under, without those the cell sets itself."""
    return tuple(
        line
        for line in text.splitlines()
        if line.startswith("export EMBLEMA_WORKER__")
        and not any(f"__{name}=" in line for name in LEFT_TO_THE_CELL)
    )


def _labels_digest(text: str) -> str:
    key, checksum = text.split()
    digest = checksum.removeprefix("sha256:")
    if key != f"transient/sha256/{digest}":
        raise SystemExit(f"the labels' key {key} does not match their digest {digest}")
    return digest


def cell_of(plan: CellPlan, platform: Platform = COLAB) -> str:
    """The cell's text, ``%%bash`` first.

    Raises:
        SystemExit: If the plan shares one card among workers on a platform without MPS, or its
            tasks read answers the cell does not bring.
    """
    if platform.needs_cards and plan.cards < 1:
        raise SystemExit(f"a {platform.name} cell spreads its workers over cards: none named")
    if not platform.reads_raw:
        _refuse_answers_not_brought(plan)
    labels = sorted({piece.label for piece in plan.pieces})
    title = "Campaign orders, workers claiming pieces in the plan's order."
    parts = [
        head(platform, title=title, name=plan.name, commit=plan.commit, extras=plan.extras),
        _labels_step(plan.labels) if plan.labels and not platform.reads_raw else "",
        "\n".join(_fetch_step(each) for each in plan.fetches if not platform.reads_raw),
        blocks_step(plan.blocks) if plan.blocks else "",
        _BEFORE_WORKERS,
        "\n".join(line for export in plan.worker_exports for line in fitted(export)),
        MPS if plan.cards == 0 and platform.shares_by_mps else "",
        LIMITED,
        _WORKERS.format(
            workers=plan.workers,
            threads=plan.threads,
            cards=plan.cards,
            cutoff=cutoff_line(platform),
            device=platform.device,
        ),
        heredoc("pieces", [f"{p.label} {p.lane}/{p.lanes}" for p in plan.pieces]),
        heredoc("orders", [f"{label} {plan.orders[label]}" for label in labels]),
        heredoc("weights", [f"{label} {plan.weights[label]}" for label in labels]),
        _tail(platform),
    ]
    return "\n".join(part.strip("\n") for part in parts if part) + "\n"


def _refuse_answers_not_brought(plan: CellPlan) -> None:
    unknown = sorted(set(plan.answers) - FETCHED_AS.keys())
    if unknown:
        raise SystemExit(f"no worker reads answers of {unknown}; it knows {sorted(FETCHED_AS)}")
    # A fetch that names no corpus fetches every one.
    named = [[each for each in fetch if not each.startswith("-")] for fetch in plan.fetches]
    missing = sorted(
        corpus
        for corpus in plan.answers
        if not (corpus in IN_THE_LABELS and plan.labels)
        and not any(not keys or FETCHED_AS[corpus] in keys for keys in named)
    )
    if missing:
        raise SystemExit(
            f"the tasks read answers of {missing}, which no block holds and no fetch or labels "
            f"bring; add a fetch of {sorted(FETCHED_AS[corpus] for corpus in missing)}"
        )


def _tail(platform: Platform) -> str:
    start, stop, utilisation = monitors(platform, every=30)
    return (
        _TAIL.replace("@MONITOR@\n", start.lstrip("\n"))
        .replace("@MEMORY@\n", _MEMORY.lstrip("\n") if not platform.hosted else "")
        .replace("@STOP@\n", f"{stop}\nkill $MEM 2>/dev/null\n" if stop else "")
        .replace("@UTILISATION@\n", utilisation.lstrip("\n"))
    )


def _labels_step(digest: str) -> str:
    return _LABELS.replace("@DIGEST@", digest)


def _fetch_step(arguments: Sequence[str]) -> str:
    return _FETCH.replace("@ARGUMENTS@", " ".join(arguments))


def main(argv: Sequence[str] | None = None, *, head: str | None = None) -> None:
    """Write the plan's cell, if the tree stands at the orders' commit and the cell is sound.

    Args:
        argv: The command line.
        head: The commit the tree stands at; asked of git unless given.

    Raises:
        SystemExit: If the tree is at another commit or the cell breaks its own rules.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--platform", choices=sorted(PLATFORMS), default=COLAB.name)
    arguments = parser.parse_args(argv)
    plan = read_plan(arguments.plan)
    at = head or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if at != plan.commit:
        raise SystemExit(f"the tree is at {at}, the orders at {plan.commit}")
    platform = PLATFORMS[arguments.platform]
    cell = cell_of(plan, platform)
    written = write_cells(arguments.out, cell, platform)
    print(
        f"{', '.join(map(str, written))}: {len(cell.splitlines())} lines, {len(plan.pieces)} pieces"
    )


_LABELS = """
# The labels the tasks read, from one small archive of the store.
GOT=no
for attempt in 1 2 3; do
  if timeout $NET python scripts/fetch_published.py labels \\
    @DIGEST@
  then GOT=yes; break; fi
  echo "labels attempt $attempt did not finish"
done
if [ "$GOT" != yes ]; then echo "labels not fetched"; exit 0; fi
"""

_FETCH = """
FETCHED=no
for attempt in 1 2 3; do
  if timeout $(( NET * 3 )) python scripts/fetch_corpora.py \\
    @ARGUMENTS@ >> $W/fetch.log 2>&1
  then FETCHED=yes; break; fi
  echo "corpora attempt $attempt did not finish (see fetch.log)"
done
if [ "$FETCHED" != yes ]; then
  tail -3 $W/fetch.log; echo "corpora not fetched"; exit 0
fi
"""

_BEFORE_WORKERS = """
export MLFLOW_DISABLE_AGENT_HINT=1
export EMBLEMA_WORKER__WORKSPACE=$B
export EMBLEMA_WORKER__CORPORA=data/raw
"""

_WORKERS = """
export EMBLEMA_WORKER__DEVICE={device}
WORKERS=${{WORKERS:-{workers}}}
CARDS={cards}
{cutoff}
# Many processes on one machine: each keeps to a few threads rather than all of them.
export OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}
last_ref() {{  # name: the newest result answered in its logs, for a resume
  cat "$W/$1".log.* "$W/$1".log 2>/dev/null \\
    | grep -oE "answered: durable/[^ ]+ sha256:[0-9a-f]+" \\
    | tail -1 | cut -d' ' -f2-
}}
finished() {{ grep -q "^\\[$1\\] exit 0" $W/results.txt 2>/dev/null; }}
piece() {{  # name order weights lane
  local start status ref args
  for attempt in 1 2; do
    args=(--order "durable/sha256/$2" "sha256:$2" --lane "$4")
    ref=$(last_ref "$1")
    # shellcheck disable=SC2206
    [ -n "$ref" ] && args+=(--resume $ref)
    [ -f "$W/$1.log" ] && mv "$W/$1.log" "$W/$1.log.$(date +%s)"
    start=$(date +%s)
    EMBLEMA_WORKER__BACKBONE="durable/sha256/$3@sha256:$3" \\
      limited python -m emblema.entrypoints.cli.campaign_run "${{args[@]}}" \\
      > "$W/$1.out" 2> "$W/$1.log"
    status=$?
    echo "[$1] exit $status in $(( $(date +%s) - start )) s: $(tail -n 1 "$W/$1.out")" \\
      >> $W/results.txt
    [ $status -eq 0 ] && return
    past_cutoff && return
  done
}}
digest() {{ awk -v k="$2" '$1 == k {{ print $2 }}' "$W/$1.txt"; }}
worker() {{  # card, or nothing to share one
  local label lane name
  [ -n "${{1:-}}" ] && export CUDA_VISIBLE_DEVICES=$1
  while read -r label lane; do
    name="$label-${{lane%%/*}}"
    finished "$name" && continue
    # The shell's exclusive create: uutils mkdir (Ubuntu 26.04) lets two through.
    ( set -C; : > $T/claimed/$name ) 2>/dev/null || continue
    if past_cutoff; then
      echo "[$name] not claimed: past the cutoff" >> $W/results.txt; continue
    fi
    piece "$name" "$(digest orders "$label")" "$(digest weights "$label")" "$lane"
  done < $W/pieces.txt
}}
"""

_MEMORY = """
free -g | head -2 > $W/memory.txt
# Its sleep outlives the kill, so it must not hold the output the notebook waits on.
( while sleep 300; do free -g | sed -n 2p >> $W/memory.txt; done ) >/dev/null 2>&1 &
MEM=$!
disown $MEM
"""

_TAIL = """
mkdir -p $T/claimed
@MONITOR@
@MEMORY@
for w in $(seq 0 $(( WORKERS - 1 ))); do
  if [ "$CARDS" -gt 0 ]; then worker $(( w % CARDS )) & else worker & fi
done
wait
@STOP@
if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then echo quit | nvidia-cuda-mps-control; fi
echo "=== results (paste back)"; cat $W/results.txt
# A lane run again appends its report, so only each lane's last one says how it ended.
echo "=== failed"
awk '{ k = substr($0, 1, index($0, "]")); if (!(k in last)) order[++n] = k; last[k] = $0 }
  END { for (i = 1; i <= n; i++) {
          k = order[i]
          if (substr(last[k], length(k) + 1, 8) != " exit 0 ") { print last[k]; f = 1 }
        }
        if (!f) print "none" }' $W/results.txt 2>/dev/null || echo none
echo "=== not finished"
while read -r label lane; do
  finished "$label-${lane%%/*}" || echo "$label-${lane%%/*}"
done < $W/pieces.txt
@UTILISATION@
echo "took $(( $(date +%s) - START )) s"
exit 0
"""


if __name__ == "__main__":  # pragma: no cover
    main()
