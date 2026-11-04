"""Write the Colab cell that fulfils a batch of campaign orders on one card.

One GPU, many processes under CUDA MPS. Workers claim the next piece of work (one order's lane,
``campaign_run --lane i/N``) from a list kept in the plan's order, so the plan puts the longest
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
    pieces = [["label", 24], ...]     # order label and its lanes, longest first

The first cell of the notebook (store secrets from Colab's own store) and the Drive mount run first.
This refuses to write a cell from any commit but the orders', and a cell that breaks its own rules.

    uv run scripts/colab_cell.py --plan plan.toml --out cell.sh
"""

import argparse
import re
import subprocess
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

LONGEST_LINE = 90
NETWORK_STEP = re.compile(r"git clone|pip install|fetch_corpora|curl |wget |python .*prefetch")
# Set by the cell itself, or too long for a line a notebook shows whole.
LEFT_TO_THE_CELL = ("BACKBONE", "WORKSPACE", "CORPORA", "DEVICE")


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


def read_plan(path: Path) -> CellPlan:
    """The plan at ``path``, its files read relative to it.

    Raises:
        SystemExit: If the orders do not name their commit, a piece has no order or no weights, or
            the labels' key does not match their digest.
    """
    plan = tomllib.loads(path.read_text())
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


def cell_of(plan: CellPlan) -> str:
    """The cell's text, ``%%bash`` first."""
    labels = sorted({piece.label for piece in plan.pieces})
    parts = [
        _HEAD.format(name=plan.name, commit=plan.commit, extras=plan.extras),
        _labels_step(plan.labels) if plan.labels else "",
        _blocks_step(plan.blocks) if plan.blocks else "",
        _BEFORE_WORKERS,
        "\n".join(line for export in plan.worker_exports for line in _fitted(export)),
        _WORKERS.format(workers=plan.workers, threads=plan.threads),
        _heredoc("pieces", [f"{p.label} {p.lane}/{p.lanes}" for p in plan.pieces]),
        _heredoc("orders", [f"{label} {plan.orders[label]}" for label in labels]),
        _heredoc("weights", [f"{label} {plan.weights[label]}" for label in labels]),
        _TAIL,
    ]
    return "\n".join(part.strip("\n") for part in parts if part) + "\n"


def _fitted(export: str) -> list[str]:
    """An ``export`` line cut at its commas into lines a notebook shows whole.

    Raises:
        SystemExit: If the value holds a quote or a space, or an item too long for a line: cut
            there, it would break the cell or change the setting.
    """
    if len(export) <= LONGEST_LINE:
        return [export]
    name, value = export.removeprefix("export ").split("=", 1)
    items = value.split(",")
    room = LONGEST_LINE - 10
    if re.search(r"[\"'\s]", value) or any(len(item) > room for item in items):
        raise SystemExit(f"{name} cannot be cut at its commas into lines of {LONGEST_LINE}")
    parts: list[str] = [items[0]]
    for item in items[1:]:
        joined = f"{parts[-1]},{item}"
        if len(joined) > room:
            parts.append(item)
        else:
            parts[-1] = joined
    held = [f"P{index}={part}" for index, part in enumerate(parts)]
    together = ",".join(f"$P{index}" for index in range(len(parts)))
    return [*held, f'export {name}="{together}"']


def _heredoc(name: str, lines: Sequence[str]) -> str:
    return "\n".join((f"cat > $W/{name}.txt <<'LIST'", *lines, "LIST"))


def _labels_step(digest: str) -> str:
    return _LABELS.replace("@DIGEST@", digest)


def _blocks_step(digests: Sequence[str]) -> str:
    return _BLOCKS.replace("@DIGESTS@", " \\\n    ".join(digests))


def faults_of(cell: str) -> list[str]:
    """What breaks a cell's own rules: a line too long to show, a network step without a limit."""
    faults = []
    for number, line in enumerate(cell.splitlines(), start=1):
        if len(line) > LONGEST_LINE:
            faults.append(f"line {number} has {len(line)} characters")
        if (
            NETWORK_STEP.search(line)
            and not line.lstrip().startswith("#")
            and "timeout " not in line
        ):
            faults.append(f"line {number} is a network step without a limit: {line.strip()}")
    return faults


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
    arguments = parser.parse_args(argv)
    plan = read_plan(arguments.plan)
    at = head or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if at != plan.commit:
        raise SystemExit(f"the tree is at {at}, the orders at {plan.commit}")
    cell = cell_of(plan)
    faults = faults_of(cell)
    if faults:
        raise SystemExit("the cell breaks its own rules:\n" + "\n".join(faults))
    arguments.out.write_text(cell)
    print(f"{arguments.out}: {len(cell.splitlines())} lines, {len(plan.pieces)} pieces")


_HEAD = """
%%bash
# {name}: orders on one card under CUDA MPS, processes claiming pieces in the
# plan's order. Commit {commit}.
# The store's secrets and the Drive mount first. Run again to resume.
set -u
ROOT=${{CELL_ROOT:-/content}}
if [ ! -d $ROOT/drive/MyDrive ]; then echo "mount Drive first"; exit 0; fi
NET=${{NET_TIMEOUT:-600}}
NET_BLOCKS=${{NET_BLOCKS_TIMEOUT:-2400}}
START=$(date +%s)
T=$(mktemp -d)
cd $ROOT
CLONED=no
for attempt in 1 2 3; do
  rm -rf emblema
  if timeout $NET git clone -q https://github.com/lstasiak/emblema; then
    CLONED=yes; break
  fi
  echo "clone attempt $attempt did not finish"
done
if [ "$CLONED" != yes ]; then echo "repository not cloned"; exit 0; fi
cd emblema
C={commit}
if ! git checkout -q $C; then echo "commit not on the remote"; exit 0; fi
git log -1 --format='%h %s'
INSTALLED=no
for attempt in 1 2 3; do
  if timeout $NET pip install -q -e ".[{extras}]" 2>/dev/null; then
    INSTALLED=yes; break
  fi
  echo "install attempt $attempt did not finish"
done
if [ "$INSTALLED" != yes ]; then echo "package not installed"; exit 0; fi
W=$ROOT/drive/MyDrive/{name}
B=$ROOT/blocks
mkdir -p $W $B
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $W/machine.txt
nproc | sed 's/^/vCPUs: /' | tee -a $W/machine.txt
"""

_LABELS = """
# The labels the tasks read, from one small archive of the store.
mkdir -p data/raw
cat > $T/labels.py <<'PY'
import io, sys, tarfile
from emblema.config.settings import Settings
from emblema.entrypoints.configured import configured_store
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
digest = sys.argv[1]
ref = ArtifactRef(
    key=f"transient/sha256/{digest}", checksum=Checksum.parse(f"sha256:{digest}")
)
data = configured_store(Settings()).get(ref)
with tarfile.open(fileobj=io.BytesIO(data)) as tar:
    tar.extractall("data/raw", filter="data")
print("labels:", len(data), "bytes")
PY
LD=@DIGEST@
GOT=no
for attempt in 1 2 3; do
  if timeout $NET python $T/labels.py $LD; then GOT=yes; break; fi
  echo "labels attempt $attempt did not finish"
done
if [ "$GOT" != yes ]; then echo "labels not fetched"; exit 0; fi
"""

_BLOCKS = """
# The blocks once, before any worker: each worker fetching them would pay the transfer.
cat > $T/prefetch.py <<'PY'
import sys
from pathlib import Path
from emblema.catalog.contracts.published_corpus_manifest_json import (
    PublishedCorpusManifestJson,
)
from emblema.config.settings import Settings
from emblema.entrypoints.configured import configured_store
from emblema.shared.adapters.windows.block_workspace import BlockWorkspace
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
store = configured_store(Settings())
workspace = BlockWorkspace(store, Path(sys.argv[1]))
for digest in sys.argv[2:]:
    ref = ArtifactRef(f"durable/sha256/{digest}", Checksum.parse(f"sha256:{digest}"))
    manifest = PublishedCorpusManifestJson().decode(store.get(ref))
    path = workspace.fetched(manifest.block)
    print(f"{manifest.corpus}: {path.stat().st_size / 2**30:.1f} GB", flush=True)
PY
FETCHED=no
for attempt in 1 2 3; do
  if timeout $NET_BLOCKS python $T/prefetch.py $B \\
    @DIGESTS@
  then FETCHED=yes; break; fi
  echo "blocks attempt $attempt did not finish"
done
if [ "$FETCHED" != yes ]; then echo "blocks not fetched"; exit 0; fi
"""

_BEFORE_WORKERS = """
echo "tree: $(git status --porcelain | wc -l) changes"
export MLFLOW_DISABLE_AGENT_HINT=1
export EMBLEMA_WORKER__WORKSPACE=$B
export EMBLEMA_WORKER__CORPORA=data/raw
"""

_WORKERS = """
export EMBLEMA_WORKER__DEVICE=cuda
WORKERS=${{WORKERS:-{workers}}}
# Many processes on one machine: each keeps to a few threads rather than all of them.
export OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}
# A fixed place, so a cell run again on the machine finds the daemon it started.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
if nvidia-cuda-mps-control -d 2>$W/mps.err; then echo "MPS: started"
else echo "MPS: unavailable"; unset CUDA_MPS_PIPE_DIRECTORY; fi
last_ref() {{  # name: the newest result answered in its logs, for a resume
  cat "$W/$1".log* 2>/dev/null | grep -oE "answered: durable/[^ ]+ sha256:[0-9a-f]+" \\
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
      python -m emblema.entrypoints.cli.campaign_run "${{args[@]}}" \\
      > "$W/$1.out" 2> "$W/$1.log"
    status=$?
    echo "[$1] exit $status in $(( $(date +%s) - start )) s: $(tail -n 1 "$W/$1.out")" \\
      >> $W/results.txt
    [ $status -eq 0 ] && return
  done
}}
digest() {{ awk -v k="$2" '$1 == k {{ print $2 }}' "$W/$1.txt"; }}
worker() {{
  local label lane name
  while read -r label lane; do
    name="$label-${{lane%%/*}}"
    finished "$name" && continue
    # The shell's exclusive create: uutils mkdir (Ubuntu 26.04) lets two through.
    ( set -C; : > $T/claimed/$name ) 2>/dev/null || continue
    piece "$name" "$(digest orders "$label")" "$(digest weights "$label")" "$lane"
  done < $W/pieces.txt
}}
"""

_TAIL = """
mkdir -p $T/claimed
Q=utilization.gpu,memory.used
nvidia-smi --query-gpu=$Q --format=csv,noheader,nounits -l 30 >> $W/smi.csv 2>&1 &
SMI=$!
# Out of the job table, so the final wait does not wait on this endless logger.
disown $SMI
free -g | head -2 > $W/memory.txt
# Its sleep outlives the kill, so it must not hold the output the notebook waits on.
( while sleep 300; do free -g | sed -n 2p >> $W/memory.txt; done ) >/dev/null 2>&1 &
MEM=$!
disown $MEM
for w in $(seq 1 $WORKERS); do worker & done
wait
kill $SMI $MEM 2>/dev/null
if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then echo quit | nvidia-cuda-mps-control; fi
echo "=== results (paste back)"; cat $W/results.txt
echo "=== failed"; grep -v "exit 0" $W/results.txt || echo none
echo "=== not finished"
while read -r label lane; do
  finished "$label-${lane%%/*}" || echo "$label-${lane%%/*}"
done < $W/pieces.txt
awk -F', ' '{u+=$1; n++} END {if (n) print "mean utilisation", u/n, "%"}' $W/smi.csv
echo "took $(( $(date +%s) - START )) s"
exit 0
"""


if __name__ == "__main__":  # pragma: no cover
    main()
