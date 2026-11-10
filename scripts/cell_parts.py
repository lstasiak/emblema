"""What every notebook cell shares, whichever work it does and whichever platform runs it.

The platform decides where the cell works and how it hands results back. A cell clones the commit
its orders were placed at, installs, and from then on every network step runs under a limit and
ends the cell cleanly when it runs out, so a stalled server does not hold a paid GPU. A cell
never exits non-zero, since a notebook would raise on it.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

LONGEST_LINE = 90
NETWORK_STEP = re.compile(
    r"git clone|git fetch|pip3? install|fetch_corpora|curl |wget |python .*fetch_published"
)
# A killed batch session keeps nothing, so a running piece is stopped this long after the cutoff.
GRACE_SECONDS = 3600
STORE_SETTINGS = ("ENDPOINT_URL", "REGION", "BUCKET", "KEY_PREFIX", "ACCESS_KEY", "SECRET_KEY")


@dataclass(frozen=True, kw_only=True)
class Platform:
    """Where a cell runs.

    Attributes:
        name: As the command line names it.
        root: Where the repository is cloned.
        guard: What must hold before anything is spent, as bash, or nothing.
        work: The directory of results, surviving the session as the platform allows.
        before: What the person does before the cell, for the cell's own header.
        session_hours: After how many hours no new piece is claimed, or ``None``.
        store_secrets: The Python of the cell that loads the store's settings, or ``None``.
        release: The Python of the cell that gives the card back, or ``None``.
        device: What the training process is told to run on.
        hosted: A script on this machine rather than a notebook cell on a rented one.
        shares_by_mps: Whether processes share one card under CUDA MPS.
        needs_cards: Whether a plan must name the cards its workers are spread over.
        reads_raw: Whether the raw corpora and labels are already on the machine.
    """

    name: str
    root: str
    guard: str
    work: str
    before: str
    session_hours: float | None
    store_secrets: str | None
    release: str | None
    device: str = "cuda"
    hosted: bool = False
    shares_by_mps: bool = False
    needs_cards: bool = False
    reads_raw: bool = False


_NAMES = ", ".join(f'"{name}"' for name in STORE_SETTINGS)

COLAB = Platform(
    name="colab",
    root="/content",
    guard='if [ ! -d $ROOT/drive/MyDrive ]; then echo "mount Drive first"; exit 0; fi',
    work="$ROOT/drive/MyDrive/@NAME@",
    before="The store's secrets and the Drive mount first. Run again to resume.",
    session_hours=None,
    store_secrets=(
        "import os\n\nfrom google.colab import userdata\n\n"
        f"for name in ({_NAMES}):\n"
        '    key = f"EMBLEMA_ARTIFACT_STORE__{name}"\n'
        "    os.environ[key] = userdata.get(key)\n"
        'print("store configured")\n'
    ),
    release=(
        "# Give the card back once the results are pasted: the run is on Drive and in the store.\n"
        "from google.colab import runtime\n\nruntime.unassign()\n"
    ),
    shares_by_mps=True,
)

# A batch session ends at twelve hours and takes its output with it only if it ends by itself.
KAGGLE = Platform(
    name="kaggle",
    root="/tmp",
    guard="",
    work="${CELL_WORK:-/kaggle/working}/@NAME@",
    before="The store's secrets first; then Save & Run All.",
    session_hours=10.5,
    store_secrets=(
        "import os\n\nfrom kaggle_secrets import UserSecretsClient\n\n"
        "secrets = UserSecretsClient()\n"
        f"for name in ({_NAMES}):\n"
        '    key = f"EMBLEMA_ARTIFACT_STORE__{name}"\n'
        "    os.environ[key] = secrets.get_secret(key)\n"
        'print("store configured")\n'
    ),
    release=None,
    needs_cards=True,
)

# Commits come first: orders refuse a tree that is not the one they were placed at.
MAC = Platform(
    name="mac",
    root="$(git rev-parse --show-toplevel)",
    guard="",
    work="$ROOT/data/report/@NAME@",
    before="Commit first; start it detached. Run again to resume.",
    session_hours=None,
    store_secrets=None,
    release=None,
    device="mps",
    hosted=True,
    reads_raw=True,
)

PLATFORMS = {platform.name: platform for platform in (COLAB, KAGGLE, MAC)}


def head(platform: Platform, *, title: str, name: str, commit: str, extras: str) -> str:
    """The cell up to its first piece of work: the checkout, the install and the directories."""
    return (
        (_HOST_HEAD if platform.hosted else _HEAD)
        .replace("@TITLE@", title)
        .replace("@COMMIT@", commit)
        .replace("@BEFORE@", platform.before)
        .replace("@ROOT@", platform.root)
        .replace("@GUARD@", platform.guard)
        .replace("@WORK@", platform.work.replace("@NAME@", name))
        .replace("@EXTRAS@", extras)
    )


def cutoff_line(platform: Platform) -> str:
    """When a piece stops being claimed, and when a running one is stopped; empty if never."""
    if platform.session_hours is None:
        return "CUTOFF=\nHARD_STOP="
    seconds = int(platform.session_hours * 3600)
    return (
        f"CUTOFF=$(( START + ${{SESSION_SECONDS:-{seconds}}} ))\n"
        f"HARD_STOP=$(( CUTOFF + ${{GRACE_SECONDS:-{GRACE_SECONDS}}} ))"
    )


def fitted(export: str) -> list[str]:
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


def heredoc(name: str, lines: Sequence[str]) -> str:
    return "\n".join((f"cat > $W/{name}.txt <<'LIST'", *lines, "LIST"))


def faults_of(cell: str) -> list[str]:
    """What breaks a cell's own rules: a line too long to show, a network step without a limit."""
    faults = []
    for number, line in enumerate(cell.splitlines(), start=1):
        if len(line) > LONGEST_LINE:
            faults.append(f"line {number} has {len(line)} characters")
        step = NETWORK_STEP.search(line)
        limit = line.find("timeout ")
        if step and not line.lstrip().startswith("#") and not 0 <= limit < step.start():
            faults.append(f"line {number} is a network step without a limit: {line.strip()}")
    return faults


def monitors(platform: Platform, *, every: int) -> tuple[str, str, str]:
    """The card's logging, for a machine with such tools: start, stop and the mean it read."""
    if platform.hosted:
        return "", "", ""
    return (
        _MONITOR.replace("@EVERY@", str(every)),
        "kill $SMI 2>/dev/null",
        _UTILISATION,
    )


def blocks_step(digests: Sequence[str]) -> str:
    """Fetch the publications once, before any process needs them."""
    return _BLOCKS.replace("@DIGESTS@", " \\\n    ".join(digests))


def write_cells(out: Path, cell: str, platform: Platform) -> list[Path]:
    """The cell and, beside it, the cells that come before and after it on the platform.

    Raises:
        SystemExit: If the cell breaks its own rules; nothing is written.
    """
    faults = faults_of(cell)
    if faults:
        raise SystemExit("the cell breaks its own rules:\n" + "\n".join(faults))
    written = [out]
    out.write_text(cell)
    if platform.hosted:
        out.chmod(0o755)
    for suffix, text in (("secrets", platform.store_secrets), ("release", platform.release)):
        if text is not None:
            beside = out.with_name(f"{out.stem}-{suffix}.py")
            beside.write_text(text)
            written.append(beside)
    return written


_MONITOR = """
Q=utilization.gpu,memory.used
nvidia-smi --query-gpu=$Q --format=csv,noheader,nounits -l @EVERY@ >> $W/smi.csv 2>&1 &
SMI=$!
# Out of the job table, so the final wait does not wait on this endless logger.
disown $SMI
"""

_UTILISATION = """
awk -F', ' '{u+=$1; n++} END {if (n) print "mean utilisation", u/n, "%"}' $W/smi.csv
"""

_HOST_HEAD = """
#!/bin/bash
# @TITLE@
# Commit @COMMIT@.
# @BEFORE@
set -u
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate > /dev/null; then
  CAFFEINATED=1 exec caffeinate -is /bin/bash "$0" "$@"
fi
ROOT=${CELL_ROOT:-@ROOT@}
cd $ROOT
C=@COMMIT@
AT=$(git rev-parse HEAD)
if [ "$AT" != "$C" ]; then echo "the tree is at $AT, the orders at $C"; exit 0; fi
if [ -n "$(git status --porcelain)" ]; then
  echo "the tree is dirty"; exit 0
fi
NET=${NET_TIMEOUT:-600}
NET_BLOCKS=${NET_BLOCKS_TIMEOUT:-2400}
START=$(date +%s)
T=$(mktemp -d)
mkdir -p $T/bin
cat > $T/bin/python <<'SH'
#!/bin/bash
exec uv run --env-file .env.r2 python "$@"
SH
cat > $T/bin/timeout <<'SH'
#!/bin/bash
seconds=$1; shift
exec perl -e 'alarm shift; exec @ARGV' "$seconds" "$@"
SH
chmod +x $T/bin/python $T/bin/timeout
export PATH=$T/bin:$PATH
W=@WORK@
B=$ROOT/data/artifacts
mkdir -p $W $B
"""

LIMITED = """
past_cutoff() { [ -n "$CUTOFF" ] && [ "$(date +%s)" -ge "$CUTOFF" ]; }
limited() {  # the command, stopped before a batch session would lose its output
  if [ -z "$HARD_STOP" ]; then "$@"; return; fi
  local left=$(( HARD_STOP - $(date +%s) ))
  timeout $(( left > 0 ? left : 1 )) "$@"
}
"""

_HEAD = """
%%bash
# @TITLE@
# Commit @COMMIT@.
# @BEFORE@
set -u
ROOT=${CELL_ROOT:-@ROOT@}
@GUARD@
NET=${NET_TIMEOUT:-600}
NET_BLOCKS=${NET_BLOCKS_TIMEOUT:-2400}
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
C=@COMMIT@
if ! git checkout -q $C; then echo "commit not on the remote"; exit 0; fi
git log -1 --format='%h %s'
INSTALLED=no
for attempt in 1 2 3; do
  if timeout $NET pip install -q -e ".[@EXTRAS@]" 2>/dev/null; then
    INSTALLED=yes; break
  fi
  echo "install attempt $attempt did not finish"
done
if [ "$INSTALLED" != yes ]; then echo "package not installed"; exit 0; fi
W=@WORK@
B=$ROOT/blocks
mkdir -p $W $B
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $W/machine.txt
nproc | sed 's/^/vCPUs: /' | tee -a $W/machine.txt
echo "tree: $(git status --porcelain | wc -l) changes"
"""

_BLOCKS = """
# The blocks once, before any worker; an attempt continues what the last one received.
FETCHED=no
for attempt in 1 2 3; do
  if timeout $NET_BLOCKS python scripts/fetch_published.py blocks $B \\
    @DIGESTS@ 2>> $W/block.log
  then FETCHED=yes; break; fi
  echo "blocks attempt $attempt did not finish (see block.log)"
done
if [ "$FETCHED" != yes ]; then
  tail -3 $W/block.log; echo "blocks not fetched"; exit 0
fi
"""

MPS = """
# A fixed place, so a cell run again on the machine finds the daemon it started.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
if pgrep -x nvidia-cuda-mps-control > /dev/null; then echo "MPS: already running"
elif nvidia-cuda-mps-control -d 2>$W/mps.err; then echo "MPS: started"
else echo "MPS: unavailable"; unset CUDA_MPS_PIPE_DIRECTORY; fi
"""
