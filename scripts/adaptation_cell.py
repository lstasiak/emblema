"""Write the Colab cell that fine-tunes the published forecaster and answers cells with each fit.

The inputs — the hourly means and the cells, which no publication holds — come as one transient
archive ``put_archive.py`` wrote; the forecaster's weights are fetched once, under a limit, and
every process after runs offline. Fits run several to a card under CUDA MPS; each piece's command
is ``fleet_chronos2_adapted.py`` with the plan's arguments, ``@IN@`` standing for the inputs,
``@F@`` for the fits on the machine's disk and ``@W@`` for the directory of results on Drive. The
answers run, as many at once, once the fits have ended; an answer reads its fits' checkpoints
(``--fits @F@/fits/<fit's label>...``) only once each of those fits ended well, and again after
any of them is redone. A cell run again skips what is done and redoes what failed or was cut
short, a fit from its first step. What came of it, with each checkpoint's adapters but no full
weights or optimiser states, goes back to the store as one transient archive, whose key is
printed to paste back. Every network step runs under a limit and the cell never exits non-zero.

The plan:

    name = "emblema-adaptation"    # the directory of results on Drive
    commit = "<40 hex>"            # the tree the pieces run at, on the remote
    inputs = "<digest>"            # put_archive.py's archive of the inputs
    per_card = 3                   # fits sharing the card
    fits = "fits.txt"              # "label arguments...", one piece a line
    answers = "answers.txt"        # the same, run after the fits

    uv run scripts/adaptation_cell.py --plan plan.toml --out cell.sh
"""

import argparse
import re
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cell_parts import (  # noqa: E402
    COLAB,
    LONGEST_LINE,
    MPS,
    head,
    heredoc,
    monitors,
    write_cells,
)

COMMIT = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"[0-9a-f]{64}")
PIECE = re.compile(r"[A-Za-z0-9_.-]+")
# A full model's weights and the optimiser's states stay on the machine; the adapters, a few MB a
# checkpoint, go back, so that a fit can be answered again without being run again.
WEIGHTS = ("*/model.safetensors", "*.bin", "*.pt", "*.pth")
ADAPTERS = "adapter_model.safetensors"
FORECASTER = ("chronos-forecasting==2.3.2", "peft==0.18.1")


@dataclass(frozen=True, kw_only=True)
class AdaptationPlan:
    """All a cell states.

    Attributes:
        name: The directory the cell keeps its results in.
        commit: The commit the pieces run at.
        inputs: The digest of the inputs' archive.
        per_card: How many fits share the card at once.
        fits: Each fit's label and arguments, in the order they are claimed.
        answers: Each answer's label and arguments, run after every fit.
    """

    name: str
    commit: str
    inputs: str
    per_card: int
    fits: tuple[str, ...]
    answers: tuple[str, ...]


def pieces_of(text: str) -> tuple[str, ...]:
    """The non-empty, uncommented lines, each a label and its arguments.

    Raises:
        SystemExit: If a label is malformed or named twice, or a line holds a quote or a
            backslash, which the cell's shell would read.
    """
    lines = tuple(
        line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    )
    labels = [line.split()[0] for line in lines]
    bad = [label for label in labels if PIECE.fullmatch(label) is None]
    if bad or len(set(labels)) != len(labels):
        raise SystemExit(f"labels must be plain and named once: {bad or labels}")
    if any(re.search(r"[\"'`$\\]", line) for line in lines):
        raise SystemExit("a piece's arguments are words: no quote, no shell")
    return lines


def read_plan(path: Path) -> AdaptationPlan:
    """The plan at ``path``, its files read relative to it.

    Raises:
        SystemExit: If the commit, the inputs' digest or the name is malformed, nothing is fitted,
            or a fit and an answer share a label.
    """
    plan = tomllib.loads(path.read_text())
    if COMMIT.fullmatch(plan["commit"]) is None or DIGEST.fullmatch(plan["inputs"]) is None:
        raise SystemExit("the plan names a commit by its 40 hex and the inputs by their digest")
    if PIECE.fullmatch(plan["name"]) is None:
        raise SystemExit(f"{plan['name']!r} is no plain directory name")
    fits = pieces_of((path.parent / plan["fits"]).read_text())
    answers = pieces_of((path.parent / plan["answers"]).read_text())
    if not fits:
        raise SystemExit("the plan fits nothing")
    shared = {f.split()[0] for f in fits} & {a.split()[0] for a in answers}
    if shared:
        raise SystemExit(f"a fit and an answer share the labels {sorted(shared)}")
    per_card = plan.get("per_card", 1)
    if per_card < 1:
        raise SystemExit("a card holds at least one fit")
    return AdaptationPlan(
        name=plan["name"],
        commit=plan["commit"],
        inputs=plan["inputs"],
        per_card=per_card,
        fits=fits,
        answers=answers,
    )


def wrapped(piece: str) -> list[str]:
    """A piece over lines a notebook shows whole, each but the last ending in a backslash."""
    lines = [""]
    for word in piece.split():
        if lines[-1] and len(lines[-1]) + len(word) + 3 > LONGEST_LINE:
            lines.append("")
        lines[-1] = f"{lines[-1]} {word}".lstrip()
    return [f"{line} \\" for line in lines[:-1]] + lines[-1:]


def cell_of(plan: AdaptationPlan) -> str:
    """The cell's text, ``%%bash`` first."""
    start, stop, utilisation = monitors(COLAB, every=30)
    parts = [
        head(
            COLAB, title="Fine-tune the forecaster.", name=plan.name, commit=plan.commit, extras=""
        ),
        _INPUTS.replace("@INPUTS@", plan.inputs).replace("@FORECASTER@", " ".join(FORECASTER)),
        heredoc("fits", [line for piece in plan.fits for line in wrapped(piece)]),
        heredoc("answers", [line for piece in plan.answers for line in wrapped(piece)]),
        MPS if plan.per_card > 1 else "",
        _WORK.replace("@PER_CARD@", str(plan.per_card))
        .replace("@MONITOR@", start.strip("\n"))
        .replace("@STOP@", stop)
        .replace("@UTILISATION@", utilisation.strip("\n"))
        .replace("@WEIGHTS@", " ".join(f"'{glob}'" for glob in WEIGHTS))
        .replace("@ADAPTERS@", ADAPTERS),
    ]
    return "\n".join(part.strip("\n") for part in parts if part) + "\n"


def main(argv: Sequence[str] | None = None, *, head_commit: str | None = None) -> None:
    """Write the plan's cell, if the tree stands at the plan's commit and the cell is sound.

    Raises:
        SystemExit: If the tree is at another commit or the cell breaks its own rules.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    plan = read_plan(arguments.plan)
    at = head_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if at != plan.commit:
        raise SystemExit(f"the tree is at {at}, the plan at {plan.commit}")
    cell = cell_of(plan)
    written = write_cells(arguments.out, cell, COLAB)
    names = ", ".join(map(str, written))
    print(f"{names}: {len(cell.splitlines())} lines, {len(plan.fits)} fits")


_INPUTS = r"""
I=$ROOT/inputs
# The fits' weights on the machine's own disk: several GB, unread once answered.
F=$ROOT/fits
INPUTS=@INPUTS@
GOT=no
for attempt in 1 2 3; do
  if timeout $NET pip install -q @FORECASTER@
  then GOT=yes; break; fi
  echo "forecaster install attempt $attempt did not finish"
done
if [ "$GOT" != yes ]; then echo "forecaster not installed"; exit 0; fi
pip list 2>/dev/null | grep -iE '^(torch|transformers|accelerate|peft|chronos)' \
  | tee -a $W/machine.txt
python -c 'import torch; print("capability", torch.cuda.get_device_capability())' \
  | tee -a $W/machine.txt
GOT=no
for attempt in 1 2 3; do
  if timeout $NET_BLOCKS python scripts/fetch_published.py archive $INPUTS $I \
    2>> $W/inputs.log
  then GOT=yes; break; fi
  echo "inputs attempt $attempt did not finish (see inputs.log)"
done
if [ "$GOT" != yes ]; then tail -3 $W/inputs.log; echo "inputs not fetched"; exit 0; fi
GOT=no
for attempt in 1 2 3; do
  if timeout $NET hf download amazon/chronos-2 > /dev/null; then GOT=yes; break; fi
  echo "weights attempt $attempt did not finish"
done
if [ "$GOT" != yes ]; then echo "weights not fetched"; exit 0; fi
export HF_HUB_OFFLINE=1
"""

_WORK = r"""
PER_CARD=@PER_CARD@
@MONITOR@
mkdir -p $W/logs $W/answers $F/fits $T/claimed
touch $W/results.txt
# The line of results.txt where a label last ended well, or 0: a later failure undoes it.
done_at() {
  grep -n "^\[$1\] exit" $W/results.txt | tail -1 \
    | awk -F: '/\] exit 0 / { print $1; found = 1 } END { if (!found) print 0 }'
}
# The fits of this cell an answer reads: each @F@ path after --fits, up to the next flag.
fits_of() {
  local tail=${1#*--fits } path
  [ "$tail" = "$1" ] && return 0
  set -f
  for path in ${tail%% --*}; do
    case $path in @F@/fits/*) echo "${path#@F@/fits/}" ;; esac
  done
  set +f
}
# A fit is done once it ends well; an answer once it ended well after all its fits did.
current() {  # label rest
  local fit at
  at=$(done_at "$1")
  [ "$at" -gt 0 ] || return 1
  for fit in $(fits_of "$2"); do [ "$at" -gt "$(done_at "$fit")" ] || return 1; done
}
# Whether every fit an answer reads has ended well.
ready() {  # rest
  local fit
  for fit in $(fits_of "$1"); do [ "$(done_at "$fit")" -gt 0 ] || return 1; done
}
piece() {  # kind label arguments...
  local kind=$1 label=$2 start status
  shift 2
  start=$(date +%s)
  python -I scripts/fleet_chronos2_adapted.py "$kind" "$@" --device cuda \
    > "$W/logs/$label.out" 2> "$W/logs/$label.log"
  status=$?
  echo "[$label] exit $status in $(( $(date +%s) - start )) s" >> $W/results.txt
}
worker() {  # list
  local line label rest args
  # Without -r: a backslash at a line's end joins a piece written over several lines.
  while read line; do
    label=${line%% *}; rest=${line#* }
    current "$label" "$rest" && continue
    # An answer waits on its fits ending well; their own lines say why it did not run.
    ready "$rest" || continue
    # The shell's exclusive create: uutils mkdir (Ubuntu 26.04) lets two through.
    ( set -C; : > $T/claimed/$label ) 2>/dev/null || continue
    args=${rest//@IN@/$I}; args=${args//@W@/$W}; args=${args//@F@/$F}
    # Words only (the plan refuses quotes): split, never globbed.
    set -f; set -- $args; set +f
    piece "$1" "$label" "${@:2}"
  done < "$W/$1.txt"
}
for kind in fits answers; do
  for w in $(seq 1 $PER_CARD); do worker $kind & done
  wait
done
@STOP@
if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then echo quit | nvidia-cuda-mps-control; fi
echo "=== results"; cat $W/results.txt
echo "=== not done (failed, waiting on a fit, or not reached)"
LEFT=0
while read line; do
  label=${line%% *}; rest=${line#* }
  if ! current "$label" "$rest"; then echo "$label"; LEFT=$(( LEFT + 1 )); fi
done < <(cat $W/fits.txt $W/answers.txt)
[ "$LEFT" -eq 0 ] && echo none
# The fits' records and adapters beside the answers; the copy keeps paths on BSD and GNU.
( cd $F && find fits \( -name '*.json' -o -name @ADAPTERS@ \) \
  | while read -r kept; do mkdir -p "$W/${kept%/*}" && cp "$kept" "$W/$kept"; done )
GOT=no
for attempt in 1 2 3; do
  if timeout $NET python scripts/put_archive.py --root $W answers fits logs results.txt \
    machine.txt --exclude @WEIGHTS@ > $T/key.txt
  then GOT=yes; break; fi
  echo "results attempt $attempt did not finish"
done
if [ "$GOT" = yes ]; then echo "=== results archive (paste back)"; cat $T/key.txt
else echo "results not archived: they are on Drive in $W"; fi
@UTILISATION@
echo "took $(( $(date +%s) - START )) s"
exit 0
"""


if __name__ == "__main__":  # pragma: no cover
    main()
