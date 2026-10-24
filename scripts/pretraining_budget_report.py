"""How many epochs a recipe trains: the epoch after which no term of its objective still learns.

Read off the log of one pretraining run with a cap well past where its terms stop falling. For
every epoch the log states each validation corpus's loss per term beside the trivial
predictor's — the hidden values (``reconstruction``) and, where the objective asks for them, the
cells' ``presence`` and ``count``. A term's cost at an epoch is the mean over corpora of its loss
over the trivial predictor's, as the run's own validation is a mean over corpora, so that no
corpus decides by the size of its values.

A term has stopped learning at the first epoch whose cost none of the next ``WINDOW`` epochs
lowers by more than ``TOLERANCE`` of itself. The budget is the latest such epoch over the terms:
the run kept to it trains every term to where it stopped learning here. A term that never stops
within the cap leaves no budget, and the cap is reported as too short. The epoch the run keeps —
the lowest mean cost of the values alone, which is all its validation reads — is reported beside
it, with how far it falls before the presence stops learning, since a kept epoch long before
that point would keep a backbone whose presence is not yet learnt.

The cap of this run is longer than the budget it gives, and its rate decays over the whole cap,
so at the budget's epoch this run still trained at a higher rate than a run capped there will:
a run whose rate is spent by then reaches at least this cost, which makes the budget a bound
from above rather than a guess.

    uv run scripts/pretraining_budget_report.py --log <run.log> [--out DIR]
    uv run scripts/pretraining_budget_report.py --report-only DIR
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The import below follows, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.reporting import table

WORKSPACE = REPO_ROOT / "data" / "report" / "pretraining-budget"
COSTS = "costs.csv"
WINDOW = 3
TOLERANCE = 0.01
# How many epochs the kept epoch may fall before the presence stops learning without a word.
KEPT_LAG = 2
VALUES = "reconstruction"
PRESENCE = "presence"

_EPOCH = re.compile(r"epoch (\d+) of (\d+) done in .*?; (?P<corpora>[a-z0-9_]+ validation .*)$")
_CORPUS = re.compile(
    r"(?P<corpus>[a-z0-9_]+) validation [0-9.]+, [0-9.]+ of the trivial predictor's"
    r"(?: \((?P<terms>[^)]*)\))?"
)
_TERM = re.compile(r"(?P<name>[a-z_]+) (?P<loss>[0-9.eE+-]+) of (?P<trivial>[0-9.eE+-]+)")


@dataclass(frozen=True)
class CostRow:
    """One term's validation loss on one corpus at one epoch, beside the trivial predictor's."""

    epoch: int
    corpus: str
    term: str
    loss: float
    trivial: float

    @property
    def cost(self) -> float:
        return self.loss / self.trivial


def costs_of(lines: Iterable[str]) -> list[CostRow]:
    """Every term of every validation corpus at every epoch the log reports as done.

    A run picked up from a checkpoint reports its epochs again from where it stood; the last
    report of an epoch is the one kept.

    Raises:
        ValueError: If an epoch's line names a corpus without its terms beside the trivial
            predictor's, so its cost cannot be read.
    """
    latest: dict[tuple[int, str, str], CostRow] = {}
    for line in lines:
        epoch_match = _EPOCH.search(line)
        if epoch_match is None:
            continue
        epoch = int(epoch_match.group(1))
        for corpus_match in _CORPUS.finditer(epoch_match.group("corpora")):
            corpus, terms = corpus_match.group("corpus"), corpus_match.group("terms")
            if terms is None:
                raise ValueError(
                    f"epoch {epoch}: {corpus} states no terms beside the trivial predictor's"
                )
            for term in _TERM.finditer(terms):
                row = CostRow(
                    epoch,
                    corpus,
                    term.group("name"),
                    float(term.group("loss")),
                    float(term.group("trivial")),
                )
                latest[(row.epoch, row.corpus, row.term)] = row
    return sorted(latest.values(), key=lambda row: (row.epoch, row.term, row.corpus))


def mean_costs(rows: Iterable[CostRow]) -> dict[str, dict[int, float]]:
    """Each term's cost at each epoch, the mean over corpora."""
    gathered: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        gathered[row.term][row.epoch].append(row.cost)
    return {
        term: {epoch: sum(costs) / len(costs) for epoch, costs in sorted(by_epoch.items())}
        for term, by_epoch in gathered.items()
    }


def stops_learning(
    costs: dict[int, float], *, window: int = WINDOW, tolerance: float = TOLERANCE
) -> int | None:
    """The first epoch whose cost none of the next ``window`` epochs lowers by over ``tolerance``.

    ``None`` where no epoch has that many epochs after it that leave it standing.
    """
    epochs = sorted(costs)
    for position, epoch in enumerate(epochs):
        after = epochs[position + 1 : position + 1 + window]
        if len(after) < window:
            return None
        if min(costs[later] for later in after) >= costs[epoch] * (1.0 - tolerance):
            return epoch
    return None


@dataclass(frozen=True)
class Budget:
    """What a run's validation says of how long its recipe trains.

    Attributes:
        stops: Each term's epoch of no further learning, ``None`` where it learns to the cap.
        epochs: The budget, the latest of ``stops``; ``None`` where a term never stops.
        kept: The epoch the run keeps, the lowest mean cost of the values.
        lag: Epochs from the kept one to the presence's stop, where both are known.
    """

    stops: dict[str, int | None]
    epochs: int | None
    kept: int
    lag: int | None

    @property
    def keeps_too_early(self) -> bool:
        return self.lag is not None and self.lag > KEPT_LAG


def budget_of(rows: Sequence[CostRow]) -> Budget:
    """The budget the rule gives for the run the rows were read from.

    Raises:
        ValueError: If the rows hold no cost of the values, which every run reports.
    """
    costs = mean_costs(rows)
    if VALUES not in costs:
        raise ValueError(f"no epoch reports the cost of the {VALUES}")
    stops = {term: stops_learning(by_epoch) for term, by_epoch in sorted(costs.items())}
    known = list(stops.values())
    epochs = None if any(stop is None for stop in known) else max(s for s in known if s is not None)
    values = costs[VALUES]
    kept = min(values, key=lambda epoch: (values[epoch], epoch))
    presence = stops.get(PRESENCE)
    return Budget(
        stops=stops, epochs=epochs, kept=kept, lag=None if presence is None else presence - kept
    )


def write_costs(path: Path, rows: Iterable[CostRow]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "corpus", "term", "loss", "trivial"))
        for row in rows:
            writer.writerow(
                (row.epoch, row.corpus, row.term, f"{row.loss:.6g}", f"{row.trivial:.6g}")
            )


def read_costs(path: Path) -> list[CostRow]:
    with path.open(newline="") as handle:
        return [
            CostRow(int(r["epoch"]), r["corpus"], r["term"], float(r["loss"]), float(r["trivial"]))
            for r in csv.DictReader(handle)
        ]


def render(rows: Sequence[CostRow]) -> str:
    costs = mean_costs(rows)
    terms = sorted(costs)
    epochs = sorted({row.epoch for row in rows})
    budget = budget_of(rows)
    lines = [
        [str(epoch)]
        + [f"{costs[term][epoch]:.4f}" if epoch in costs[term] else "—" for term in terms]
        for epoch in epochs
    ]
    stops = ", ".join(
        f"{term} {'not within the cap' if stop is None else f'epoch {stop}'}"
        for term, stop in budget.stops.items()
    )
    verdict = (
        "no budget: a term learns to the cap, which is too short"
        if budget.epochs is None
        else f"budget: {budget.epochs} epochs"
    )
    kept = f"kept epoch (values alone): {budget.kept}"
    if budget.lag is not None:
        kept += f", {budget.lag} before the presence stops learning"
        if budget.keeps_too_early:
            kept += f" — more than {KEPT_LAG}: the kept epoch would leave the presence unlearnt"
    return "\n\n".join(
        [
            f"**Each term's validation cost, the mean over corpora of its loss over the trivial "
            f"predictor's.**\n\n{table(['epoch', *terms], lines)}",
            f"Stops learning (no lower by over {TOLERANCE:.0%} within {WINDOW} epochs): {stops}.",
            f"{verdict}; {kept}.",
        ]
    )


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log", type=Path, help="the log of the pretraining run")
    parser.add_argument("--out", type=Path, help="directory the CSV file is written to")
    parser.add_argument(
        "--report-only", type=Path, help="render the budget of a directory already read"
    )
    arguments = parser.parse_args(argv)
    if arguments.report_only is None and arguments.log is None:
        parser.error("--log is needed to read a run; --report-only to render")
    return arguments


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - the command line
    arguments = parse_arguments(argv)
    if arguments.report_only is not None:
        print(render(read_costs(arguments.report_only / COSTS)))
        return
    rows = costs_of(arguments.log.read_text().splitlines())
    if not rows:
        raise SystemExit(f"{arguments.log} reports no finished epoch")
    out = arguments.out or WORKSPACE / arguments.log.stem
    out.mkdir(parents=True, exist_ok=True)
    write_costs(out / COSTS, rows)
    print(render(rows))


if __name__ == "__main__":
    main()
