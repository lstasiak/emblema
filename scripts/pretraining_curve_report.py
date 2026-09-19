"""Write the curve of an accepted pretraining run as CSV, and render its table from that file.

Measuring and drawing are two steps: a run of hours on an accelerator leaves a result document
in the store, and this turns the epochs of that document into one row per corpus and epoch —
every number a figure or a note needs, so that a redraw never asks for the run again.

    uv run --env-file .env.r2 scripts/pretraining_curve_report.py --result <key> <checksum>
    uv run scripts/pretraining_curve_report.py --report-only data/report/pretraining/<backbone>
"""

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.config.settings import Settings
from emblema.entrypoints.cli.configured import configured_store
from emblema.pretraining.adapters.handoff.artifact_store_handoff_exchange import (
    ArtifactStoreHandoffExchange,
)
from emblema.pretraining.domain.handoff.pretraining_result import PretrainingResult
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from scripts.reporting import table

EPOCHS = "epochs.csv"
COLUMNS = (
    "experiment",
    "tier",
    "precision",
    "epoch",
    "corpus",
    "tokens",
    "loss",
    "trivial",
    "relative",
    "training_loss",
    "hidden_ratio",
    "seconds",
    "kept",
    "backbone",
)


@dataclass(frozen=True)
class CurveRow:
    """One corpus of one epoch of a run, with the epoch's own numbers repeated beside it.

    Attributes:
        experiment: Name of the experiment the run was made under.
        tier: Compute tier the run declared.
        precision: What the run computed at.
        epoch: Which epoch, counted from zero.
        corpus: Which corpus of the mixture this row scores.
        tokens: Hidden validation tokens of that corpus the epoch was scored on.
        loss: The run's reading over those tokens.
        trivial: The channel-mean predictor's reading over the same tokens.
        relative: ``loss`` over ``trivial``.
        training_loss: The epoch's training loss over the whole mixture.
        hidden_ratio: Share of observed validation tokens the masks hid.
        seconds: Wall-clock seconds the epoch took.
        kept: Whether the epoch wrote its weights as the best so far.
        backbone: Whether the run's backbone is this epoch's weights.
    """

    experiment: str
    tier: str
    precision: str
    epoch: int
    corpus: str
    tokens: int
    loss: float
    trivial: float
    relative: float
    training_loss: float
    hidden_ratio: float
    seconds: float
    kept: bool
    backbone: bool

    @classmethod
    def of(cls, result: PretrainingResult) -> list[Self]:
        """Every corpus of every epoch of the result, in the result's order."""
        stated = result.configuration
        kept = result.outcome.backbone
        return [
            cls(
                experiment=stated.name,
                tier=str(stated.tier),
                precision=str(stated.precision),
                epoch=epoch.epoch,
                corpus=scored.corpus,
                tokens=scored.tokens,
                loss=scored.loss,
                trivial=scored.trivial,
                relative=scored.relative,
                training_loss=epoch.training_loss,
                hidden_ratio=epoch.hidden_ratio,
                seconds=epoch.seconds,
                kept=epoch.weights is not None,
                backbone=epoch.weights == kept,
            )
            for epoch in result.outcome.epochs
            for scored in epoch.validation
        ]

    @classmethod
    def parse(cls, record: dict[str, str]) -> Self:
        return cls(
            experiment=record["experiment"],
            tier=record["tier"],
            precision=record["precision"],
            epoch=int(record["epoch"]),
            corpus=record["corpus"],
            tokens=int(record["tokens"]),
            loss=float(record["loss"]),
            trivial=float(record["trivial"]),
            relative=float(record["relative"]),
            training_loss=float(record["training_loss"]),
            hidden_ratio=float(record["hidden_ratio"]),
            seconds=float(record["seconds"]),
            kept=record["kept"] == "yes",
            backbone=record["backbone"] == "yes",
        )

    def record(self) -> dict[str, str]:
        return {
            "experiment": self.experiment,
            "tier": self.tier,
            "precision": self.precision,
            "epoch": str(self.epoch),
            "corpus": self.corpus,
            "tokens": str(self.tokens),
            "loss": repr(self.loss),
            "trivial": repr(self.trivial),
            "relative": repr(self.relative),
            "training_loss": repr(self.training_loss),
            "hidden_ratio": repr(self.hidden_ratio),
            "seconds": repr(self.seconds),
            "kept": "yes" if self.kept else "no",
            "backbone": "yes" if self.backbone else "no",
        }


def write(rows: Sequence[CurveRow], directory: Path) -> Path:
    """Store the rows under ``directory``; say where."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / EPOCHS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.record())
    return path


def read(directory: Path) -> list[CurveRow]:
    """The rows stored under ``directory``.

    Raises:
        FileNotFoundError: If nothing was stored there.
    """
    with (directory / EPOCHS).open(newline="", encoding="utf-8") as handle:
        return [CurveRow.parse(record) for record in csv.DictReader(handle)]


def corpora_of(rows: Sequence[CurveRow]) -> list[str]:
    """The corpora in the order the run read them, each once."""
    return list(dict.fromkeys(row.corpus for row in rows))


def render(rows: Sequence[CurveRow]) -> str:
    """The curve as a table: one line per epoch, a column per corpus's relative validation."""
    corpora = corpora_of(rows)
    by_epoch: dict[int, list[CurveRow]] = {}
    for row in rows:
        by_epoch.setdefault(row.epoch, []).append(row)
    lines = []
    for epoch, scored in sorted(by_epoch.items()):
        relative = {row.corpus: row.relative for row in scored}
        mean = sum(relative.values()) / len(relative)
        first = scored[0]
        lines.append(
            [
                str(epoch + 1),
                f"{first.training_loss:.5f}",
                *(f"{relative[corpus]:.3f}" for corpus in corpora),
                f"{mean:.4f}",
                f"{first.seconds:.0f}",
                "backbone" if first.backbone else ("kept" if first.kept else ""),
            ]
        )
    first = rows[0]
    heading = f"{first.experiment}: tier {first.tier}, {first.precision}; validation, not test"
    return f"{heading}\n\n" + table(
        ("epoch", "training", *corpora, "mean relative", "seconds", ""), lines
    )


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", type=Path, metavar="DIR", help="render a stored curve")
    parser.add_argument("--result", nargs=2, metavar=("KEY", "CHECKSUM"), help="the result")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where the curve is stored; data/report/pretraining/<backbone> unless given",
    )
    arguments = parser.parse_args(argv)
    if (arguments.report_only is None) == (arguments.result is None):
        parser.error("give --result to read a result, or --report-only to render a stored one")
    return arguments


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if arguments.report_only is not None:
        print(render(read(arguments.report_only)))
        return
    key, checksum = arguments.result
    exchange = ArtifactStoreHandoffExchange(configured_store(Settings()))
    result = exchange.read_result(ArtifactRef(key, Checksum.parse(checksum)))
    rows = CurveRow.of(result)
    out = arguments.out or REPO_ROOT / "data" / "report" / "pretraining" / str(result.backbone)
    stored = write(rows, out)
    print(render(rows))
    print(f"\nstored under {stored.parent}")


if __name__ == "__main__":
    main()
