"""Compare one transfer mode under two backbones, paired on the same engines and windows.

A backbone replaces another only when the arm adapted from it beats the same arm adapted from the
other on the same validation engines and windows: the reduction of the error, pooled over the
seeds both sides hold, with the whole 95 per cent interval of a bootstrap over the engines above
zero (`docs/preregistration.md`, 2026-09-22). Comparing each side's reduction against its own
control instead would choose whichever run was luckier. The arithmetic is the Evaluation
context's; what is here reads the stored cells and refuses a pair it cannot read as one.

    uv run scripts/backbone_comparison_report.py --old DIR [DIR ...] --new DIR [DIR ...]
    uv run scripts/backbone_comparison_report.py --mode lora --budget 50 --old ... --new ...

Every number is validation, not test.
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.domain.statistics.paired_difference import PairedDifference
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.statistics.paired_unit_errors import PairedUnitErrors
from emblema.evaluation.domain.transfer.unit_error import UnitError
from emblema.evaluation.domain.transfer.window_prediction import WindowPrediction
from scripts.label_curve_report import RESAMPLES, predictions_of
from scripts.transfer_grid import Stored


@dataclass(frozen=True, kw_only=True)
class Side:
    """One mode at one budget under one backbone: its predictions, seed by seed.

    Attributes:
        backbone: The stored reference of the weights the mode was adapted from.
        cells: Each seed's predictions over the validation windows.
    """

    backbone: str
    cells: Mapping[int, tuple[WindowPrediction, ...]]


@dataclass(frozen=True, kw_only=True)
class BackboneComparison:
    """What pairing the two sides found, and whether the new backbone replaces the old.

    Attributes:
        mode: The transfer mode compared.
        budget: The budget of labels, as the grid names it.
        seeds: The seeds both sides hold, pooled per engine.
        engines: How many validation engines the pairs are read over.
        windows: How many validation windows a seed scores.
        old: The side under the backbone in place.
        new: The side under the backbone that would replace it.
        rmse_old: The old side's error, pooled over the seeds.
        rmse_new: The new side's error, pooled over the seeds.
        difference: How much lower the new side's error is, with its interval.
    """

    mode: str
    budget: str
    seeds: tuple[int, ...]
    engines: int
    windows: int
    old: Side
    new: Side
    rmse_old: float
    rmse_new: float
    difference: PairedDifference

    @property
    def replaces(self) -> bool:
        return self.difference.interval.low > 0.0

    def render(self) -> str:
        interval = self.difference.interval
        return "\n".join(
            (
                f"{self.mode} at {self.budget}, seeds {list(self.seeds)}, {self.engines} engines, "
                f"{self.windows} windows a seed",
                f"  old {_short(self.old.backbone)}: RMSE {self.rmse_old:.3f}",
                f"  new {_short(self.new.backbone)}: RMSE {self.rmse_new:.3f}",
                f"  reduction {self.difference.reduction:+.3f} "
                f"({self.difference.relative_reduction:+.1%}), 95 % interval "
                f"[{interval.low:+.3f}, {interval.high:+.3f}], p {self.difference.p_value:.4f}",
                "  the new backbone replaces the old"
                if self.replaces
                else "  the old backbone stays",
            )
        )


def side_of(directories: Sequence[Path], mode: str, budget: str) -> Side:
    """The stored cells of ``mode`` at ``budget`` in the directories, which must share a backbone.

    Raises:
        SystemExit: If the directories hold no such cell, or hold it under several backbones.
    """
    shards = [Stored.existing(directory) for directory in directories]
    backbones = {
        run["backbone"]
        for shard in shards
        for run in shard.runs()
        if run["mode"] == mode and run["budget"] == budget
    }
    if not backbones:
        raise SystemExit(f"{directories[0]} and the rest hold no cell of {mode} at {budget}")
    if len(backbones) > 1:
        raise SystemExit(f"{directories[0]} and the rest hold {mode} under several backbones")
    cells = {
        seed: rows
        for (held_mode, held_budget, seed), rows in predictions_of(shards).items()
        if held_mode == mode and held_budget == budget
    }
    return Side(backbone=backbones.pop(), cells=cells)


def compare(
    old: Side, new: Side, *, mode: str, budget: str, resamples: int = RESAMPLES
) -> BackboneComparison:
    """Pair the two sides engine by engine over the seeds both hold.

    Raises:
        SystemExit: If both sides adapt the same backbone, share no seed, or were scored on
            different windows or targets under a seed.
    """
    if old.backbone == new.backbone:
        raise SystemExit("both sides adapt the same backbone")
    seeds = tuple(sorted(set(old.cells) & set(new.cells)))
    if not seeds:
        raise SystemExit("no seed is held by both sides")
    for seed in seeds:
        if _scored(old.cells[seed]) != _scored(new.cells[seed]):
            raise SystemExit(
                f"seed {seed}: the two sides were scored on different windows or targets"
            )
    paired = PairedUnitErrors.pooled(
        control=[UnitError.per_unit(old.cells[seed]) for seed in seeds],
        candidate=[UnitError.per_unit(new.cells[seed]) for seed in seeds],
    )
    return BackboneComparison(
        mode=mode,
        budget=budget,
        seeds=seeds,
        engines=len(paired.units),
        windows=len(old.cells[seeds[0]]),
        old=old,
        new=new,
        rmse_old=paired.rmse_control,
        rmse_new=paired.rmse_candidate,
        difference=PairedUnitBootstrap(resamples=resamples, seed=1).compare(paired),
    )


def _scored(predictions: Sequence[WindowPrediction]) -> list[tuple[str, int, float]]:
    return sorted((str(p.window.unit), p.window.position, p.target) for p in predictions)


def _short(backbone: str) -> str:
    return f"{backbone.rsplit('/', 1)[-1][:12]}…"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="full_fine_tuning", help="the transfer mode compared")
    parser.add_argument(
        "--budget", default="200", help="the budget of labels, as the grid names it"
    )
    parser.add_argument(
        "--old", nargs="+", type=Path, required=True, help="cells under the backbone in place"
    )
    parser.add_argument(
        "--new", nargs="+", type=Path, required=True, help="cells under the candidate backbone"
    )
    arguments = parser.parse_args(argv)
    comparison = compare(
        side_of(arguments.old, arguments.mode, arguments.budget),
        side_of(arguments.new, arguments.mode, arguments.budget),
        mode=arguments.mode,
        budget=arguments.budget,
    )
    print(comparison.render())


if __name__ == "__main__":
    main()
