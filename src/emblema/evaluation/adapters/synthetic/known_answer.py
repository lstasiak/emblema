import random
from collections.abc import Sequence
from dataclasses import dataclass

from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.evaluation.domain.statistics.paired_unit_errors import PairedUnitErrors


@dataclass(frozen=True, kw_only=True)
class KnownAnswer:
    """Paired errors with a known answer, for checking the procedure rather than a dataset.

    Draws what a cell of a grid leaves behind — a squared error and a window count per unit, for
    a control and a candidate, over one or more repeats — from levels stated up front, so the true
    reduction is known before the procedure reads it. Two sources of scatter are kept apart
    because the two-level bootstrap exists to tell them apart: a unit's difficulty, shared by
    both sides and every repeat, and a repeat's luck, drawn anew per side and per repeat.

    Attributes:
        control_rmse: The control's error, before any scatter.
        candidate_rmse: The candidate's error, before any scatter; the true reduction is the
            difference of the two.
        units: How many units each side is scored on.
        windows: How many windows each unit holds.
        unit_scatter: How far a unit's difficulty may multiply both sides' errors, as a share
            either way; shared by both sides and by every repeat.
        repeat_scatter: How far one repeat's luck may multiply one side's error, as a share
            either way; drawn anew per side and per repeat.
    """

    control_rmse: float
    candidate_rmse: float
    units: int = 18
    windows: int = 30
    unit_scatter: float = 0.3
    repeat_scatter: float = 0.1

    @property
    def true_reduction(self) -> float:
        return self.control_rmse - self.candidate_rmse

    def paired(self, *, seed: int) -> PairedUnitErrors:
        """One repeat's pair over the units, under ``seed``."""
        return self.repeats(1, seed=seed)[0]

    def repeats(self, count: int, *, seed: int) -> tuple[PairedUnitErrors, ...]:
        """``count`` repeats' pairs over the same units, each with its own luck."""
        draws = random.Random(seed)
        difficulty = [
            draws.uniform(1.0 - self.unit_scatter, 1.0 + self.unit_scatter)
            for _ in range(self.units)
        ]
        return tuple(
            PairedUnitErrors(
                control=self._side(self.control_rmse, difficulty, draws),
                candidate=self._side(self.candidate_rmse, difficulty, draws),
            )
            for _ in range(count)
        )

    def _side(
        self, level: float, difficulty: Sequence[float], draws: random.Random
    ) -> tuple[UnitError, ...]:
        return tuple(
            UnitError(
                unit=UnitKey(f"unit/{index}"),
                squared_error=self.windows
                * (
                    level
                    * scale
                    * draws.uniform(1.0 - self.repeat_scatter, 1.0 + self.repeat_scatter)
                )
                ** 2,
                windows=self.windows,
            )
            for index, scale in enumerate(difficulty)
        )
