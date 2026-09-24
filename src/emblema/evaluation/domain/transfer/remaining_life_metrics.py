from collections.abc import Iterable
from dataclasses import dataclass
from math import exp, inf, isfinite, isnan, sqrt
from typing import Self

from emblema.evaluation.domain.exceptions import InvalidRemainingLifeMetricsError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction


@dataclass(frozen=True, kw_only=True)
class RemainingLifeMetrics:
    """What a candidate's answers on a remaining-life task are worth, read four more ways.

    The error over every window decides the question; the readings here are reported beside it
    and carry no threshold. Each is arithmetic over the same answers: the error where the label
    is below its ceiling, the regime in which the task is a task; the error on the last window of
    each unit, which is the benchmark's protocol on a test side cut short of failure and the
    error at the end of life on a validation side that runs to it; the share of answers within a
    fifth of the true remaining life, which is closer to the decision a schedule makes; and the
    asymmetric score of the benchmark, which grows exponentially with lateness and is dominated
    by a few units. The score is a mean per window rather than the benchmark's sum, so that it
    does not scale with how many windows a side holds and reads across sides and tasks.

    Invariants: at least one window; every error is finite and not negative, the share lies in
    ``[0, 1]``, the score is not negative — and infinite where an answer is so late that its
    penalty does not fit a float, since a reading that is only read must not stop a report.

    Attributes:
        windows: How many answers the readings run over.
        rmse: Root mean squared error over every window.
        rmse_below_ceiling: The same over windows whose label lies below the ceiling; ``None``
            where no window does.
        last_window_rmse: The same over the last window of each unit.
        alpha_lambda_accuracy: Share of windows whose answer lies within ±20 % of the label.
        asymmetric_score: The benchmark's score, a mean over the windows; lower is better.
    """

    windows: int
    rmse: float
    rmse_below_ceiling: float | None
    last_window_rmse: float
    alpha_lambda_accuracy: float
    asymmetric_score: float

    def __post_init__(self) -> None:
        if self.windows < 1:
            raise InvalidRemainingLifeMetricsError(
                f"the readings need at least one window, got {self.windows}"
            )
        for label, value in (
            ("rmse", self.rmse),
            ("rmse_below_ceiling", self.rmse_below_ceiling),
            ("last_window_rmse", self.last_window_rmse),
        ):
            if value is not None and (not isfinite(value) or value < 0.0):
                raise InvalidRemainingLifeMetricsError(
                    f"{label} must be finite and not negative, got {value}"
                )
        if not 0.0 <= self.alpha_lambda_accuracy <= 1.0:
            raise InvalidRemainingLifeMetricsError(
                f"alpha_lambda_accuracy must lie in [0, 1], got {self.alpha_lambda_accuracy}"
            )
        if isnan(self.asymmetric_score) or self.asymmetric_score < 0.0:
            raise InvalidRemainingLifeMetricsError(
                f"asymmetric_score must be a non-negative number, got {self.asymmetric_score}"
            )

    @classmethod
    def of(cls, predictions: Iterable[WindowPrediction], *, ceiling: float) -> Self:
        """The readings over ``predictions``, labelled under ``ceiling``.

        Raises:
            InvalidRemainingLifeMetricsError: If there is no prediction or the ceiling is not a
                positive finite number.
        """
        if not isfinite(ceiling) or ceiling <= 0.0:
            raise InvalidRemainingLifeMetricsError(
                f"the ceiling must be positive and finite, got {ceiling}"
            )
        rows = tuple(predictions)
        if not rows:
            raise InvalidRemainingLifeMetricsError("the readings need at least one prediction")
        below = [row for row in rows if row.target < ceiling]
        last: dict[UnitKey, WindowPrediction] = {}
        for row in rows:
            kept = last.get(row.window.unit)
            if kept is None or row.window.ends_at > kept.window.ends_at:
                last[row.window.unit] = row
        return cls(
            windows=len(rows),
            rmse=_rmse(rows),
            rmse_below_ceiling=_rmse(below) if below else None,
            last_window_rmse=_rmse(last.values()),
            alpha_lambda_accuracy=sum(1 for row in rows if _within_a_fifth(row)) / len(rows),
            asymmetric_score=sum(_score(row.predicted - row.target) for row in rows) / len(rows),
        )


def _rmse(rows: Iterable[WindowPrediction]) -> float:
    rows = tuple(rows)
    return sqrt(sum(row.squared_error for row in rows) / len(rows))


def _within_a_fifth(row: WindowPrediction) -> bool:
    return abs(row.predicted - row.target) <= 0.2 * row.target


def _score(late_by: float) -> float:
    """The benchmark's penalty for one answer: an early one decays over 13, a late one over 10."""
    try:
        return exp(-late_by / 13.0) - 1.0 if late_by < 0.0 else exp(late_by / 10.0) - 1.0
    except OverflowError:
        return inf
