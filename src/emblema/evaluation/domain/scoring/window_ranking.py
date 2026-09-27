from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Self

from emblema.evaluation.domain.exceptions import InvalidWindowRankingError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.outcome_scheme import OutcomeScheme
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction


@dataclass(frozen=True, kw_only=True)
class WindowRanking:
    """One run's answers put in order, to read how well they rank the two outcomes apart.

    The area under the ROC curve is the share of (positive, negative) pairs the answers order
    correctly, a tie counting half, and it does not split into a sum per unit the way a squared
    error does. What does split is the order: sorted once, the windows are swept from the
    lowest answer up, each group of equal answers adding its positives times the negatives
    below it and half the negatives beside it. A resample of units only changes how many times
    each unit counts, so the sweep takes a weight per unit and the sorting is never repeated.

    Units are held in their own order, the order a paired comparison names them in, so two
    rankings of the same units weigh the same unit by the same index.

    Invariants: at least one window; every target is an outcome, zero or one; the windows hold
    both outcomes, since an area needs a positive and a negative to compare.

    Attributes:
        units: Every unit the windows belong to, in unit order.
        sequence: Index into ``units`` of each window, windows in ascending order of answer.
        positive: Whether each window of ``sequence`` is a positive.
        closes_tie: Whether each window of ``sequence`` is the last of its group of equal
            answers.
        outcomes: Each window's outcome in window order (unit, then position), so two rankings
            can be checked to answer the same question.
    """

    units: tuple[UnitKey, ...]
    sequence: tuple[int, ...]
    positive: tuple[bool, ...]
    closes_tie: tuple[bool, ...]
    outcomes: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.sequence:
            raise InvalidWindowRankingError("a ranking needs a window")
        if not (
            len(self.sequence) == len(self.positive) == len(self.closes_tie) == len(self.outcomes)
        ):
            raise InvalidWindowRankingError("every window needs an outcome and a place among ties")
        if all(self.positive) or not any(self.positive):
            raise InvalidWindowRankingError("a ranking needs both outcomes to tell apart")

    @classmethod
    def of(cls, predictions: Iterable[WindowPrediction]) -> Self:
        """The ranking of ``predictions`` by their answers.

        Raises:
            InvalidWindowRankingError: If there is no prediction, a target is not an outcome,
                or the predictions hold one outcome only.
        """
        rows = sorted(predictions, key=lambda p: (str(p.window.unit), p.window.position))
        for row in rows:
            if row.target not in OutcomeScheme.OUTCOMES:
                raise InvalidWindowRankingError(
                    f"a window of {row.window.unit} has target {row.target}, not an outcome"
                )
        units = tuple(sorted({row.window.unit for row in rows}, key=str))
        index = {unit: place for place, unit in enumerate(units)}
        ranked = sorted(rows, key=lambda row: row.predicted)
        return cls(
            units=units,
            sequence=tuple(index[row.window.unit] for row in ranked),
            positive=tuple(row.target == 1.0 for row in ranked),
            closes_tie=tuple(
                place == len(ranked) - 1 or ranked[place + 1].predicted != row.predicted
                for place, row in enumerate(ranked)
            ),
            outcomes=tuple(row.target == 1.0 for row in rows),
        )

    @property
    def auroc(self) -> float:
        """The area under the ROC curve over every window, each unit counted once."""
        return self.auroc_weighted([1] * len(self.units))

    def auroc_weighted(self, weights: Sequence[int]) -> float:
        """The area with the windows of unit ``i`` counted ``weights[i]`` times over.

        Raises:
            InvalidWindowRankingError: If the weights leave one outcome only.
        """
        below = 0
        area = 0.0
        positives = negatives = 0
        tied_positive = tied_negative = 0
        for unit, positive, closes in zip(
            self.sequence, self.positive, self.closes_tie, strict=True
        ):
            weight = weights[unit]
            if positive:
                tied_positive += weight
            else:
                tied_negative += weight
            if closes:
                area += tied_positive * (below + tied_negative / 2.0)
                below += tied_negative
                positives += tied_positive
                negatives += tied_negative
                tied_positive = tied_negative = 0
        if positives == 0 or negatives == 0:
            raise InvalidWindowRankingError("the weights leave one outcome only")
        return area / (positives * negatives)
