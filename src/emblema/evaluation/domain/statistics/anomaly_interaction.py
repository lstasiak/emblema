from collections.abc import Sequence
from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import InvalidPairedAnomalyRankingsError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.statistics.paired_anomaly_rankings import PairedAnomalyRankings


@dataclass(frozen=True, kw_only=True)
class AnomalyInteraction:
    """Two paired comparisons of the same units, read as the difference of their reductions.

    What a grid of two encoders by two teachings asks: whether the gain of one detector over
    another grows when both are taught the same events. The reduction is the taught pair's less
    the untaught pair's, and a resample of the units moves all four sides together, so the
    interval carries the sampling of months and events once, not twice. The control's shortfall
    has no share to read the difference as a share of.

    Invariants: both pairs name the same units and the same windows with the same events.

    Attributes:
        taught: The candidate against the control, both taught the events.
        untaught: The same candidate against the same control, taught none.
    """

    taught: PairedAnomalyRankings
    untaught: PairedAnomalyRankings

    def __post_init__(self) -> None:
        if self.taught.units != self.untaught.units:
            raise InvalidPairedAnomalyRankingsError("the two pairs disagree about the units")
        if self.taught.control[0].outcomes != self.untaught.control[0].outcomes:
            raise InvalidPairedAnomalyRankingsError(
                "the two pairs disagree about the windows or their events"
            )

    @property
    def units(self) -> tuple[UnitKey, ...]:
        return self.taught.units

    @property
    def strata(self) -> tuple[tuple[int, ...], ...]:
        """Indices of the units holding an event's window, then of those holding none."""
        return self.taught.strata

    @property
    def reduction(self) -> float:
        """How much more area the candidate gains over the control taught than untaught."""
        return self.taught.reduction - self.untaught.reduction

    @property
    def relative_reduction(self) -> None:
        """A difference of two reductions has no control's shortfall to be a share of."""
        return None

    def reduction_over(self, picks: Sequence[int]) -> float:
        """The interaction over the units at ``picks``, a unit counted as often as it is picked."""
        return self.taught.reduction_over(picks) - self.untaught.reduction_over(picks)
