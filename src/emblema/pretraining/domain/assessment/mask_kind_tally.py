from dataclasses import dataclass
from typing import Self

from emblema.pretraining.domain.exceptions import IncompatibleTalliesError
from emblema.pretraining.domain.mask_kind import MaskKind


@dataclass(frozen=True)
class MaskKindTally:
    """Squared errors summed over the hidden tokens of one kind of mask in one group of windows.

    Sums rather than means, so that groups combine by addition: whoever judges the model against
    its baselines resamples groups — the units windows were cut from, whose windows overlap and
    share one realisation of the process — and needs nothing but these to do it. The judgement
    itself is not made here; a comparison of two means over tokens that are not independent is not
    a verdict.

    Attributes:
        kind: The kind of mask the tokens were hidden by.
        apart: Whether the tokens belong to channels reported apart from the verdict — constant
            ones, whose value every method recovers, and timeless ones, which no interpolation
            reaches.
        group: The group of windows the sums cover, as the caller named it.
        tokens: Hidden tokens of the kind in the group.
        model: Summed squared error of the model.
        matched: Of the baseline the plan matches to the kind — interpolation within the channel
            for a block or a single token, the cross-channel regression for a channel hidden whole.
        linear: Of the strongest linear baseline on the same inputs — the regression on the other
            channels and the channel's own line together, or the cross-channel regression alone
            where nothing of the channel is left.
        mean: Of predicting the channel's mean, zero once normalised: the summed squared targets.
        floor: Summed variance of the measurement noise in the targets, which no predictor
            recovers; ``None`` where the corpus states no noise.
    """

    kind: MaskKind
    apart: bool
    group: str
    tokens: int
    model: float
    matched: float
    linear: float
    mean: float
    floor: float | None

    def __add__(self, other: Self) -> Self:
        """The sums of both tallies, which must cover the same kind, side and group.

        Raises:
            IncompatibleTalliesError: If the tallies cover different tokens, or one states a noise
                floor and the other does not — a sum of a floor and no floor is neither.
        """
        if (self.kind, self.apart, self.group) != (other.kind, other.apart, other.group):
            raise IncompatibleTalliesError(
                f"tallies of {self.kind.value}/{self.apart}/{self.group} and "
                f"{other.kind.value}/{other.apart}/{other.group} do not add up"
            )
        if (self.floor is None) != (other.floor is None):
            raise IncompatibleTalliesError(
                "a tally with a noise floor does not add up with one without"
            )
        return type(self)(
            kind=self.kind,
            apart=self.apart,
            group=self.group,
            tokens=self.tokens + other.tokens,
            model=self.model + other.model,
            matched=self.matched + other.matched,
            linear=self.linear + other.linear,
            mean=self.mean + other.mean,
            floor=None if self.floor is None or other.floor is None else self.floor + other.floor,
        )
