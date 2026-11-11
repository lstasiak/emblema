from dataclasses import dataclass

from emblema.catalog.domain.cleaning.cleaning_policy import CleaningPolicy
from emblema.catalog.domain.exceptions import InvalidCleaningRecordError


@dataclass(frozen=True, kw_only=True)
class ChannelCleaning:
    """How many of one channel's values a cleaning policy removed and clipped, over every unit.

    Attributes:
        channel: Name of the channel within its corpus.
        removed: Values removed.
        clipped: Values clipped to a bound and kept.
    """

    channel: str
    removed: int
    clipped: int

    def __post_init__(self) -> None:
        if self.removed < 0 or self.clipped < 0:
            raise InvalidCleaningRecordError(
                f"counts of {self.channel!r} must not be negative: {self.removed}, {self.clipped}"
            )


@dataclass(frozen=True, kw_only=True)
class CleaningRecord:
    """A cleaning policy as a publication applied it: the bounds and what each one changed.

    Invariants: one count per channel the policy names, in the policy's order.

    Attributes:
        policy: The bounds the publication held the corpus's values to.
        counts: Per channel named, the values removed and clipped across every unit.
    """

    policy: CleaningPolicy
    counts: tuple[ChannelCleaning, ...]

    def __post_init__(self) -> None:
        named = tuple(each.channel for each in self.policy.bounds)
        counted = tuple(each.channel for each in self.counts)
        if counted != named:
            raise InvalidCleaningRecordError(
                f"counts must name the policy's channels in its order: {counted} for {named}"
            )
