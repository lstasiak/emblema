from dataclasses import dataclass
from typing import ClassVar

from emblema.evaluation.domain.detection.channel_reduction import ChannelReduction
from emblema.evaluation.domain.detection.departure_cells import DepartureCells
from emblema.evaluation.domain.exceptions import InvalidDetectorRecipeError


@dataclass(frozen=True, kw_only=True)
class ChannelHistory:
    """How far back a detector reads each channel's own history, and how it scores a window by it.

    Each channel of a window is read against the same channel over the windows that end in the
    ``span - 1`` window lengths before it: how far its values lie from their median there, over
    their spread there. A channel with fewer than half those windows holding a reading has no
    history to depart from and departs by nothing.

    Invariants: the span holds the window and at least one before it.

    Attributes:
        span: How many window lengths the history and the window cover together.
        departure: Whether the window's summary or each of its spans departs.
        channels: How the channels' departures become the window's score.
    """

    STANDARD_SPAN: ClassVar[int] = 6

    span: int = STANDARD_SPAN
    departure: DepartureCells = DepartureCells.WINDOW
    channels: ChannelReduction = ChannelReduction.MAX

    def __post_init__(self) -> None:
        if self.span < 2:
            raise InvalidDetectorRecipeError(
                f"a history covers the window and at least one before it, got a span of {self.span}"
            )

    @property
    def windows_before(self) -> int:
        """How many window lengths before the window the history reaches."""
        return self.span - 1

    @property
    def least_history(self) -> int:
        """How many earlier windows with a reading a channel needs to depart from its history."""
        return -(-self.windows_before // 2)
