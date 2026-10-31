from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.detection.channel_errors import ChannelErrors


@dataclass(frozen=True)
class ChannelErrorScales:
    """How much each channel is missed on the fitting side: what its errors are read over.

    A channel's scale is the median of its per-window error over the fitting windows that scored
    it, floored at a share of the scales' own median so a channel the pretext answers exactly
    does not divide the rest away; where that median is itself nought, the floor is the smallest
    scale above nought. A channel no fitting window scored is unknown rather than quiet, so it
    takes the scales' median, and a window that holds it is read neither up nor down for it.

    Invariants: every scale is positive and finite.

    Attributes:
        channels: ``[channels]`` — the channel of every scale.
        scales: ``[channels]`` — what the channel's error is divided by.
    """

    # A channel missed less than a tenth of the typical channel is read as if missed a tenth:
    # below that the division amplifies rounding rather than departures.
    FLOOR: ClassVar[float] = 0.1

    channels: NDArray[np.int64]
    scales: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.scales.shape != self.channels.shape:
            raise ValueError("one scale per channel")
        if not np.all(np.isfinite(self.scales) & (self.scales > 0.0)):
            raise ValueError("every scale is positive and finite")

    @classmethod
    def fitted_on(cls, errors: ChannelErrors) -> Self:
        """The scales over the fitting side's errors."""
        scored = errors.counted.sum(axis=0) > 0
        typical = np.full(len(errors.channels), np.nan)
        typical[scored] = np.nanmedian(errors.per_channel()[:, scored], axis=0)
        known = typical[scored]
        positive = known[known > 0.0]
        middle = float(np.median(known)) if known.size else 1.0
        floor = cls.FLOOR * middle
        if floor <= 0.0:
            floor = float(positive.min()) if positive.size else 1.0
        return cls(
            channels=errors.channels,
            scales=np.where(scored, np.maximum(typical, floor), max(middle, floor)),
        )

    def read(self, errors: ChannelErrors) -> NDArray[np.float64]:
        """``[windows]`` — the mean over the scored channels of each channel's error over its scale.

        Raises:
            ValueError: If the errors lie on other channels than the scales.
        """
        if not np.array_equal(errors.channels, self.channels):
            raise ValueError("the errors lie on other channels than the scales")
        read = np.zeros(len(errors.missed))
        scored = ~errors.unread
        read[scored] = np.nanmean(errors.per_channel()[scored] / self.scales, axis=1)
        return read
