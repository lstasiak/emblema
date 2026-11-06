import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.adapters.detection.score_alarm import ScoreAlarm
from emblema.evaluation.domain.detection.channel_history import ChannelHistory
from emblema.evaluation.domain.detection.channel_reduction import ChannelReduction
from emblema.evaluation.domain.detection.departure_cells import DepartureCells
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError


@dataclass(frozen=True)
class ChannelLocalDetector:
    """A detector that scores a window by how far its channels depart from their own history.

    No model is fitted: the departures are each channel's against its own recent windows, and
    what the fitting side sets is the floor of their spreads and the alarm, over the fitting
    windows' own scores and read against a unit as every detector's is. A window's score is its
    largest channel's departure, or the mean over the channels holding a reading in it, nought
    where none does. The history is kept with the floor, so the kept form says how every
    departure it was set over was measured.

    Attributes:
        history: How far back a channel was read, what departed and how the channels' departures
            became the score.
        floor: ``[channels, 5]``, the least spread a channel's history is read over.
        alarm: Where a score raises an alarm.
    """

    FORMAT: ClassVar[str] = "channel-local-npz"

    history: ChannelHistory
    floor: NDArray[np.float64]
    alarm: ScoreAlarm

    @classmethod
    def fitted_on(
        cls,
        departures: NDArray[np.float64],
        present: NDArray[np.bool_],
        *,
        history: ChannelHistory,
        floor: NDArray[np.float64],
        contamination: float,
        units: Sequence[str] | None = None,
    ) -> Self:
        """The detector over the fitting windows' ``departures``, each of ``units`` if given.

        Raises:
            ValueError: If ``units`` does not name one unit per window.
            InvalidScoredOutcomeError: If units are given and none holds two windows.
        """
        return cls(
            history=history,
            floor=floor,
            alarm=ScoreAlarm.set_over(
                cls.reduced(departures, present, history.channels),
                contamination=contamination,
                units=units,
            ),
        )

    def scores(
        self,
        departures: NDArray[np.float64],
        present: NDArray[np.bool_],
        *,
        units: Sequence[str] | None = None,
    ) -> NDArray[np.float64]:
        """The score above the alarm of every window, in the order given.

        Raises:
            ValueError: If the windows come with units and the detector reads none, or the other
                way round, or ``units`` does not name one unit per window.
            InvalidScoredOutcomeError: If units are given and none holds two windows.
        """
        self.alarm.require(units)
        return self.alarm.above(
            self.reduced(departures, present, self.history.channels), units=units
        )

    @staticmethod
    def reduced(
        departures: NDArray[np.float64], present: NDArray[np.bool_], channels: ChannelReduction
    ) -> NDArray[np.float64]:
        """``[windows]``: each window's largest departure over its channels, or their mean."""
        if channels is ChannelReduction.MAX:
            largest: NDArray[np.float64] = departures.max(axis=1)
            return largest
        count = present.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean: NDArray[np.float64] = np.where(
                count > 0, np.where(present, departures, 0.0).sum(axis=1) / count, 0.0
            )
        return mean

    def kept(self) -> RepresentationBytes:
        """The detector as it is kept, under the format ``from_bytes`` reads back."""
        return RepresentationBytes(self.FORMAT, self.to_bytes())

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        np.savez(
            buffer,
            span=np.asarray(self.history.span),
            departure=np.asarray(str(self.history.departure)),
            channels=np.asarray(str(self.history.channels)),
            floor=self.floor,
            alarm=np.asarray(self.alarm.level),
            per_unit=np.asarray(self.alarm.per_unit),
            unit_spread=np.asarray(self.alarm.unit_spread),
        )
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, content: bytes) -> Self:
        """The detector kept as ``content``.

        Raises:
            UnreadableFittedCandidateError: If the bytes are not an archive this reads.
        """
        try:
            with np.load(io.BytesIO(content), allow_pickle=False) as archive:
                return cls(
                    history=ChannelHistory(
                        span=int(archive["span"]),
                        departure=DepartureCells(str(archive["departure"])),
                        channels=ChannelReduction(str(archive["channels"])),
                    ),
                    floor=np.asarray(archive["floor"], dtype=np.float64),
                    alarm=ScoreAlarm(
                        level=float(archive["alarm"]),
                        per_unit=bool(archive["per_unit"]),
                        unit_spread=float(archive["unit_spread"]),
                    ),
                )
        except (KeyError, ValueError, OSError, EOFError) as error:
            raise UnreadableFittedCandidateError(
                "the bytes are not a channel-local detector's archive"
            ) from error
