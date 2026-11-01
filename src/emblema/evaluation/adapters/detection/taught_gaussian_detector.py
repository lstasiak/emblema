import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.adapters.detection.robust_scale import RobustScale
from emblema.evaluation.adapters.detection.score_alarm import ScoreAlarm
from emblema.evaluation.adapters.detection.shrunk_covariance import ShrunkCovariance
from emblema.evaluation.adapters.detection.standardisation import Standardisation
from emblema.evaluation.adapters.detection.taught_rows import TaughtRows
from emblema.evaluation.domain.detection.taught_head import TaughtHead
from emblema.evaluation.domain.exceptions import (
    InvalidDetectorRecipeError,
    InvalidTaughtEventsError,
    UnreadableFittedCandidateError,
)


@dataclass(frozen=True)
class TaughtGaussianDetector:
    """A detector taught a few events, scoring a row by where it lies along their direction.

    Fisher's direction under a shared covariance: the fitting rows outside the taught events
    give the mean and the shrunk covariance, as the whitened detector's do, and the taught rows
    give a second mean, each event weighing one over its windows. A row's score is its whitened
    coordinate along the line from the one mean to the other, signed, so a row on the events'
    side scores high and one on the far side low; in whitened coordinates the fitting rows
    spread by about one along it. A head that combines reads that coordinate and the squared
    whitened distance from the fitting mean each over its own fitting median and spread, and
    averages them, so an event unlike the taught ones still raises the distance. The alarm is
    set over every fitting row's own score, the taught rows included, and read against a unit
    as the other detectors' are.

    Invariants: the head is the direction or the combination; the two scales are kept exactly
    for the combination.

    Attributes:
        standardisation: How every row is centred and scaled, from the rows outside the events.
        root: The inverse square root of those rows' shrunk covariance.
        axis: ``[features]``: a standardised row's product with it is the whitened coordinate
            along the events' direction.
        head: Whether the coordinate alone is the score, or its combination with the distance.
        scales: The fitting median and spread of the distance and of the coordinate, in that
            order, for the combination; ``None`` for the direction alone.
        alarm: Where a score raises an alarm.
    """

    FORMAT: ClassVar[str] = "taught-gaussian-npz"
    HEADS: ClassVar[frozenset[TaughtHead]] = frozenset({TaughtHead.DIRECTION, TaughtHead.COMBINED})

    standardisation: Standardisation
    root: NDArray[np.float64]
    axis: NDArray[np.float64]
    head: TaughtHead
    scales: tuple[RobustScale, RobustScale] | None
    alarm: ScoreAlarm

    def __post_init__(self) -> None:
        if self.head not in self.HEADS:
            raise InvalidDetectorRecipeError(
                f"a Gaussian taught events scores by one of {sorted(self.HEADS)}, not {self.head}"
            )
        if (self.scales is None) != (self.head is TaughtHead.DIRECTION):
            raise InvalidDetectorRecipeError(
                "the combination reads two scales, the direction alone none"
            )

    @classmethod
    def fitted_on(
        cls,
        rows: NDArray[np.float64],
        taught: TaughtRows,
        *,
        head: TaughtHead,
        contamination: float,
        units: Sequence[str] | None = None,
    ) -> Self:
        """The detector fitted on ``rows``, one per fitting window, taught the rows ``taught`` name.

        Raises:
            InvalidDetectorRecipeError: If the head is not one a Gaussian scores by.
            InvalidTaughtEventsError: If a taught row lies past the rows, or the taught events'
                mean is the fitting mean, which gives no direction.
            ValueError: If fewer than two rows lie outside the taught events, or ``units`` does
                not name one unit per row.
            InvalidScoredOutcomeError: If units are given and none holds two rows.
        """
        if head not in cls.HEADS:
            raise InvalidDetectorRecipeError(
                f"a Gaussian taught events scores by one of {sorted(cls.HEADS)}, not {head}"
            )
        rows = np.asarray(rows, dtype=np.float64)
        outside = ~taught.mask(len(rows))
        if outside.sum() < 2:
            raise ValueError("a detector taught events needs two fitting rows outside them")
        standardisation = Standardisation.fitted_on(rows[outside])
        standardised = standardisation.of(rows)
        root = ShrunkCovariance.of(standardised[outside]).inverse_root()
        weights = taught.weights()
        towards = root @ ((weights / weights.sum()) @ standardised[list(taught.positions)])
        length = float(np.linalg.norm(towards))
        if length <= np.finfo(np.float64).tiny:
            raise InvalidTaughtEventsError(
                "the taught events lie at the fitting mean; there is no direction to score along"
            )
        axis = root @ (towards / length)
        distance, along = cls._statistics(standardised, root, axis)
        scales = (
            None
            if head is TaughtHead.DIRECTION
            else (RobustScale.of(distance), RobustScale.of(along))
        )
        own = cls._combined(scales, distance, along)
        return cls(
            standardisation=standardisation,
            root=root,
            axis=axis,
            head=head,
            scales=scales,
            alarm=ScoreAlarm.set_over(own, contamination=contamination, units=units),
        )

    def scores(
        self, rows: NDArray[np.float64], *, units: Sequence[str] | None = None
    ) -> NDArray[np.float64]:
        """The score above the alarm of every row, in the order given.

        Raises:
            ValueError: If the rows come with units and the detector reads none, or the other
                way round, or ``units`` does not name one unit per row.
            InvalidScoredOutcomeError: If units are given and none holds two rows.
        """
        self.alarm.require(units)
        distance, along = self._statistics(self.standardisation.of(rows), self.root, self.axis)
        return self.alarm.above(self._combined(self.scales, distance, along), units=units)

    @staticmethod
    def _statistics(
        standardised: NDArray[np.float64], root: NDArray[np.float64], axis: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The squared whitened distance from the fitting mean, and the coordinate on the axis."""
        whitened = standardised @ root
        distance: NDArray[np.float64] = np.einsum("ij,ij->i", whitened, whitened)
        along: NDArray[np.float64] = standardised @ axis
        return distance, along

    @staticmethod
    def _combined(
        scales: tuple[RobustScale, RobustScale] | None,
        distance: NDArray[np.float64],
        along: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if scales is None:
            return along
        combined: NDArray[np.float64] = (scales[0].read(distance) + scales[1].read(along)) / 2.0
        return combined

    def kept(self) -> RepresentationBytes:
        """The detector as it is kept, under the format ``from_bytes`` reads back."""
        return RepresentationBytes(self.FORMAT, self.to_bytes())

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        np.savez(
            buffer,
            centre=self.standardisation.centre,
            scale=self.standardisation.scale,
            root=self.root,
            axis=self.axis,
            head=np.asarray(str(self.head)),
            scales=np.asarray(
                []
                if self.scales is None
                else [s for scale in self.scales for s in (scale.centre, scale.spread)]
            ),
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
                kept = [float(s) for s in archive["scales"]]
                return cls(
                    standardisation=Standardisation(
                        centre=np.asarray(archive["centre"], dtype=np.float64),
                        scale=np.asarray(archive["scale"], dtype=np.float64),
                    ),
                    root=np.asarray(archive["root"], dtype=np.float64),
                    axis=np.asarray(archive["axis"], dtype=np.float64),
                    head=TaughtHead(str(archive["head"])),
                    scales=(
                        None
                        if not kept
                        else (RobustScale(kept[0], kept[1]), RobustScale(kept[2], kept[3]))
                    ),
                    alarm=ScoreAlarm(
                        level=float(archive["alarm"]),
                        per_unit=bool(archive["per_unit"]),
                        unit_spread=float(archive["unit_spread"]),
                    ),
                )
        except (KeyError, ValueError, OSError, EOFError, IndexError) as error:
            raise UnreadableFittedCandidateError(
                "the bytes are not a taught Gaussian detector's archive"
            ) from error
