import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.adapters.detection.score_alarm import ScoreAlarm
from emblema.evaluation.adapters.detection.standardisation import Standardisation
from emblema.evaluation.domain.detection.score_distance import ScoreDistance
from emblema.evaluation.domain.exceptions import (
    InvalidDetectorRecipeError,
    UnreadableFittedCandidateError,
)


@dataclass(frozen=True)
class MeanDistanceDetector:
    """A detector fitted on rows without a label, scoring a row by how far it lies from their mean.

    The simplest bars a representation is held to: over features standardised as every detector
    standardises them, the squared Euclidean distance from the fitting mean, which weighs every
    feature's departure alike, or the farthest single feature's departure, which flags a window
    for one reading far out whatever the rest do, as a threshold on each channel's deviations
    would. The alarm is set over the fitting rows' own scores and read against a unit as the
    other detectors' are.

    Invariants: the distance is one of the two from the mean.

    Attributes:
        standardisation: How every row is centred and scaled before its distance is read.
        distance: Which distance from the mean: ``CENTRE`` or ``FARTHEST``.
        alarm: Where a distance raises an alarm.
    """

    FORMAT: ClassVar[str] = "mean-distance-npz"
    DISTANCES: ClassVar[frozenset[ScoreDistance]] = frozenset(
        {ScoreDistance.CENTRE, ScoreDistance.FARTHEST}
    )

    standardisation: Standardisation
    distance: ScoreDistance
    alarm: ScoreAlarm

    def __post_init__(self) -> None:
        self._require_from_mean(self.distance)

    @classmethod
    def fitted_on(
        cls,
        rows: NDArray[np.float64],
        *,
        distance: ScoreDistance,
        contamination: float,
        units: Sequence[str] | None = None,
    ) -> Self:
        """The detector fitted on ``rows``, one per fitting window, each of ``units`` if given.

        Raises:
            InvalidDetectorRecipeError: If the distance is not one from the mean.
            ValueError: If ``units`` does not name one unit per row.
            InvalidScoredOutcomeError: If units are given and none holds two rows.
        """
        cls._require_from_mean(distance)
        standardisation = Standardisation.fitted_on(np.asarray(rows, dtype=np.float64))
        own = cls._from_mean(standardisation.of(rows), distance)
        return cls(
            standardisation=standardisation,
            distance=distance,
            alarm=ScoreAlarm.set_over(own, contamination=contamination, units=units),
        )

    def scores(
        self, rows: NDArray[np.float64], *, units: Sequence[str] | None = None
    ) -> NDArray[np.float64]:
        """The distance above the alarm of every row, in the order given.

        Raises:
            ValueError: If the rows come with units and the detector reads none, or the other
                way round, or ``units`` does not name one unit per row.
            InvalidScoredOutcomeError: If units are given and none holds two rows.
        """
        self.alarm.require(units)
        distances = self._from_mean(self.standardisation.of(rows), self.distance)
        return self.alarm.above(distances, units=units)

    @classmethod
    def _require_from_mean(cls, distance: ScoreDistance) -> None:
        if distance not in cls.DISTANCES:
            raise InvalidDetectorRecipeError(
                f"a distance from the mean is one of {sorted(cls.DISTANCES)}, not {distance}"
            )

    @staticmethod
    def _from_mean(
        standardised: NDArray[np.float64], distance: ScoreDistance
    ) -> NDArray[np.float64]:
        if distance is ScoreDistance.FARTHEST:
            farthest: NDArray[np.float64] = np.abs(standardised).max(axis=1)
            return farthest
        squared: NDArray[np.float64] = np.einsum("ij,ij->i", standardised, standardised)
        return squared

    def kept(self) -> RepresentationBytes:
        """The detector as it is kept, under the format ``from_bytes`` reads back."""
        return RepresentationBytes(self.FORMAT, self.to_bytes())

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        np.savez(
            buffer,
            centre=self.standardisation.centre,
            scale=self.standardisation.scale,
            distance=np.asarray(str(self.distance)),
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
                    standardisation=Standardisation(
                        centre=np.asarray(archive["centre"], dtype=np.float64),
                        scale=np.asarray(archive["scale"], dtype=np.float64),
                    ),
                    distance=ScoreDistance(str(archive["distance"])),
                    alarm=ScoreAlarm(
                        level=float(archive["alarm"]),
                        per_unit=bool(archive["per_unit"]),
                        unit_spread=float(archive["unit_spread"]),
                    ),
                )
        except (KeyError, ValueError, OSError, EOFError) as error:
            raise UnreadableFittedCandidateError(
                "the bytes are not a mean-distance detector's archive"
            ) from error
