import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.adapters.detection.score_alarm import ScoreAlarm
from emblema.evaluation.adapters.detection.shrunk_covariance import ShrunkCovariance
from emblema.evaluation.adapters.detection.standardisation import Standardisation
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError


@dataclass(frozen=True)
class WhitenedGaussianDetector:
    """A detector fitted on rows without a label, scoring a row by its whitened distance from them.

    The score is the squared Mahalanobis distance from the fitting mean under the fitting rows'
    shrunk covariance, over features standardised as the nearest-neighbour detector standardises
    them; the alarm is set over the fitting rows' own scores and read against a unit as its is.
    A feature the fitting rows never vary in keeps only the shrinkage as its variance, so a row
    that varies in it scores far out.

    Attributes:
        standardisation: How every row is centred and scaled before it is whitened.
        root: The inverse square root of the standardised fitting rows' covariance.
        alarm: Where a distance raises an alarm.
    """

    FORMAT: ClassVar[str] = "whitened-gaussian-npz"

    standardisation: Standardisation
    root: NDArray[np.float64]
    alarm: ScoreAlarm

    @classmethod
    def fitted_on(
        cls,
        rows: NDArray[np.float64],
        *,
        contamination: float,
        units: Sequence[str] | None = None,
    ) -> Self:
        """The detector fitted on ``rows``, one per fitting window, each of ``units`` if given.

        Raises:
            ValueError: If there are fewer than two rows, or ``units`` does not name one unit
                per row.
            InvalidScoredOutcomeError: If units are given and none holds two rows.
        """
        standardisation = Standardisation.fitted_on(np.asarray(rows, dtype=np.float64))
        standardised = standardisation.of(rows)
        root = ShrunkCovariance.of(standardised).inverse_root()
        return cls(
            standardisation=standardisation,
            root=root,
            alarm=ScoreAlarm.set_over(
                cls._squared(standardised, root), contamination=contamination, units=units
            ),
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
        squared = self._squared(self.standardisation.of(rows), self.root)
        return self.alarm.above(squared, units=units)

    @staticmethod
    def _squared(
        standardised: NDArray[np.float64], root: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        whitened = standardised @ root
        squared: NDArray[np.float64] = np.einsum("ij,ij->i", whitened, whitened)
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
            root=self.root,
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
                    root=np.asarray(archive["root"], dtype=np.float64),
                    alarm=ScoreAlarm(
                        level=float(archive["alarm"]),
                        per_unit=bool(archive["per_unit"]),
                        unit_spread=float(archive["unit_spread"]),
                    ),
                )
        except (KeyError, ValueError, OSError, EOFError) as error:
            raise UnreadableFittedCandidateError(
                "the bytes are not a whitened Gaussian detector's archive"
            ) from error
