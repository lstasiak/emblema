import io
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError

# Scored rows are compared with the fitted ones this many at a time, so the matrix of distances
# never holds more than a few hundred thousand entries whatever the fitting side's size.
_CHUNK_ROWS = 512


@dataclass(frozen=True)
class NearestNeighbourDetector:
    """A detector fitted on rows without a label: how far a row lies from the nearest fitted ones.

    Every feature is standardised over the fitting rows, a feature without spread left as it is,
    and a statistic a row cannot support, which the summaries mark as missing, is put at the
    fitting mean, so it moves no distance. A row's score is its mean Euclidean distance to its
    ``neighbours`` nearest fitting rows. The alarm is the quantile of the fitting rows' own
    scores, each read against the other fitting rows, that leaves the stated contamination of
    them above it, so a run answers the distance above the alarm and positive is an alarm.

    Attributes:
        centre: The fitting mean of every feature.
        scale: The fitting spread of every feature, one where there was none.
        fitted: The fitting rows, standardised.
        neighbours: How many nearest fitted rows a score is the mean distance to.
        alarm: The score above which a row raises an alarm.
    """

    FORMAT: ClassVar[str] = "nearest-neighbour-npz"

    centre: NDArray[np.float64]
    scale: NDArray[np.float64]
    fitted: NDArray[np.float64]
    neighbours: int
    alarm: float

    @classmethod
    def fitted_on(cls, rows: NDArray[np.float64], *, neighbours: int, contamination: float) -> Self:
        """The detector fitted on ``rows``, one per fitting window.

        A fitting side with no more rows than neighbours scores every row against all the others
        it has, and the detector keeps that count, so a scored row is read against as many
        neighbours as the alarm was set over.
        """
        rows = np.asarray(rows, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            centre = np.nan_to_num(np.nanmean(rows, axis=0), nan=0.0)
            scale = np.nan_to_num(np.nanstd(rows, axis=0), nan=0.0)
        scale = np.where(scale > 0.0, scale, 1.0)
        standardised = cls._standardised(rows, centre, scale)
        kept = min(neighbours, max(len(standardised) - 1, 1))
        # Each fitting row's nearest neighbour is itself at no distance, so one more is asked
        # for and the first dropped; a lone fitting row scores zero and the alarm stays there.
        own = (
            cls._distances(standardised, standardised, kept + 1)[:, 1:]
            if len(standardised) > 1
            else np.zeros((1, 1))
        )
        scores = own.mean(axis=1)
        alarm = float(np.quantile(scores, 1.0 - contamination))
        return cls(centre=centre, scale=scale, fitted=standardised, neighbours=kept, alarm=alarm)

    def scores(self, rows: NDArray[np.float64]) -> NDArray[np.float64]:
        """The distance above the alarm of every row, in the order given."""
        standardised = self._standardised(
            np.asarray(rows, dtype=np.float64), self.centre, self.scale
        )
        return self._distances(standardised, self.fitted, self.neighbours).mean(axis=1) - self.alarm

    @staticmethod
    def _standardised(
        rows: NDArray[np.float64], centre: NDArray[np.float64], scale: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.nan_to_num((rows - centre) / scale, nan=0.0)

    @staticmethod
    def _distances(
        rows: NDArray[np.float64], against: NDArray[np.float64], nearest: int
    ) -> NDArray[np.float64]:
        """The ``nearest`` smallest Euclidean distances from each of ``rows`` to ``against``."""
        squared_against = np.einsum("ij,ij->i", against, against)
        kept = []
        for start in range(0, len(rows), _CHUNK_ROWS):
            chunk = rows[start : start + _CHUNK_ROWS]
            squared = (
                np.einsum("ij,ij->i", chunk, chunk)[:, None]
                + squared_against[None, :]
                - 2.0 * chunk @ against.T
            )
            nearest_squared = np.partition(np.maximum(squared, 0.0), nearest - 1, axis=1)[
                :, :nearest
            ]
            kept.append(np.sqrt(np.sort(nearest_squared, axis=1)))
        return np.concatenate(kept)

    def to_bytes(self) -> bytes:
        """The detector as numpy's archive of its arrays, the form it is kept in."""
        buffer = io.BytesIO()
        np.savez(
            buffer,
            centre=self.centre,
            scale=self.scale,
            fitted=self.fitted,
            neighbours=np.asarray(self.neighbours),
            alarm=np.asarray(self.alarm),
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
                    centre=np.asarray(archive["centre"], dtype=np.float64),
                    scale=np.asarray(archive["scale"], dtype=np.float64),
                    fitted=np.asarray(archive["fitted"], dtype=np.float64),
                    neighbours=int(archive["neighbours"]),
                    alarm=float(archive["alarm"]),
                )
        except (KeyError, ValueError, OSError, EOFError) as error:
            raise UnreadableFittedCandidateError(f"not a kept detector: {error}") from error
