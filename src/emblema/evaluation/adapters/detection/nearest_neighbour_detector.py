import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.domain.exceptions import (
    InvalidScoredOutcomeError,
    UnreadableFittedCandidateError,
)

# Scored rows are compared with the fitted ones this many at a time, so the matrix of distances
# never holds more than a few hundred thousand entries whatever the fitting side's size.
_CHUNK_ROWS = 512
# A unit's spread is put on the scale of a standard deviation, so the fallback spread a unit
# without one borrows reads in the same units as every other.
_MAD_TO_DEVIATION = 1.4826
# A spread this small beside its median is the rounding of the distances, not a spread: the
# distances are expanded as |a|² + |b|² - 2ab, which leaves noise of about 1e-8.
_NO_SPREAD = 1e-9


@dataclass(frozen=True)
class NearestNeighbourDetector:
    """A detector fitted on rows without a label: how far a row lies from the nearest fitted ones.

    Every feature is standardised over the fitting rows, a feature without spread left as it is,
    and a statistic a row cannot support, which the summaries mark as missing, is put at the
    fitting mean, so it moves no distance. A row's score is its mean Euclidean distance to its
    ``neighbours`` nearest fitting rows. The alarm is the quantile of the fitting rows' own
    scores, each read against the other fitting rows, that leaves the stated contamination of
    them above it, so a run answers the distance above the alarm and positive is an alarm.

    Fitted with the unit of every row, the detector reads each distance against its own unit
    first, the distance less the median of the unit's distances over their spread, and sets the
    alarm over the fitting rows read so; it then scores only rows that come with their units.

    Attributes:
        centre: The fitting mean of every feature.
        scale: The fitting spread of every feature, one where there was none.
        fitted: The fitting rows, standardised.
        neighbours: How many nearest fitted rows a score is the mean distance to.
        alarm: The score above which a row raises an alarm.
        per_unit: Whether a distance is read against the other rows of its unit.
        unit_spread: The spread a unit without one of its own is read over: the fitting side's,
            over all its distances; one where that has none either.
    """

    FORMAT: ClassVar[str] = "nearest-neighbour-npz"

    centre: NDArray[np.float64]
    scale: NDArray[np.float64]
    fitted: NDArray[np.float64]
    neighbours: int
    alarm: float
    per_unit: bool = False
    unit_spread: float = 1.0

    @classmethod
    def fitted_on(
        cls,
        rows: NDArray[np.float64],
        *,
        neighbours: int,
        contamination: float,
        units: Sequence[str] | None = None,
    ) -> Self:
        """The detector fitted on ``rows``, one per fitting window, each of ``units`` if given.

        A fitting side with no more rows than neighbours scores every row against all the others
        it has, and the detector keeps that count, so a scored row is read against as many
        neighbours as the alarm was set over.

        Raises:
            ValueError: If ``units`` does not name one unit per row.
            InvalidScoredOutcomeError: If units are given and none holds two rows, so no row
                has anything of its unit to be read against.
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
        unit_spread = 1.0
        if units is not None:
            unit_spread = cls._spread(scores) or 1.0
            scores = cls._against_own_unit(scores, units, fallback=unit_spread)
        alarm = float(np.quantile(scores, 1.0 - contamination))
        return cls(
            centre=centre,
            scale=scale,
            fitted=standardised,
            neighbours=kept,
            alarm=alarm,
            per_unit=units is not None,
            unit_spread=unit_spread,
        )

    def scores(
        self, rows: NDArray[np.float64], *, units: Sequence[str] | None = None
    ) -> NDArray[np.float64]:
        """The distance above the alarm of every row, in the order given.

        Read against their units, the rows of a unit are read together: a row's score depends on
        the other rows of its unit given in the same call, so a unit is scored whole, once.

        Raises:
            ValueError: If the rows come with units and the detector reads none, or the other
                way round, or ``units`` does not name one unit per row.
            InvalidScoredOutcomeError: If units are given and none holds two rows.
        """
        if (units is not None) != self.per_unit:
            raise ValueError(
                "this detector reads a distance against its unit and was given no units"
                if self.per_unit
                else "this detector reads no unit and was given units"
            )
        standardised = self._standardised(
            np.asarray(rows, dtype=np.float64), self.centre, self.scale
        )
        distances = self._distances(standardised, self.fitted, self.neighbours).mean(axis=1)
        if units is not None:
            distances = self._against_own_unit(distances, units, fallback=self.unit_spread)
        return distances - self.alarm

    @staticmethod
    def _spread(scores: NDArray[np.float64]) -> float:
        """The scaled median absolute deviation of ``scores``; zero where it is only rounding."""
        centre = float(np.median(scores))
        spread = _MAD_TO_DEVIATION * float(np.median(np.abs(scores - centre)))
        return spread if spread > _NO_SPREAD * (1.0 + abs(centre)) else 0.0

    @classmethod
    def _against_own_unit(
        cls, scores: NDArray[np.float64], units: Sequence[str], *, fallback: float
    ) -> NDArray[np.float64]:
        """Every score less its unit's median, over the unit's scaled median absolute deviation.

        A unit whose scores have no such spread, a lone row or rows alike, is read over
        ``fallback`` instead, so its scores stay on the scale of every other unit's.

        Raises:
            ValueError: If ``units`` does not name one unit per score.
            InvalidScoredOutcomeError: If no unit holds two scores.
        """
        keys = np.asarray([str(unit) for unit in units])
        if len(keys) != len(scores):
            raise ValueError(f"{len(keys)} units were given for {len(scores)} rows")
        found, counts = np.unique(keys, return_counts=True)
        if counts.max() < 2:
            raise InvalidScoredOutcomeError(
                "no unit holds two windows, so no window has its unit to be read against"
            )
        read = np.empty_like(scores)
        for unit in found:
            mask = keys == unit
            own = scores[mask]
            read[mask] = (own - float(np.median(own))) / (cls._spread(own) or fallback)
        return read

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
            per_unit=np.asarray(self.per_unit),
            unit_spread=np.asarray(self.unit_spread),
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
                    # A detector kept before units were read reads none.
                    per_unit="per_unit" in archive.files and bool(archive["per_unit"]),
                    unit_spread=(
                        float(archive["unit_spread"]) if "unit_spread" in archive.files else 1.0
                    ),
                )
        except (KeyError, ValueError, OSError, EOFError) as error:
            raise UnreadableFittedCandidateError(f"not a kept detector: {error}") from error
