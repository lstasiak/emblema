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
class SubspaceDetector:
    """A detector fitted on rows without a label, scoring a row by where it leaves their subspace.

    The classical reading of principal components: the leading components that carry most of the
    fitting rows' variance span the subspace normal rows move in. A row's score adds how far it
    lies within that subspace, Hotelling's statistic, each component's projection squared over
    its variance, to how far it lies outside it, the squared residual the subspace does not
    account for; each is read over its median on the fitting rows, so neither swamps the other.
    Features are standardised and the covariance shrunk as the whitened detector's are, and no
    more components are kept than the fitting rows span, since a direction only the shrinkage
    put variance in is no direction of theirs; the alarm is set over the fitting rows' own scores
    and read against a unit as the others' are.

    Attributes:
        standardisation: How every row is centred and scaled before it is projected.
        components: ``[features, kept]`` — the leading components, as columns.
        variances: ``[kept]`` — what each kept component's projection is read over.
        scales: The fitting medians of the within and the residual statistics, in that order.
        alarm: Where a score raises an alarm.
    """

    FORMAT: ClassVar[str] = "subspace-npz"
    # The leading components that together carry this share of the fitting variance are kept.
    SHARE: ClassVar[float] = 0.9
    # A statistic's scale is floored at this share of what the fitted covariance would give it
    # on average, the components kept for the one within and the variance left outside them for
    # the residual: fewer rows than features leave the fitting residual at nought, and a score
    # read over nought would be no score.
    SCALE_FLOOR: ClassVar[float] = 0.1

    standardisation: Standardisation
    components: NDArray[np.float64]
    variances: NDArray[np.float64]
    scales: tuple[float, float]
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
        eigenvalues, vectors = np.linalg.eigh(ShrunkCovariance.of(standardised).matrix)
        descending = np.argsort(eigenvalues)[::-1]
        eigenvalues, vectors = eigenvalues[descending], vectors[:, descending]
        kept = int(np.searchsorted(np.cumsum(eigenvalues) / eigenvalues.sum(), cls.SHARE) + 1)
        kept = min(kept, len(standardised) - 1, standardised.shape[1])
        components = vectors[:, :kept]
        variances = np.clip(eigenvalues[:kept], np.finfo(np.float64).tiny, None)
        within, outside = cls._statistics(standardised, components, variances)
        scales = (
            cls._median_scale(within, cls.SCALE_FLOOR * kept),
            cls._median_scale(outside, cls.SCALE_FLOOR * float(eigenvalues[kept:].sum())),
        )
        own = cls._combined(within, outside, scales)
        return cls(
            standardisation=standardisation,
            components=components,
            variances=variances,
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
        within, outside = self._statistics(
            self.standardisation.of(rows), self.components, self.variances
        )
        return self.alarm.above(self._combined(within, outside, self.scales), units=units)

    @staticmethod
    def _statistics(
        standardised: NDArray[np.float64],
        components: NDArray[np.float64],
        variances: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Hotelling's statistic within the subspace and the squared residual outside it."""
        projected = standardised @ components
        within: NDArray[np.float64] = (projected**2 / variances).sum(axis=1)
        reconstructed = projected @ components.T
        residual = standardised - reconstructed
        outside: NDArray[np.float64] = np.einsum("ij,ij->i", residual, residual)
        return within, outside

    @staticmethod
    def _median_scale(statistic: NDArray[np.float64], floor: float) -> float:
        """The fitting median of a statistic, floored where it is rounding rather than a spread."""
        return max(float(np.median(statistic)), floor, np.finfo(np.float64).tiny)

    @staticmethod
    def _combined(
        within: NDArray[np.float64], outside: NDArray[np.float64], scales: tuple[float, float]
    ) -> NDArray[np.float64]:
        combined: NDArray[np.float64] = within / scales[0] + outside / scales[1]
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
            components=self.components,
            variances=self.variances,
            scales=np.asarray(self.scales),
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
                within, outside = (float(s) for s in archive["scales"])
                return cls(
                    standardisation=Standardisation(
                        centre=np.asarray(archive["centre"], dtype=np.float64),
                        scale=np.asarray(archive["scale"], dtype=np.float64),
                    ),
                    components=np.asarray(archive["components"], dtype=np.float64),
                    variances=np.asarray(archive["variances"], dtype=np.float64),
                    scales=(within, outside),
                    alarm=ScoreAlarm(
                        level=float(archive["alarm"]),
                        per_unit=bool(archive["per_unit"]),
                        unit_spread=float(archive["unit_spread"]),
                    ),
                )
        except (KeyError, ValueError, OSError, EOFError) as error:
            raise UnreadableFittedCandidateError(
                "the bytes are not a subspace detector's archive"
            ) from error
