from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ShrunkCovariance:
    """The covariance of fitting rows, shrunk towards a scaled identity so it always inverts.

    The shrinkage intensity is Ledoit and Wolf's closed form (2004), the one scikit-learn's
    ``LedoitWolf`` computes: a side of many rows is shrunk hardly at all, a side of fewer rows
    than features enough to invert.

    Invariants: at least two rows went into it; the intensity lies in [0, 1].

    Attributes:
        matrix: The shrunk covariance.
        shrinkage: How far the sample covariance was pulled towards the scaled identity.
    """

    # A covariance without spread in some direction has nothing there to invert; the floor,
    # relative to the largest eigenvalue or to the unit scale of standardised rows where that
    # is nought, keeps the inverse root finite.
    EIGENVALUE_FLOOR: ClassVar[float] = 1e-12

    matrix: NDArray[np.float64]
    shrinkage: float

    @classmethod
    def of(cls, rows: NDArray[np.float64]) -> Self:
        """The shrunk covariance of ``rows``, one per fitting window.

        Raises:
            ValueError: If there are fewer than two rows.
        """
        if len(rows) < 2:
            raise ValueError(f"a covariance needs two rows, got {len(rows)}")
        centred = rows - rows.mean(axis=0)
        count, width = centred.shape
        sample = centred.T @ centred / count
        mu = float(np.trace(sample)) / width
        squared = centred**2
        beta_sum = float((squared.T @ squared).sum()) / count - float((sample**2).sum())
        delta = float(((sample - mu * np.eye(width)) ** 2).sum())
        shrinkage = min(beta_sum / count, delta) / delta if delta > 0 else 0.0
        return cls(
            matrix=(1 - shrinkage) * sample + shrinkage * mu * np.eye(width),
            shrinkage=shrinkage,
        )

    def inverse_root(self) -> NDArray[np.float64]:
        """The symmetric inverse square root: rows multiplied by it are whitened."""
        eigenvalues, vectors = np.linalg.eigh(self.matrix)
        floor = self.EIGENVALUE_FLOOR * max(float(eigenvalues.max()), 1.0)
        root: NDArray[np.float64] = (
            vectors @ np.diag(1.0 / np.sqrt(np.clip(eigenvalues, floor, None))) @ vectors.T
        )
        return root
