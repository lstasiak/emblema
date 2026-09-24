import io
import json
import zipfile
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import RidgeCV
from threadpoolctl import threadpool_limits

from emblema.evaluation.adapters.grid.regular_grid import RegularGrid
from emblema.evaluation.adapters.minirocket.minirocket_transform import MiniRocketTransform
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.classical.random_convolutions import RandomConvolutions
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError
from emblema.shared.kernel.tokens import TokenWindow

# What comes back from handing numpy bytes it did not write, or an archive missing what this
# reads. The reasons differ and to a caller they are one thing: this artifact is not ours.
UNREADABLE_BYTES = (KeyError, ValueError, TypeError, OSError, EOFError, zipfile.BadZipFile)


@dataclass(frozen=True, eq=False)
class FittedConvolutions:
    """A MiniRocket candidate as one fit left it, and everything it takes to answer again.

    The convolutions alone would not be enough: a window has to be laid on the same grid, its
    features scaled as the fitted ones were, and the answer read back in the task's unit. So
    the grid, the scale of every feature, the linear map and the target's scale travel with the
    transform, and the object that answered the scored windows of a fit is the one that is kept.

    Stored as named arrays in NumPy's own archive format, read back without unpickling anything,
    so the document outlives the release of every library that fitted it.

    Attributes:
        parameters: The recipe the candidate was fitted under, flattened to scalars.
        grid_steps: Equal steps a window is laid on, one per unit of the corpus's time.
        channels: Channels of the corpus the grid is laid over.
        transform: The fitted convolutions.
        feature_scale: What each feature is divided by before the linear map.
        weights: The linear map, one weight per feature.
        intercept: What the map adds.
        penalty: The penalty the fit chose.
        target_scale: What the targets were divided by, so an answer is read in the task's unit.
    """

    parameters: dict[str, str | int | float]
    grid_steps: int
    channels: int
    transform: MiniRocketTransform
    feature_scale: NDArray[np.float64]
    weights: NDArray[np.float64]
    intercept: float
    penalty: float
    target_scale: float

    @classmethod
    def fitted(
        cls,
        recipe: ClassicalRecipe,
        method: RandomConvolutions,
        channels: int,
        grid_steps: int,
        windows: list[TokenWindow],
        targets: NDArray[np.float64],
        target_scale: float,
    ) -> Self:
        """Fit ``method`` on ``windows`` answering ``targets``, each already divided by the scale.

        Every draw is seeded by the recipe and the linear algebra runs on the threads it names.
        """
        steps = grid_steps
        with threadpool_limits(limits=method.ridge.threads):
            laid = RegularGrid(steps, channels).of(windows)
            transform = MiniRocketTransform.fitted(
                laid, method.convolutions.features, np.random.default_rng(recipe.seed)
            )
            features = transform.of(laid)
            scale = cls._scale_of(features)
            ridge = RidgeCV(alphas=method.ridge.penalties).fit(features / scale, targets)
        return cls(
            parameters=recipe.parameters(),
            grid_steps=steps,
            channels=channels,
            transform=transform,
            feature_scale=scale,
            weights=np.asarray(ridge.coef_, dtype=np.float64),
            intercept=float(ridge.intercept_),
            penalty=float(ridge.alpha_),
            target_scale=target_scale,
        )

    def predict(self, windows: list[TokenWindow], *, threads: int) -> NDArray[np.float64]:
        """The answer for each window, in the task's own unit."""
        with threadpool_limits(limits=threads):
            features = self.transform.of(RegularGrid(self.grid_steps, self.channels).of(windows))
            answers = (features / self.feature_scale) @ self.weights + self.intercept
        return np.asarray(answers * self.target_scale, dtype=np.float64)

    @staticmethod
    def _scale_of(features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Each feature's spread, and one for a feature that never varies, which then stays put.

        Scaled but not centred, as the reference regressor is: the intercept absorbs the means.
        """
        spread = features.std(axis=0)
        return np.where(spread > 0.0, spread, 1.0)

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        np.savez(
            buffer,
            parameters=np.array(json.dumps(self.parameters)),
            grid=np.array([self.grid_steps, self.channels, self.transform.length]),
            dilations=self.transform.dilations,
            per_dilation=self.transform.per_dilation,
            combination_sizes=self.transform.combination_sizes,
            combination_channels=self.transform.channels,
            biases=self.transform.biases,
            feature_scale=self.feature_scale,
            weights=self.weights,
            scalars=np.array([self.intercept, self.penalty, self.target_scale]),
        )
        return buffer.getvalue()

    @classmethod
    def read(cls, content: bytes) -> Self:
        """The candidate those bytes hold.

        Raises:
            UnreadableFittedCandidateError: If the bytes are not fitted convolutions of ours.
        """
        try:
            with np.load(io.BytesIO(content), allow_pickle=False) as stored:
                steps, channels, length = (int(value) for value in stored["grid"])
                intercept, penalty, target_scale = (float(value) for value in stored["scalars"])
                return cls(
                    parameters=json.loads(str(stored["parameters"])),
                    grid_steps=steps,
                    channels=channels,
                    transform=MiniRocketTransform(
                        length=length,
                        dilations=stored["dilations"],
                        per_dilation=stored["per_dilation"],
                        combination_sizes=stored["combination_sizes"],
                        channels=stored["combination_channels"],
                        biases=stored["biases"],
                    ),
                    feature_scale=stored["feature_scale"],
                    weights=stored["weights"],
                    intercept=intercept,
                    penalty=penalty,
                    target_scale=target_scale,
                )
        except UNREADABLE_BYTES as error:
            raise UnreadableFittedCandidateError(
                f"not fitted convolutions this can read: {error}"
            ) from error
