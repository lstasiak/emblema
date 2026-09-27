import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import RidgeCV
from threadpoolctl import threadpool_limits

from emblema.evaluation.adapters.grid.regular_grid import RegularGrid
from emblema.evaluation.adapters.minirocket.minirocket_transform import MiniRocketTransform
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.classical.random_convolutions import RandomConvolutions
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError
from emblema.evaluation.domain.heads.logistic_calibration import LogisticCalibration
from emblema.evaluation.domain.labels.target_kind import TargetKind
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

    The grid's rows are the ones the fitted windows give something to read — a sensor read per
    operating condition is six channels of the vocabulary, and a task over one condition holds
    one of them — and they are kept with the candidate, so it answers every window over the rows
    it was fitted on.

    Over outcomes the linear map ranks them as the reference classifier's does, and a logistic
    calibration fitted on the leave-one-out answers of the chosen penalty turns its score into
    a probability; the calibration's presence is what marks a candidate of a binary task.

    Stored as named arrays in NumPy's own archive format, read back without unpickling anything,
    so the document outlives the release of every library that fitted it. A kept candidate's
    manifest names this form ``FORMAT``; it is the measured form and the only one.

    Attributes:
        parameters: The recipe the candidate was fitted under, flattened to scalars.
        grid_steps: Equal steps a window is laid on, one per unit of the corpus's time.
        channels: Channels of the corpus's vocabulary a window is first laid over.
        rows: The rows of that grid the transform reads: values, then masks, as grid rows.
        transform: The fitted convolutions.
        feature_scale: What each feature is divided by before the linear map.
        weights: The linear map, one weight per feature.
        intercept: What the map adds.
        penalty: The penalty the fit chose.
        target_scale: What the targets were divided by, so an answer is read in the task's unit.
        calibration: How the map's score becomes the probability of the positive outcome, for a
            candidate of a binary task; ``None`` for one that answers a quantity.
    """

    FORMAT: ClassVar[str] = "minirocket-npz"

    parameters: dict[str, str | int | float]
    grid_steps: int
    channels: int
    rows: NDArray[np.int64]
    transform: MiniRocketTransform
    feature_scale: NDArray[np.float64]
    weights: NDArray[np.float64]
    intercept: float
    penalty: float
    target_scale: float
    calibration: LogisticCalibration | None

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
        kind: TargetKind,
    ) -> Self:
        """Fit ``method`` on ``windows`` answering ``targets``, each already divided by the scale.

        Every draw is seeded by the recipe and the linear algebra runs on the threads it names.
        The penalty is chosen by leave-one-out squared error for either kind of target; over
        outcomes the leave-one-out answers are kept to calibrate the chosen map on.

        Raises:
            UncalibratableScoresError: If the outcomes cannot be calibrated.
        """
        steps = grid_steps
        with threadpool_limits(limits=method.ridge.threads):
            grid = RegularGrid(steps, channels)
            whole = grid.of(windows)
            rows = grid.rows_read(whole)
            laid = whole[:, rows]
            transform = MiniRocketTransform.fitted(
                laid, method.convolutions.features, np.random.default_rng(recipe.seed)
            )
            features = transform.of(laid)
            scale = cls._scale_of(features)
            calibration = None
            match kind:
                case TargetKind.CONTINUOUS:
                    ridge = RidgeCV(alphas=method.ridge.penalties).fit(features / scale, targets)
                case TargetKind.BINARY:
                    # Named rather than left to the default, because only a named scoring keeps
                    # the leave-one-out answers; it chooses the same penalty the default does.
                    ridge = RidgeCV(
                        alphas=method.ridge.penalties,
                        scoring="neg_mean_squared_error",
                        store_cv_results=True,
                    ).fit(features / scale, targets)
                    chosen = list(method.ridge.penalties).index(float(ridge.alpha_))
                    calibration = LogisticCalibration.fitted(
                        ridge.cv_results_[:, chosen].tolist(), targets.tolist()
                    )
        return cls(
            parameters=recipe.parameters(),
            grid_steps=steps,
            channels=channels,
            rows=rows,
            transform=transform,
            feature_scale=scale,
            weights=np.asarray(ridge.coef_, dtype=np.float64),
            intercept=float(ridge.intercept_),
            penalty=float(ridge.alpha_),
            target_scale=target_scale,
            calibration=calibration,
        )

    def predict(self, windows: list[TokenWindow], *, threads: int) -> NDArray[np.float64]:
        """The answer for each window: in the task's own unit, or a probability."""
        with threadpool_limits(limits=threads):
            laid = RegularGrid(self.grid_steps, self.channels).of(windows)[:, self.rows]
            features = self.transform.of(laid)
            answers = (features / self.feature_scale) @ self.weights + self.intercept
        if self.calibration is None:
            return np.asarray(answers * self.target_scale, dtype=np.float64)
        log_odds = self.calibration.slope * answers + self.calibration.intercept
        # The logistic function written through a log-sum, so no log-odds overflows it.
        return np.asarray(np.exp(-np.logaddexp(0.0, -log_odds)), dtype=np.float64)

    @staticmethod
    def _scale_of(features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Each feature's spread, and one for a feature that never varies, which then stays put.

        Scaled but not centred, as the reference regressor is: the intercept absorbs the means.
        """
        spread = features.std(axis=0)
        return np.where(spread > 0.0, spread, 1.0)

    def to_bytes(self) -> bytes:
        arrays: dict[str, NDArray[Any]] = {
            "parameters": np.array(json.dumps(self.parameters)),
            "grid": np.array([self.grid_steps, self.channels, self.transform.length]),
            "rows": self.rows,
            "dilations": self.transform.dilations,
            "per_dilation": self.transform.per_dilation,
            "combination_sizes": self.transform.combination_sizes,
            "combination_channels": self.transform.channels,
            "biases": self.transform.biases,
            "feature_scale": self.feature_scale,
            "weights": self.weights,
            "scalars": np.array([self.intercept, self.penalty, self.target_scale]),
        }
        if self.calibration is not None:
            arrays["calibration"] = np.array([self.calibration.slope, self.calibration.intercept])
        buffer = io.BytesIO()
        np.savez(buffer, allow_pickle=False, **arrays)
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
                # A candidate kept before outcomes were answered holds no calibration.
                calibration = (
                    None
                    if "calibration" not in stored.files
                    else LogisticCalibration(
                        slope=float(stored["calibration"][0]),
                        intercept=float(stored["calibration"][1]),
                    )
                )
                return cls(
                    parameters=json.loads(str(stored["parameters"])),
                    grid_steps=steps,
                    channels=channels,
                    rows=stored["rows"],
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
                    calibration=calibration,
                )
        except UNREADABLE_BYTES as error:
            raise UnreadableFittedCandidateError(
                f"not fitted convolutions this can read: {error}"
            ) from error
