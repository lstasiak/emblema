import io
import json
import warnings
import zipfile
from dataclasses import dataclass
from typing import Any, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from threadpoolctl import threadpool_limits

from emblema.evaluation.adapters.grid.regular_grid import RegularGrid
from emblema.evaluation.adapters.minirocket.minirocket_transform import MiniRocketTransform
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.classical.random_convolutions import RandomConvolutions
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError, UnsolvableHeadError
from emblema.evaluation.domain.heads.outcome_folds import OutcomeFolds
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

    Over outcomes the linear map is a logistic regression over the same scaled features, its L2
    penalty chosen among the same strengths by the log-loss of ``OutcomeFolds``: its answer is
    the log-odds, read as a probability with nothing fitted afterwards that could reorder or
    flatten it. The kind of target is kept with the map, since it says how an answer is read.

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
        kind: What the map answers: a quantity in units of the scale, or an outcome's log-odds.
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
    kind: TargetKind

    # Enough for the quasi-Newton solver to reach its tolerance at the weakest penalty over ten
    # thousand features; a fit that needs more is refused rather than read half-way.
    _ITERATIONS: ClassVar[int] = 5000
    # A spread below this share of a feature's largest value is rounding: thousands of units in
    # the last place, where a feature that truly varies moves by at least one position of its
    # convolution in a window's length.
    _RESOLUTION: ClassVar[float] = 1e-12

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
        A quantity's penalty is chosen by leave-one-out squared error, an outcome's by the
        log-loss of the folds.

        Raises:
            UnfoldableOutcomesError: If the outcomes leave no folds to choose a penalty on.
            UnsolvableHeadError: If the logistic regression does not converge.
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
            match kind:
                case TargetKind.CONTINUOUS:
                    ridge = RidgeCV(alphas=method.ridge.penalties).fit(features / scale, targets)
                    weights, intercept = np.asarray(ridge.coef_), float(ridge.intercept_)
                    penalty = float(ridge.alpha_)
                case TargetKind.BINARY:
                    weights, intercept, penalty = cls._logistic(
                        features / scale, targets, method.ridge.penalties
                    )
        return cls(
            parameters=recipe.parameters(),
            grid_steps=steps,
            channels=channels,
            rows=rows,
            transform=transform,
            feature_scale=scale,
            weights=np.asarray(weights, dtype=np.float64),
            intercept=intercept,
            penalty=penalty,
            target_scale=target_scale,
            kind=kind,
        )

    def predict(self, windows: list[TokenWindow], *, threads: int) -> NDArray[np.float64]:
        """The answer for each window: in the task's own unit, or a probability."""
        with threadpool_limits(limits=threads):
            laid = RegularGrid(self.grid_steps, self.channels).of(windows)[:, self.rows]
            features = self.transform.of(laid)
            answers = (features / self.feature_scale) @ self.weights + self.intercept
        match self.kind:
            case TargetKind.CONTINUOUS:
                return np.asarray(answers * self.target_scale, dtype=np.float64)
            case TargetKind.BINARY:
                # The logistic function written through a log-sum, so no log-odds overflows it.
                return np.asarray(np.exp(-np.logaddexp(0.0, -answers)), dtype=np.float64)

    @classmethod
    def _logistic(
        cls,
        scaled: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        penalties: tuple[float, ...],
    ) -> tuple[NDArray[np.float64], float, float]:
        """The weights, intercept and penalty of the logistic regression chosen over the folds.

        The library's inverse strength is the reciprocal of the penalty, given in descending
        order so that the first best, which it keeps, is the smaller penalty on a tie.

        Raises:
            UnfoldableOutcomesError: If the outcomes leave no folds to choose a penalty on.
            UnsolvableHeadError: If a fit does not converge.
        """
        folds = OutcomeFolds.of(outcomes.tolist())
        splits = [
            (list(folds.kept(fold)), list(folds.held_out(fold))) for fold in range(folds.count)
        ]
        # The only report that covers the fit on every window as well as those on the folds.
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always", ConvergenceWarning)
            fit = LogisticRegressionCV(
                Cs=[1.0 / penalty for penalty in penalties],
                l1_ratios=(0.0,),
                cv=splits,
                scoring="neg_log_loss",
                max_iter=cls._ITERATIONS,
                use_legacy_attributes=False,
            ).fit(scaled, outcomes)
        if any(issubclass(warning.category, ConvergenceWarning) for warning in raised):
            raise UnsolvableHeadError(
                f"the logistic regression did not converge in {cls._ITERATIONS} iterations"
            )
        return (
            np.asarray(fit.coef_[0], dtype=np.float64),
            float(fit.intercept_[0]),
            1.0 / float(fit.C_),
        )

    @classmethod
    def _scale_of(cls, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Each feature's spread, and one for a feature that never varies, which then stays put.

        Scaled but not centred, as the reference regressor is: the intercept absorbs the means.
        A feature varies when its spread stands above the rounding of its own values: one that is
        constant in value but not in its last bits, divided by that spread, would turn rounding
        into a feature of unit spread around an enormous mean, which least squares centres away
        and a logistic regression cannot step past.
        """
        spread = features.std(axis=0)
        magnitude = np.abs(features).max(axis=0, initial=0.0)
        return np.where(spread > cls._RESOLUTION * magnitude, spread, 1.0)

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
            "kind": np.array(self.kind.value),
        }
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
                # A candidate kept before outcomes were answered names no kind: it answers a
                # quantity.
                kind = (
                    TargetKind(str(stored["kind"]))
                    if "kind" in stored.files
                    else TargetKind.CONTINUOUS
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
                    kind=kind,
                )
        except UNREADABLE_BYTES as error:
            raise UnreadableFittedCandidateError(
                f"not fitted convolutions this can read: {error}"
            ) from error
