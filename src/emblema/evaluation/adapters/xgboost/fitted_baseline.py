import io
import pickle
import zipfile
from dataclasses import dataclass
from typing import ClassVar, Self

import joblib
import xgboost

from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError

# What comes back from handing joblib bytes it did not write, or a document missing what this
# reads. The reasons differ and to a caller they are one thing: this artifact is not ours.
UNREADABLE_BYTES = (
    KeyError,
    IndexError,
    TypeError,
    ValueError,
    AttributeError,
    ModuleNotFoundError,
    EOFError,
    OSError,
    pickle.UnpicklingError,
    zipfile.BadZipFile,
    xgboost.core.XGBoostError,
)


@dataclass(frozen=True)
class FittedBaseline:
    """A classical candidate as one fit left it: its trees, and what it takes to read them again.

    The trees alone would not be enough. A row means what the feature scheme says it means, and
    an answer is in units of the target the fit was scaled to, so the scheme's column names and
    that scale travel with the model. Together they are what another context needs in order to
    put a window in and get an answer out without being told anything else.

    The model is kept in the format XGBoost writes for keeping models, not as a pickled
    estimator: the document that carries it is then plain data, and a release of the library
    that no longer unpickles last year's object still loads last year's trees. A kept
    candidate's manifest names this form ``FORMAT``; it is the measured form and the only one.

    Attributes:
        parameters: The recipe the candidate was fitted under, flattened to scalars.
        feature_names: What each column of a row holds, in order.
        target_scale: What the targets were divided by, so an answer can be read back in the
            task's own unit.
        model: The fitted trees, as XGBoost's own portable encoding of them.
    """

    FORMAT: ClassVar[str] = "xgboost-joblib"

    parameters: dict[str, str | int | float]
    feature_names: tuple[str, ...]
    target_scale: float
    model: bytes

    @classmethod
    def of(
        cls,
        recipe: ClassicalRecipe,
        fitted: xgboost.XGBRegressor,
        *,
        feature_names: tuple[str, ...],
        target_scale: float,
    ) -> Self:
        return cls(
            parameters=recipe.parameters(),
            feature_names=feature_names,
            target_scale=target_scale,
            model=bytes(fitted.get_booster().save_raw(raw_format="ubj")),
        )

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        joblib.dump(
            {
                "parameters": self.parameters,
                "feature_names": list(self.feature_names),
                "target_scale": self.target_scale,
                "model": self.model,
            },
            buffer,
        )
        return buffer.getvalue()

    @classmethod
    def read(cls, content: bytes) -> Self:
        """The candidate those bytes hold.

        Raises:
            UnreadableFittedCandidateError: If the bytes are not a fitted baseline of ours.
        """
        try:
            stored = joblib.load(io.BytesIO(content))
            return cls(
                parameters=stored["parameters"],
                feature_names=tuple(stored["feature_names"]),
                target_scale=stored["target_scale"],
                model=stored["model"],
            )
        except UNREADABLE_BYTES as error:
            raise UnreadableFittedCandidateError(
                f"not a fitted baseline this can read: {error}"
            ) from error

    def booster(self) -> xgboost.Booster:
        """The trees, loaded and ready to answer rows laid out as ``feature_names`` says.

        Raises:
            UnreadableFittedCandidateError: If the stored trees are not ones XGBoost reads.
        """
        loaded = xgboost.Booster()
        try:
            loaded.load_model(bytearray(self.model))
        except UNREADABLE_BYTES as error:
            raise UnreadableFittedCandidateError(f"not trees this can read: {error}") from error
        return loaded
