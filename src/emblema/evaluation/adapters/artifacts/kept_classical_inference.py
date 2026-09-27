from collections.abc import Sequence
from typing import TYPE_CHECKING

from emblema.evaluation.adapters.features.window_features import WindowFeatures, features_for
from emblema.evaluation.adapters.minirocket.fitted_convolutions import FittedConvolutions
from emblema.evaluation.adapters.xgboost.fitted_baseline import FittedBaseline
from emblema.evaluation.contracts.exceptions import InvalidKeptRepresentationError
from emblema.evaluation.contracts.kept_representation import KeptRepresentation
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow

if TYPE_CHECKING:
    import xgboost

# Every fit records the threads it ran on; a candidate kept before that was recorded answers on
# one, which is also the only count that cannot contend with the process serving it.
_ONE_THREAD = 1


class KeptClassicalInference:
    """Answers windows from the two classical forms this context keeps: trees and convolutions.

    The service the Evaluation context publishes for whoever serves a classical candidate. A
    form is read back through the codec that wrote it and asked what it answers, under the same
    reading of a window the fit used: the trees name their feature scheme and their columns, the
    convolutions carry their grid, so nothing about how a window is read is decided here.

    Loaded candidates are kept by the checksum of their bytes for the life of the process, since
    a service answers many requests with the same few candidates and the bytes of a content-
    addressed artifact never change under their checksum.
    """

    def __init__(self) -> None:
        self._trees: dict[Checksum, tuple[xgboost.Booster, WindowFeatures, float]] = {}
        self._convolutions: dict[Checksum, FittedConvolutions] = {}

    def reads(self, format: str) -> bool:
        return format in (FittedBaseline.FORMAT, FittedConvolutions.FORMAT)

    def answer(
        self,
        form: KeptRepresentation,
        content: bytes,
        windows: Sequence[TokenWindow],
        *,
        channels: int,
    ) -> tuple[float, ...]:
        try:
            if form.format == FittedBaseline.FORMAT:
                return self._trees_answer(form.artifact.checksum, content, windows, channels)
            if form.format == FittedConvolutions.FORMAT:
                return self._convolutions_answer(form.artifact.checksum, content, windows)
        except UnreadableFittedCandidateError as error:
            raise InvalidKeptRepresentationError(
                f"the form {form.format!r} under {form.artifact.key!r} is not one this reads: "
                f"{error}"
            ) from error
        raise InvalidKeptRepresentationError(f"no classical candidate is kept as {form.format!r}")

    def _trees_answer(
        self, checksum: Checksum, content: bytes, windows: Sequence[TokenWindow], channels: int
    ) -> tuple[float, ...]:
        if checksum not in self._trees:
            fitted = FittedBaseline.read(content)
            self._trees[checksum] = (
                fitted.booster(),
                self._features_of(fitted, channels),
                fitted.target_scale,
            )
        booster, features, target_scale = self._trees[checksum]
        predicted = booster.inplace_predict(features.of(list(windows))) * target_scale
        return tuple(float(answer) for answer in predicted.tolist())

    def _convolutions_answer(
        self, checksum: Checksum, content: bytes, windows: Sequence[TokenWindow]
    ) -> tuple[float, ...]:
        if checksum not in self._convolutions:
            self._convolutions[checksum] = FittedConvolutions.read(content)
        fitted = self._convolutions[checksum]
        threads = fitted.parameters.get("threads", _ONE_THREAD)
        predicted = fitted.predict(list(windows), threads=int(threads))
        return tuple(float(answer) for answer in predicted.tolist())

    @staticmethod
    def _features_of(fitted: FittedBaseline, channels: int) -> WindowFeatures:
        """The reading of a window the trees were fitted under, held to the columns they name.

        Raises:
            UnreadableFittedCandidateError: If the trees name a scheme this context has no
                reading for, or their columns are not that reading's over ``channels``.
        """
        stated = fitted.parameters.get("features")
        try:
            scheme = FeatureScheme(str(stated))
        except ValueError as error:
            raise UnreadableFittedCandidateError(
                f"the trees name a feature scheme this cannot read: {stated!r}"
            ) from error
        features = features_for(scheme, channels)
        if features.names() != fitted.feature_names:
            raise UnreadableFittedCandidateError(
                f"the trees were fitted over columns that are not the {scheme} reading of a "
                f"corpus of {channels} channels"
            )
        return features
