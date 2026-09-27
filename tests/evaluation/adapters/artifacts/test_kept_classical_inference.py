"""The service Evaluation publishes for its classical forms answers as the fits themselves do."""

from dataclasses import replace

import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")

from emblema.evaluation.adapters.artifacts.kept_classical_inference import (  # noqa: E402
    KeptClassicalInference,
)
from emblema.evaluation.adapters.minirocket.fitted_convolutions import (  # noqa: E402
    FittedConvolutions,
)
from emblema.evaluation.adapters.xgboost.fitted_baseline import FittedBaseline  # noqa: E402
from emblema.evaluation.contracts.exceptions import InvalidKeptRepresentationError  # noqa: E402
from emblema.evaluation.contracts.kept_representation import KeptRepresentation  # noqa: E402
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme  # noqa: E402
from emblema.evaluation.domain.labels.target_kind import TargetKind  # noqa: E402
from emblema.shared.kernel.artifacts import ArtifactRef  # noqa: E402
from emblema.shared.kernel.checksums import Checksum  # noqa: E402
from tests.evaluation.adapters.features.support import timed, window  # noqa: E402
from tests.evaluation.support import convolutions, recipe  # noqa: E402
from tests.serving.kept_trees import SCALE, fitted_trees, trees_answers, trees_windows  # noqa: E402
from tests.support.openmp import skip_if_torch_shares_the_process  # noqa: E402

CHANNELS = 3


@pytest.fixture(autouse=True)
def _one_openmp_runtime() -> None:
    skip_if_torch_shares_the_process()


def form(format: str, content: bytes) -> KeptRepresentation:
    return KeptRepresentation(
        format=format,
        artifact=ArtifactRef(key=f"durable/{format}", checksum=Checksum.of_bytes(content)),
        deviation=None,
    )


def fitted_convolutions() -> FittedConvolutions:
    times = np.sort(np.random.default_rng(5).uniform(0.0, 1.0, 40))
    windows = [
        window(
            timed(1, [(float(np.sin(2 * np.pi * cycles * t)), float(t)) for t in times]),
            timed(2, [(float(t), float(t)) for t in times[::3]]),
        )
        for cycles in (1.0, 2.0, 3.0) * 2
    ]
    targets = np.array((1.0, 2.0, 3.0) * 2) / 10.0
    return FittedConvolutions.fitted(
        recipe(method=convolutions()),
        convolutions(),
        2,
        16,
        windows,
        targets,
        SCALE,
        TargetKind.CONTINUOUS,
    )


def test_the_service_reads_the_two_classical_forms_and_nothing_else() -> None:
    service = KeptClassicalInference()

    assert service.reads(FittedBaseline.FORMAT)
    assert service.reads(FittedConvolutions.FORMAT)
    assert not service.reads("onnx")
    assert not service.reads("torch-state")


def test_trees_answer_through_the_service_as_they_answer_read_back_directly() -> None:
    fitted = fitted_trees()
    windows = trees_windows(3)

    answered = KeptClassicalInference().answer(
        form(FittedBaseline.FORMAT, fitted.to_bytes()),
        fitted.to_bytes(),
        windows,
        channels=CHANNELS,
    )

    assert list(answered) == pytest.approx(trees_answers(fitted, windows))


def test_a_loaded_candidate_answers_again_without_reading_its_bytes_again() -> None:
    fitted = fitted_trees()
    content = fitted.to_bytes()
    service = KeptClassicalInference()
    service.answer(form(FittedBaseline.FORMAT, content), content, trees_windows(1), channels=3)

    # Bytes that are not a document at all: read again, they would refuse.
    answered = service.answer(
        form(FittedBaseline.FORMAT, content), b"not a document", trees_windows(1), channels=3
    )

    assert len(answered) == 1


def test_convolutions_answer_through_the_service_as_they_answer_directly() -> None:
    fitted = fitted_convolutions()
    windows = [
        window(
            timed(1, [(0.3, 0.1), (0.6, 0.5), (0.2, 0.9)]),
            timed(2, [(0.1, 0.2), (0.4, 0.8)]),
        )
    ]

    answered = KeptClassicalInference().answer(
        form(FittedConvolutions.FORMAT, fitted.to_bytes()),
        fitted.to_bytes(),
        windows,
        channels=2,
    )

    assert list(answered) == pytest.approx(fitted.predict(windows, threads=1).tolist())


@pytest.mark.parametrize("scheme", [FeatureScheme.PER_CHANNEL, FeatureScheme.SPECTRAL])
def test_trees_fitted_over_columns_of_another_reading_are_refused(scheme: FeatureScheme) -> None:
    # Declared under one scheme yet fitted over the summarised columns: the document contradicts
    # itself, and answering it over that scheme's rows would feed the trees the wrong columns.
    fitted = fitted_trees(scheme)

    with pytest.raises(InvalidKeptRepresentationError, match=str(scheme)):
        KeptClassicalInference().answer(
            form(FittedBaseline.FORMAT, fitted.to_bytes()),
            fitted.to_bytes(),
            trees_windows(1),
            channels=CHANNELS,
        )


def test_trees_naming_a_reading_this_context_has_none_for_are_refused() -> None:
    fitted = replace(fitted_trees(), parameters={"features": "hand-picked"})

    with pytest.raises(InvalidKeptRepresentationError, match="hand-picked"):
        KeptClassicalInference().answer(
            form(FittedBaseline.FORMAT, fitted.to_bytes()),
            fitted.to_bytes(),
            trees_windows(1),
            channels=CHANNELS,
        )


def test_a_form_the_service_does_not_read_is_refused_by_name() -> None:
    with pytest.raises(InvalidKeptRepresentationError, match="onnx"):
        KeptClassicalInference().answer(
            form("onnx", b"graph"), b"graph", trees_windows(1), channels=3
        )


@pytest.mark.parametrize("format", [FittedBaseline.FORMAT, FittedConvolutions.FORMAT])
def test_bytes_that_are_not_the_form_they_are_said_to_be_are_refused(format: str) -> None:
    with pytest.raises(InvalidKeptRepresentationError, match="not one this reads"):
        KeptClassicalInference().answer(
            form(format, b"not a document"), b"not a document", trees_windows(1), channels=3
        )
