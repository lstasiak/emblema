import pytest

from emblema.pretraining.domain.exceptions import InvalidTrainingOutcomeError
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.support.experiments import WEIGHTS
from tests.support.experiments import epoch_outcome as epoch

OTHER = ArtifactRef("durable/other", Checksum.of_bytes(b"other"))


def test_a_run_reports_the_epochs_it_ran_and_the_weights_of_its_last() -> None:
    outcome = TrainingOutcome(backbone=WEIGHTS, epochs=(epoch(0), epoch(1, backbone=WEIGHTS)))

    assert outcome.backbone == WEIGHTS
    assert outcome.hidden_ratio == 0.46


def test_a_resumed_run_need_not_begin_at_the_first_epoch() -> None:
    outcome = TrainingOutcome(backbone=WEIGHTS, epochs=(epoch(3), epoch(4, backbone=WEIGHTS)))

    assert [reported.epoch for reported in outcome.epochs] == [3, 4]


def test_a_gap_between_epochs_is_refused() -> None:
    with pytest.raises(InvalidTrainingOutcomeError, match="consecutive"):
        TrainingOutcome(backbone=WEIGHTS, epochs=(epoch(0), epoch(2, backbone=WEIGHTS)))


def test_weights_reported_anywhere_but_the_last_epoch_are_refused() -> None:
    with pytest.raises(InvalidTrainingOutcomeError, match="last epoch alone"):
        TrainingOutcome(
            backbone=WEIGHTS, epochs=(epoch(0, backbone=WEIGHTS), epoch(1, backbone=WEIGHTS))
        )


def test_a_run_whose_weights_are_not_its_last_epoch_s_is_refused() -> None:
    with pytest.raises(InvalidTrainingOutcomeError, match="same artifact"):
        TrainingOutcome(backbone=OTHER, epochs=(epoch(0), epoch(1, backbone=WEIGHTS)))


def test_a_run_that_trained_nothing_is_refused() -> None:
    with pytest.raises(InvalidTrainingOutcomeError, match="trained an epoch"):
        TrainingOutcome(backbone=WEIGHTS, epochs=())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epoch", -1),
        ("training_loss", -0.1),
        ("validation_loss", float("nan")),
        ("seconds", -1.0),
        ("hidden_ratio", 0.0),
        ("hidden_ratio", 1.5),
    ],
)
def test_an_epoch_that_measured_something_impossible_is_refused(field: str, value: float) -> None:
    with pytest.raises(InvalidTrainingOutcomeError):
        epoch(0, **{field: value})
