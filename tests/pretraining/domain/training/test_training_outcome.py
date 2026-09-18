import pytest

from emblema.pretraining.domain.exceptions import InvalidTrainingOutcomeError
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.support.experiments import WEIGHTS, validated
from tests.support.experiments import epoch_outcome as epoch

OTHER = ArtifactRef("durable/other", Checksum.of_bytes(b"other"))


def test_a_run_reports_the_epochs_it_ran_and_the_weights_of_its_last() -> None:
    outcome = TrainingOutcome(backbone=WEIGHTS, epochs=(epoch(0), epoch(1, backbone=WEIGHTS)))

    assert outcome.backbone == WEIGHTS
    assert outcome.hidden_ratio == 0.46


def test_the_best_epoch_is_the_one_whose_weights_the_run_kept() -> None:
    kept = TrainingOutcome(
        backbone=WEIGHTS,
        epochs=(epoch(0, weights=OTHER), epoch(1, weights=WEIGHTS), epoch(2, backbone=WEIGHTS)),
    )

    assert kept.best_epoch == 1


def test_a_run_that_kept_inherited_weights_names_no_epoch_of_its_own() -> None:
    inherited = TrainingOutcome(backbone=WEIGHTS, epochs=(epoch(3), epoch(4, backbone=WEIGHTS)))

    assert inherited.best_epoch is None


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
        ("validation", ()),
        ("seconds", -1.0),
        ("hidden_ratio", 0.0),
        ("hidden_ratio", 1.5),
    ],
)
def test_an_epoch_that_measured_something_impossible_is_refused(field: str, value: float) -> None:
    with pytest.raises(InvalidTrainingOutcomeError):
        epoch(0, **{field: value})


def test_an_epoch_reports_every_corpus_of_its_held_out_side_once() -> None:
    both = epoch(0, validation=(validated(corpus="a"), validated(corpus="b", loss=0.4)))

    assert [scored.corpus for scored in both.validation] == ["a", "b"]
    with pytest.raises(InvalidTrainingOutcomeError, match="validated twice"):
        epoch(0, validation=(validated(corpus="a"), validated(corpus="a", loss=0.4)))


def test_the_loss_over_the_whole_side_weighs_a_corpus_by_the_tokens_it_gave() -> None:
    mixed = epoch(
        0,
        validation=(
            validated(corpus="small", tokens=10, loss=1.0),
            validated(corpus="large", tokens=90, loss=2.0),
        ),
    )

    assert mixed.validation_loss == pytest.approx((10 * 1.0 + 90 * 2.0) / 100)


def test_what_a_stopping_rule_reads_is_the_mean_over_corpora_and_not_over_tokens() -> None:
    """A corpus decides neither by the size of its values nor by the count of its windows."""
    mixed = epoch(
        0,
        validation=(
            validated(corpus="small", tokens=10, loss=1.0, trivial=1.0),
            validated(corpus="large", tokens=90, loss=2.0, trivial=8.0),
        ),
    )

    assert mixed.relative_validation == pytest.approx((1.0 + 0.25) / 2)
    assert mixed.validation_loss > 1.0
