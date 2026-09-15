import pytest

from emblema.pretraining.domain.exceptions import InvalidExperimentConfigurationError
from emblema.pretraining.domain.training.precision import Precision
from tests.support.experiments import budget, configuration


def test_the_configuration_flattens_to_scalars_a_tracker_can_log() -> None:
    parameters = configuration().parameters()

    assert all(isinstance(value, str | int | float) for value in parameters.values())
    assert parameters["width"] == 16
    assert parameters["precision"] == "fp32"
    assert parameters["seed"] == 1


def test_two_configurations_differing_anywhere_differ_in_their_parameters() -> None:
    stated = configuration().parameters()

    assert configuration(precision=Precision.BF16).parameters() != stated
    assert configuration(budget=budget(seed=2)).parameters() != stated
    assert configuration(dropout=0.1).parameters() != stated


def test_the_expected_share_hidden_is_reported_beside_the_rates() -> None:
    parameters = configuration().parameters()

    assert parameters["expected_hidden_ratio"] == pytest.approx(0.4645, abs=5e-4)


def test_the_schedule_is_the_budget_s_over_the_epoch_it_is_given() -> None:
    stated = configuration(budget=budget(epochs=4, warmup_epochs=1))

    assert stated.schedule(5).total_steps == 20


@pytest.mark.parametrize(("field", "value"), [("name", " "), ("dropout", 1.0), ("dropout", -0.1)])
def test_a_configuration_no_run_could_follow_is_refused(field: str, value: object) -> None:
    with pytest.raises(InvalidExperimentConfigurationError):
        configuration(**{field: value})


def test_a_decoder_of_no_layers_is_refused() -> None:
    with pytest.raises(InvalidExperimentConfigurationError, match="decoder_layers"):
        configuration(decoder_layers=0)
