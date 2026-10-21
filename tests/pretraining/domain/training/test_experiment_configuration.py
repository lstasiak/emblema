import pytest

from emblema.pretraining.domain.exceptions import (
    InvalidCorpusShareError,
    InvalidExperimentConfigurationError,
)
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.corpus_fraction import CorpusFraction
from emblema.pretraining.domain.training.corpus_passes import CorpusPasses
from emblema.pretraining.domain.training.corpus_share import CorpusShare
from emblema.pretraining.domain.training.objective_loss import LossKind, ObjectiveLoss
from emblema.pretraining.domain.training.objective_target import ObjectiveTarget
from emblema.pretraining.domain.training.precision import Precision
from emblema.pretraining.domain.training.trained_parameters import TrainedParameters
from tests.support.experiments import (
    CONTRAST,
    LATENT,
    SUMMARY,
    WEIGHTS,
    budget,
    configuration,
)

BOUNDED = ObjectiveLoss(kind=LossKind.HUBER, huber_delta=1.0)


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
    assert configuration(corpus_fraction=0.5).parameters() != stated
    assert configuration(loss=BOUNDED).parameters() != stated


def test_rates_are_rendered_as_floats_however_they_were_built() -> None:
    # An integer zero and a float zero are one configuration, and everything that renders the
    # parameters — the tracker, the signature — must see one value.
    stated = configuration(dropout=0.0, budget=budget(learning_rate=1e-3))
    as_integers = configuration(dropout=0, budget=budget(learning_rate=1e-3))

    assert stated == as_integers
    assert stated.parameters() == as_integers.parameters()
    assert isinstance(as_integers.parameters()["dropout"], float)
    assert isinstance(as_integers.parameters()["warmup_epochs"], float)


def test_the_parameters_on_which_two_configurations_differ_are_named() -> None:
    stated = configuration()

    assert stated.differences_from(stated) == ()
    assert stated.differences_from(configuration(budget=budget(seed=2))) == ("seed",)
    assert stated.differences_from(configuration(dropout=0.1, decoder_layers=2)) == (
        "dropout",
        "decoder_layers",
    )


def test_the_reading_the_hidden_tokens_are_scored_by_is_reported_with_its_knee() -> None:
    squared = configuration().parameters()
    bounded = configuration(loss=BOUNDED).parameters()

    assert (squared["loss"], squared["huber_delta"]) == ("mse", 0.0)
    assert (bounded["loss"], bounded["huber_delta"]) == ("huber", 1.0)


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


def test_the_share_of_the_corpus_is_the_fraction_ranked_by_the_run_s_seed() -> None:
    stated = configuration(corpus_fraction=0.25, budget=budget(seed=5))

    assert stated.corpus_share == CorpusShare(fraction=0.25, seed=5)
    assert stated.parameters()["corpus_fraction"] == 0.25


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5, float("nan")])
def test_a_share_no_run_could_read_is_refused(fraction: float) -> None:
    with pytest.raises(InvalidCorpusShareError):
        configuration(corpus_fraction=fraction)


def test_a_corpus_is_read_once_unless_its_passes_are_stated() -> None:
    weighted = configuration(passes=(CorpusPasses(corpus="stays", passes=4),))

    assert configuration().passes_of("stays") == 1
    assert weighted.passes_of("stays") == 4
    assert weighted.passes_of("another") == 1


def test_passes_render_last_by_corpus_name_and_only_where_stated() -> None:
    weighted = configuration(
        passes=(CorpusPasses(corpus="engines", passes=2), CorpusPasses(corpus="stays", passes=4))
    )

    parameters = weighted.parameters()

    assert "passes.stays" not in configuration().parameters()
    assert list(parameters)[-2:] == ["passes.engines", "passes.stays"]
    assert (parameters["passes.engines"], parameters["passes.stays"]) == (2, 4)


@pytest.mark.parametrize("knob", ["passes", "fractions"])
def test_corpora_stated_out_of_the_order_of_their_names_are_refused(knob: str) -> None:
    # Equality reads the order, so a configuration listed in another order would be another
    # configuration: a database that keeps its keys in an order of its own would make a run
    # differ from the one ordered.
    stated = {
        "passes": (
            CorpusPasses(corpus="stays", passes=4),
            CorpusPasses(corpus="engines", passes=2),
        ),
        "fractions": (
            CorpusFraction(corpus="utsd/IoT_baian", fraction=0.25),
            CorpusFraction(corpus="utsd/ERA5_surface", fraction=0.25),
        ),
    }

    with pytest.raises(InvalidExperimentConfigurationError, match="order of the corpus names"):
        configuration(**{knob: stated[knob]})


def test_a_corpus_stated_more_than_once_is_refused() -> None:
    with pytest.raises(InvalidExperimentConfigurationError, match="twice"):
        configuration(
            passes=(CorpusPasses(corpus="stays", passes=4), CorpusPasses(corpus="stays", passes=2))
        )


def test_passes_stated_on_one_side_only_are_a_named_difference() -> None:
    weighted = configuration(passes=(CorpusPasses(corpus="stays", passes=4),))

    assert weighted.differences_from(configuration()) == ("passes.stays",)
    assert configuration().differences_from(weighted) == ("passes.stays",)
    assert weighted.differences_from(weighted) == ()


def test_a_corpus_is_read_at_the_mixture_s_share_unless_its_own_fraction_is_stated() -> None:
    mixed = configuration(
        corpus_fraction=0.5, fractions=(CorpusFraction(corpus="meters", fraction=0.1),)
    )

    assert mixed.corpus_share_of("meters") == CorpusShare(fraction=0.1, seed=mixed.budget.seed)
    assert mixed.corpus_share_of("stays") == mixed.corpus_share
    assert mixed.corpus_share == CorpusShare(fraction=0.5, seed=mixed.budget.seed)


def test_fractions_render_last_by_corpus_name_and_only_where_stated() -> None:
    mixed = configuration(
        passes=(CorpusPasses(corpus="stays", passes=4),),
        fractions=(
            CorpusFraction(corpus="engines", fraction=1),
            CorpusFraction(corpus="meters", fraction=0.1),
        ),
    )

    parameters = mixed.parameters()

    assert not any(key.startswith("fraction.") for key in configuration().parameters())
    assert list(parameters)[-3:] == ["passes.stays", "fraction.engines", "fraction.meters"]
    assert (parameters["fraction.engines"], parameters["fraction.meters"]) == (1.0, 0.1)
    assert isinstance(parameters["fraction.engines"], float)


def test_a_corpus_whose_fraction_is_stated_more_than_once_is_refused() -> None:
    with pytest.raises(InvalidExperimentConfigurationError, match="fraction stated twice"):
        configuration(
            fractions=(
                CorpusFraction(corpus="meters", fraction=0.1),
                CorpusFraction(corpus="meters", fraction=0.2),
            )
        )


def test_a_fraction_stated_on_one_side_only_is_a_named_difference() -> None:
    mixed = configuration(fractions=(CorpusFraction(corpus="meters", fraction=0.1),))

    assert mixed.differences_from(configuration()) == ("fraction.meters",)
    assert configuration().differences_from(mixed) == ("fraction.meters",)


def test_the_tail_is_rendered_beside_the_other_draws_and_only_where_it_is_drawn() -> None:
    forecasting = configuration(
        masking=MaskingStrategy(
            channel_rate=0.15,
            block_rate=0.0,
            block_span=0.5,
            token_rate=0.0,
            horizon_rate=1.0,
            horizon_min_span=0.15,
            horizon_max_span=0.5,
        )
    )

    parameters = list(forecasting.parameters())

    assert not any(key.startswith("horizon") for key in configuration().parameters())
    tail = parameters.index("token_rate") + 1
    assert parameters[tail : tail + 4] == [
        "horizon_rate",
        "horizon_min_span",
        "horizon_max_span",
        "expected_hidden_ratio",
    ]
    assert forecasting.parameters()["expected_hidden_ratio"] == pytest.approx(0.42625)
    assert "horizon_rate" in forecasting.differences_from(configuration())


def test_the_target_is_rendered_after_the_reading_and_only_where_it_is_a_state() -> None:
    latent = configuration(target=LATENT)

    parameters = list(latent.parameters())

    assert not any(
        key.startswith(("target", "momentum", "final_momentum"))
        for key in configuration().parameters()
    )
    knee = parameters.index("huber_delta") + 1
    assert parameters[knee : knee + 5] == [
        "target",
        "target_layers",
        "momentum",
        "final_momentum",
        "epochs",
    ]
    rendered = latent.parameters()
    assert (rendered["target"], rendered["target_layers"]) == ("latent", 1)
    assert (rendered["momentum"], rendered["final_momentum"]) == (0.996, 1.0)
    assert latent.differences_from(configuration()) == (
        "target",
        "target_layers",
        "momentum",
        "final_momentum",
    )


def test_a_target_reading_more_blocks_than_the_encoder_has_is_refused() -> None:
    with pytest.raises(InvalidExperimentConfigurationError, match="blocks of an encoder"):
        configuration(target=ObjectiveTarget.latent(layers=2))


def test_the_contrast_is_rendered_after_the_target_and_only_where_it_is_stated() -> None:
    contrasted = configuration(contrast=CONTRAST)

    parameters = list(contrasted.parameters())

    assert not any(key.startswith("contrast") for key in configuration().parameters())
    knee = parameters.index("huber_delta") + 1
    assert parameters[knee : knee + 3] == ["contrast_weight", "contrast_temperature", "epochs"]
    rendered = contrasted.parameters()
    assert (rendered["contrast_weight"], rendered["contrast_temperature"]) == (0.5, 0.2)
    assert contrasted.differences_from(configuration()) == (
        "contrast_weight",
        "contrast_temperature",
    )
    assert configuration().differences_from(contrasted) == (
        "contrast_weight",
        "contrast_temperature",
    )


def test_the_contrast_is_stated_beside_the_value_target_only() -> None:
    with pytest.raises(InvalidExperimentConfigurationError, match="value target only"):
        configuration(target=LATENT, contrast=CONTRAST)


def test_the_summaries_are_rendered_after_the_contrast_and_only_where_they_are_stated() -> None:
    summarised = configuration(summary=SUMMARY)

    parameters = list(summarised.parameters())

    assert "summary_weight" not in configuration().parameters()
    knee = parameters.index("huber_delta") + 1
    assert parameters[knee : knee + 2] == ["summary_weight", "epochs"]
    assert summarised.parameters()["summary_weight"] == 0.5
    assert summarised.differences_from(configuration()) == ("summary_weight",)


def test_the_summaries_are_stated_beside_the_value_target_only() -> None:
    with pytest.raises(InvalidExperimentConfigurationError, match="value target only"):
        configuration(target=LATENT, summary=SUMMARY)


def test_the_summaries_are_not_stated_beside_the_contrast() -> None:
    with pytest.raises(InvalidExperimentConfigurationError, match="not stated together"):
        configuration(contrast=CONTRAST, summary=SUMMARY)


def test_the_start_and_the_trained_parameters_render_only_where_they_are_stated() -> None:
    continued = configuration(start=WEIGHTS, trained=TrainedParameters.CHANNEL_ROWS)

    parameters = list(continued.parameters())

    assert not any(key in ("start", "trained") for key in configuration().parameters())
    after_the_checkpoint = parameters.index("checkpoint_every_steps") + 1
    assert parameters[after_the_checkpoint : after_the_checkpoint + 2] == ["start", "trained"]
    rendered = continued.parameters()
    assert rendered["start"] == str(WEIGHTS.checksum)
    assert rendered["trained"] == "channel_rows"
    assert continued.differences_from(configuration()) == ("start", "trained")
    assert configuration().differences_from(continued) == ("start", "trained")


def test_a_run_moving_everything_from_its_seed_renders_as_every_run_before_the_knobs() -> None:
    stated = configuration(start=None, trained=TrainedParameters.EVERYTHING)

    assert stated == configuration()
    assert stated.parameters() == configuration().parameters()


def test_the_rows_alone_may_be_moved_from_the_seed_as_well_as_from_stored_weights() -> None:
    # The control that tells an encoder's machinery from its rows: the same fit of the rows on
    # an encoder the seed drew.
    untrained = configuration(trained=TrainedParameters.CHANNEL_ROWS)

    assert untrained.start is None
    assert untrained.differences_from(configuration()) == ("trained",)
