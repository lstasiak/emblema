import json

import pytest

from emblema.pretraining.adapters.documents.experiment_configuration_document import (
    ExperimentConfigurationDocument,
)
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.corpus_fraction import CorpusFraction
from emblema.pretraining.domain.training.corpus_passes import CorpusPasses
from emblema.pretraining.domain.training.objective_loss import LossKind, ObjectiveLoss
from emblema.pretraining.domain.training.precision import Precision
from emblema.pretraining.domain.training.trained_parameters import TrainedParameters
from tests.support.experiments import CONTRAST, LATENT, WEIGHTS, budget, configuration

CODEC = ExperimentConfigurationDocument()


def test_a_configuration_round_trips_through_its_document() -> None:
    stated = configuration(
        name="round-trip", precision=Precision.FP16, budget=budget(seed=9, warmup_epochs=1)
    )

    assert CODEC.decode(CODEC.encode(stated)) == stated


def test_the_reading_of_the_loss_travels_with_its_knee() -> None:
    bounded = configuration(loss=ObjectiveLoss(kind=LossKind.HUBER, huber_delta=1.5))

    document = CODEC.encode(bounded)

    assert document["objective"] == {"kind": "huber", "huber_delta": 1.5}
    assert CODEC.decode(document) == bounded


def test_the_document_is_plain_json_with_the_shape_stated_outright() -> None:
    document = CODEC.encode(configuration())

    assert json.loads(json.dumps(document)) == document
    assert document["architecture"] == {
        "width": 16,
        "heads": 2,
        "layers": 1,
        "feedforward_width": 32,
        "time_frequencies": 4,
    }


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        ({"tier": "XL"}, "XL"),
        ({"precision": "fp8"}, "fp8"),
        ({"dropout": 1.5}, "dropout"),
        ({"corpus_fraction": 0}, "fraction"),
        ({"budget": {}}, "epochs"),
        ({"objective": {"kind": "mae", "huber_delta": 0.0}}, "mae"),
        ({"objective": {"kind": "mse", "huber_delta": 1.0}}, "no knee"),
        ({"architecture": None}, "architecture"),
        ({"passes": {"control-a": 1}}, "at least 2"),
        ({"passes": {"control-a": "4"}}, "control-a"),
    ],
)
def test_a_document_that_states_no_configuration_is_refused(
    broken: dict[str, object], message: str
) -> None:
    document = CODEC.encode(configuration()) | broken

    with pytest.raises(ValueError, match=message):
        CODEC.decode(document)


def test_the_passes_of_a_corpus_travel_by_its_name() -> None:
    weighted = configuration(
        passes=(CorpusPasses(corpus="engines", passes=2), CorpusPasses(corpus="stays", passes=4))
    )

    document = CODEC.encode(weighted)

    assert document["passes"] == {"engines": 2, "stays": 4}
    assert CODEC.decode(document) == weighted


def test_a_document_written_before_passes_existed_reads_every_corpus_once() -> None:
    document = dict(CODEC.encode(configuration()))
    del document["passes"]

    assert CODEC.decode(document) == configuration()


def test_the_fraction_of_a_corpus_travels_by_its_name_and_only_where_stated() -> None:
    mixed = configuration(
        fractions=(
            CorpusFraction(corpus="engines", fraction=1.0),
            CorpusFraction(corpus="meters", fraction=0.1),
        )
    )

    document = CODEC.encode(mixed)

    assert document["fraction"] == {"engines": 1.0, "meters": 0.1}
    assert CODEC.decode(document) == mixed
    # A run at the mixture's share writes the document it wrote before the knob existed.
    assert "fraction" not in CODEC.encode(configuration())
    assert CODEC.decode(CODEC.encode(configuration())) == configuration()


def test_the_tail_travels_inside_the_masking_and_only_where_it_is_drawn() -> None:
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

    document = CODEC.encode(forecasting)

    masking = document["masking"]
    assert isinstance(masking, dict)
    assert masking["horizon"] == {"rate": 1.0, "min_span": 0.15, "max_span": 0.5}
    assert CODEC.decode(json.loads(json.dumps(document))) == forecasting
    plain = CODEC.encode(configuration())["masking"]
    assert isinstance(plain, dict)
    assert "horizon" not in plain


def test_a_document_listing_its_corpora_in_another_order_states_the_same_configuration() -> None:
    # JSONB keeps the keys of an object shortest first, so the registry hands back the corpora of
    # a configuration in an order the file that stated them never had.
    mixed = configuration(
        passes=(CorpusPasses(corpus="cmapss", passes=2), CorpusPasses(corpus="stays", passes=2)),
        fractions=(
            CorpusFraction(corpus="tep", fraction=0.25),
            CorpusFraction(corpus="utsd/ERA5_surface", fraction=0.25),
            CorpusFraction(corpus="utsd/IoT_baian", fraction=0.25),
        ),
    )
    document = dict(CODEC.encode(mixed))
    document["passes"] = {"stays": 2, "cmapss": 2}
    document["fraction"] = {"tep": 0.25, "utsd/IoT_baian": 0.25, "utsd/ERA5_surface": 0.25}

    assert CODEC.decode(document) == mixed


def test_the_target_travels_inside_the_objective_and_only_where_it_is_a_state() -> None:
    latent = configuration(target=LATENT)

    document = CODEC.encode(latent)

    assert document["objective"] == {
        "kind": "mse",
        "huber_delta": 0.0,
        "target": {"kind": "latent", "layers": 1, "momentum": 0.996, "final_momentum": 1.0},
    }
    assert CODEC.decode(json.loads(json.dumps(document))) == latent
    assert CODEC.encode(configuration())["objective"] == {"kind": "mse", "huber_delta": 0.0}


def test_a_document_written_before_a_target_existed_predicts_the_token_s_value() -> None:
    document = CODEC.encode(configuration())
    assert document["objective"] == {"kind": "mse", "huber_delta": 0.0}

    assert CODEC.decode(document) == configuration()


def test_the_contrast_travels_inside_the_objective_and_only_where_it_is_stated() -> None:
    contrasted = configuration(contrast=CONTRAST)

    document = CODEC.encode(contrasted)

    assert document["objective"] == {
        "kind": "mse",
        "huber_delta": 0.0,
        "contrast": {"weight": 0.5, "temperature": 0.2},
    }
    assert CODEC.decode(json.loads(json.dumps(document))) == contrasted
    assert CODEC.encode(configuration())["objective"] == {"kind": "mse", "huber_delta": 0.0}


def test_the_start_and_the_trained_parameters_travel_only_where_stated() -> None:
    continued = configuration(start=WEIGHTS, trained=TrainedParameters.CHANNEL_ROWS)

    document = CODEC.encode(continued)

    assert document["start"] == {
        "key": WEIGHTS.key,
        "checksum": {"algorithm": "sha256", "digest": WEIGHTS.checksum.digest},
    }
    assert document["trained"] == "channel_rows"
    assert CODEC.decode(json.loads(json.dumps(document))) == continued
    plain = CODEC.encode(configuration())
    assert "start" not in plain
    assert "trained" not in plain


def test_a_document_written_before_a_run_could_continue_a_backbone_starts_from_its_seed() -> None:
    document = CODEC.encode(configuration())

    assert CODEC.decode(document) == configuration()
    assert CODEC.decode(document).start is None
    assert CODEC.decode(document).trained is TrainedParameters.EVERYTHING


def test_a_document_naming_parameters_nobody_can_train_is_refused() -> None:
    document = CODEC.encode(configuration()) | {"trained": "the decoder"}

    with pytest.raises(ValueError, match="the decoder"):
        CODEC.decode(document)
