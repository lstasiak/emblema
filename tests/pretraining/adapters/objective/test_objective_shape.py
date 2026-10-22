import pytest

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.objective.contrastive_reconstruction import (  # noqa: E402
    ContrastiveReconstruction,
)
from emblema.pretraining.adapters.objective.latent_prediction import (  # noqa: E402
    LatentPrediction,
)
from emblema.pretraining.adapters.objective.masked_reconstruction import (  # noqa: E402
    MaskedReconstruction,
)
from emblema.pretraining.adapters.objective.objective_shape import ObjectiveShape  # noqa: E402
from emblema.pretraining.adapters.objective.summary_reconstruction import (  # noqa: E402
    SummaryReconstruction,
)
from emblema.pretraining.adapters.objective.token_masking import TokenMasking  # noqa: E402
from emblema.pretraining.domain.training.contrastive_term import ContrastiveTerm  # noqa: E402
from emblema.pretraining.domain.training.objective_target import ObjectiveTarget  # noqa: E402
from tests.support.encoders import SMALL  # noqa: E402
from tests.support.experiments import CONTRAST, LATENT, SUMMARY, configuration  # noqa: E402
from tests.support.token_tensors import VOCABULARY_SIZE, random_batch  # noqa: E402

pytestmark = pytest.mark.ml


NO_CONTRAST = ContrastiveTerm.none()


def shape(target: ObjectiveTarget, contrast: ContrastiveTerm = NO_CONTRAST) -> ObjectiveShape:
    return ObjectiveShape(
        architecture=SMALL,
        vocabulary_size=VOCABULARY_SIZE,
        decoder_layers=1,
        target=target,
        contrast=contrast,
    )


def test_the_value_target_builds_the_objective_that_predicts_values() -> None:
    model = shape(ObjectiveTarget.value()).build()

    assert type(model) is MaskedReconstruction
    assert model.encoder.architecture == SMALL


def test_a_stated_contrast_builds_the_objective_that_also_tells_the_halves_apart() -> None:
    model = shape(ObjectiveTarget.value(), CONTRAST).build()

    assert isinstance(model, ContrastiveReconstruction)
    assert model.term == CONTRAST


def test_a_contrast_beside_the_latent_target_is_refused() -> None:
    with pytest.raises(ValueError, match="value target only"):
        shape(ObjectiveTarget.latent(layers=1), CONTRAST).build()


def test_stated_summaries_build_the_objective_that_also_asks_for_them() -> None:
    model = ObjectiveShape(
        architecture=SMALL,
        vocabulary_size=VOCABULARY_SIZE,
        decoder_layers=1,
        target=ObjectiveTarget.value(),
        summary=SUMMARY,
    ).build()

    assert isinstance(model, SummaryReconstruction)
    assert model.term == SUMMARY


@pytest.mark.parametrize(
    ("target", "contrast", "refusal"),
    [
        (ObjectiveTarget.latent(layers=1), NO_CONTRAST, "value target only"),
        (ObjectiveTarget.value(), CONTRAST, "not stated together"),
    ],
    ids=["latent", "contrast"],
)
def test_summaries_beside_the_latent_target_or_the_contrast_are_refused(
    target: ObjectiveTarget, contrast: ContrastiveTerm, refusal: str
) -> None:
    built = ObjectiveShape(
        architecture=SMALL,
        vocabulary_size=VOCABULARY_SIZE,
        decoder_layers=1,
        target=target,
        contrast=contrast,
        summary=SUMMARY,
    )

    with pytest.raises(ValueError, match=refusal):
        built.build()


def test_the_latent_target_builds_the_objective_that_predicts_states() -> None:
    model = shape(ObjectiveTarget.latent(layers=2)).build()

    assert isinstance(model, LatentPrediction)
    assert model.target == ObjectiveTarget.latent(layers=2)


def test_a_target_reading_more_blocks_than_the_encoder_has_is_refused() -> None:
    with pytest.raises(ValueError, match="blocks of an encoder"):
        shape(ObjectiveTarget.latent(layers=SMALL.layers + 1)).build()


def test_the_shape_is_read_off_a_configuration() -> None:
    stated = configuration(target=LATENT, decoder_layers=2)
    contrasted = configuration(contrast=CONTRAST)
    summarised = configuration(summary=SUMMARY)

    assert ObjectiveShape.of(stated, 7) == ObjectiveShape(
        architecture=stated.architecture, vocabulary_size=7, decoder_layers=2, target=LATENT
    )
    assert ObjectiveShape.of(contrasted, 7) == ObjectiveShape(
        architecture=contrasted.architecture,
        vocabulary_size=7,
        decoder_layers=1,
        target=ObjectiveTarget.value(),
        contrast=CONTRAST,
    )
    assert ObjectiveShape.of(summarised, 7).summary == SUMMARY


@pytest.mark.parametrize(
    "built",
    [
        shape(ObjectiveTarget.value()),
        shape(ObjectiveTarget.latent(layers=1)),
        shape(ObjectiveTarget.value(), CONTRAST),
        ObjectiveShape(
            architecture=SMALL,
            vocabulary_size=VOCABULARY_SIZE,
            decoder_layers=1,
            target=ObjectiveTarget.value(),
            summary=SUMMARY,
        ),
    ],
    ids=["value", "latent", "contrast", "summaries"],
)
def test_the_same_seed_builds_the_same_objective_of_any_kind(built: ObjectiveShape) -> None:
    batch = random_batch(2, 16, seed=8)
    masks = TokenMasking(configuration().masking).draw(batch, torch.Generator().manual_seed(1))
    loss = configuration().loss

    torch.manual_seed(3)
    first = built.build().eval().score(batch, masks, loss)
    torch.manual_seed(3)
    second = built.build().eval().score(batch, masks, loss)

    torch.testing.assert_close(first.error, second.error, rtol=0, atol=0)
    torch.testing.assert_close(first.trivial, second.trivial, rtol=0, atol=0)
