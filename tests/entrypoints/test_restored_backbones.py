import pytest

torch = pytest.importorskip("torch")

from emblema.entrypoints.restored_backbones import RestoredBackbones  # noqa: E402
from emblema.evaluation.adapters.torch.backbone_factory import BackboneFactory  # noqa: E402
from emblema.evaluation.domain.exceptions import UnknownBackboneError  # noqa: E402
from emblema.pretraining.adapters.encoder.grown_channel_embedding import (  # noqa: E402
    GrownChannelEmbedding,
)
from emblema.pretraining.adapters.encoder.learned_channel_embedding import (  # noqa: E402
    LearnedChannelEmbedding,
)
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.pretraining.adapters.objective.masked_reconstruction import (  # noqa: E402
    MaskedReconstruction,
)
from emblema.pretraining.adapters.training.exceptions import (  # noqa: E402
    UnreadableTrainedModelError,
)
from emblema.pretraining.adapters.training.trained_model import TrainedModel  # noqa: E402
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore  # noqa: E402
from emblema.shared.kernel.artifacts import ArtifactRef  # noqa: E402
from tests.support.experiments import CHANNELS, TINY, configuration  # noqa: E402

pytestmark = pytest.mark.ml


def stored(store: InMemoryArtifactStore, seed: int = 3) -> tuple[ArtifactRef, SetEncoder]:
    torch.manual_seed(seed)
    model = MaskedReconstruction(SetEncoder.for_vocabulary(TINY, CHANNELS), decoder_layers=1)
    trained = TrainedModel.of(configuration(), CHANNELS, model)
    return store.put(trained.to_bytes()), model.encoder


def test_the_pretrained_encoder_is_the_one_the_run_stored() -> None:
    store = InMemoryArtifactStore()
    weights, encoder = stored(store)
    # Annotated with the seam it implements, so the type checker holds it to that shape of call.
    backbones: BackboneFactory = RestoredBackbones(store, weights)

    restored = backbones.pretrained(weights, vocabulary_size=CHANNELS)

    after, before = restored.state_dict(), encoder.state_dict()
    assert after.keys() == before.keys()
    assert all(torch.equal(after[name], before[name]) for name in before)
    assert not restored.training


def test_a_fresh_encoder_has_the_stored_shape_and_weights_of_its_own() -> None:
    store = InMemoryArtifactStore()
    weights, encoder = stored(store)
    backbones = RestoredBackbones(store, weights)

    torch.manual_seed(11)
    fresh = backbones.fresh(vocabulary_size=CHANNELS)

    assert backbones.width == TINY.width
    assert fresh.state_dict().keys() == encoder.state_dict().keys()
    assert not torch.equal(
        fresh.state_dict()["value_projection.weight"],
        encoder.state_dict()["value_projection.weight"],
    )


def test_other_weights_than_the_process_was_built_over_are_refused() -> None:
    store = InMemoryArtifactStore()
    weights, _ = stored(store)
    other, _ = stored(store, seed=4)

    with pytest.raises(UnknownBackboneError, match="serves the backbone"):
        RestoredBackbones(store, weights).pretrained(other, vocabulary_size=CHANNELS)


def test_an_artifact_that_is_not_a_trained_model_is_refused_when_the_process_is_built() -> None:
    store = InMemoryArtifactStore()

    with pytest.raises(UnreadableTrainedModelError):
        RestoredBackbones(store, store.put(b"not a model"))


def test_the_pretrained_encoder_grows_rows_for_channels_past_its_table() -> None:
    store = InMemoryArtifactStore()
    weights, encoder = stored(store)
    backbones = RestoredBackbones(store, weights)

    grown = backbones.pretrained(weights, vocabulary_size=CHANNELS + 3)

    table = grown.get_submodule("channel_embedding")
    assert isinstance(table, GrownChannelEmbedding)
    assert table.vocabulary_size == CHANNELS + 3
    assert torch.equal(
        table.learnt.table.weight, encoder.get_parameter("channel_embedding.table.weight")
    )
    assert table.grown.weight.shape == (3, TINY.width)
    assert not grown.training


def test_a_fresh_encoder_covers_the_larger_of_the_tasks_and_the_stored_vocabulary() -> None:
    store = InMemoryArtifactStore()
    weights, _ = stored(store)
    backbones = RestoredBackbones(store, weights)

    larger = backbones.fresh(vocabulary_size=CHANNELS + 3)
    smaller = backbones.fresh(vocabulary_size=1)

    tables = [larger.get_submodule("channel_embedding"), smaller.get_submodule("channel_embedding")]
    assert all(isinstance(table, LearnedChannelEmbedding) for table in tables)
    assert [table.vocabulary_size for table in tables] == [CHANNELS + 3, CHANNELS]
