import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from emblema.entrypoints.restored_backbones import RestoredBackbones  # noqa: E402
from emblema.evaluation.adapters.detection.pretext_source import PretextSource  # noqa: E402
from emblema.evaluation.adapters.torch.backbone_factory import BackboneFactory  # noqa: E402
from emblema.evaluation.domain.detection.hidden_cells_choice import (  # noqa: E402
    HiddenCellsChoice,
)
from emblema.evaluation.domain.exceptions import (  # noqa: E402
    UnknownBackboneError,
    UnreadablePretextError,
)
from emblema.evaluation.domain.transfer.encoder_setting import (  # noqa: E402
    TimeEncoding,
    ValueEmbedding,
)
from emblema.evaluation.domain.transfer.encoder_shape import EncoderShape  # noqa: E402
from emblema.pretraining.adapters.encoder.fourier_time_encoding import (  # noqa: E402
    FourierTimeEncoding,
)
from emblema.pretraining.adapters.encoder.grown_channel_embedding import (  # noqa: E402
    GrownChannelEmbedding,
)
from emblema.pretraining.adapters.encoder.learned_channel_embedding import (  # noqa: E402
    LearnedChannelEmbedding,
)
from emblema.pretraining.adapters.encoder.learned_time_encoding import (  # noqa: E402
    LearnedTimeEncoding,
)
from emblema.pretraining.adapters.encoder.nonlinear_value_embedding import (  # noqa: E402
    NonlinearValueEmbedding,
)
from emblema.pretraining.adapters.encoder.self_attention import SelfAttention  # noqa: E402
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.pretraining.adapters.objective.masked_reconstruction import (  # noqa: E402
    MaskedReconstruction,
)
from emblema.pretraining.adapters.objective.objective_shape import ObjectiveShape  # noqa: E402
from emblema.pretraining.adapters.objective.pretext_reading import PretextReading  # noqa: E402
from emblema.pretraining.adapters.training.exceptions import (  # noqa: E402
    UnreadableTrainedModelError,
)
from emblema.pretraining.adapters.training.trained_model import TrainedModel  # noqa: E402
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore  # noqa: E402
from emblema.shared.kernel.artifacts import ArtifactRef  # noqa: E402
from emblema.shared.kernel.tokens import N_FEATURES  # noqa: E402
from tests.support.experiments import (  # noqa: E402
    CELLS,
    CHANNELS,
    OBSERVATION,
    TINY,
    configuration,
    windows,
)

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

    restored = backbones.pretrained(weights, vocabulary_size=CHANNELS, dropout=0.0)

    after, before = restored.state_dict(), encoder.state_dict()
    assert after.keys() == before.keys()
    assert all(torch.equal(after[name], before[name]) for name in before)
    assert not restored.training


def test_a_fresh_encoder_has_the_stored_shape_and_weights_of_its_own() -> None:
    store = InMemoryArtifactStore()
    weights, encoder = stored(store)
    backbones = RestoredBackbones(store, weights)

    torch.manual_seed(11)
    fresh = backbones.fresh(vocabulary_size=CHANNELS, dropout=0.0)

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
        RestoredBackbones(store, weights).pretrained(other, vocabulary_size=CHANNELS, dropout=0.0)


def test_an_artifact_that_is_not_a_trained_model_is_refused_when_the_process_is_built() -> None:
    store = InMemoryArtifactStore()

    with pytest.raises(UnreadableTrainedModelError):
        RestoredBackbones(store, store.put(b"not a model"))


def test_the_pretrained_encoder_grows_rows_for_channels_past_its_table() -> None:
    store = InMemoryArtifactStore()
    weights, encoder = stored(store)
    backbones = RestoredBackbones(store, weights)

    grown = backbones.pretrained(weights, vocabulary_size=CHANNELS + 3, dropout=0.0)

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

    larger = backbones.fresh(vocabulary_size=CHANNELS + 3, dropout=0.0)
    smaller = backbones.fresh(vocabulary_size=1, dropout=0.0)

    tables = [larger.get_submodule("channel_embedding"), smaller.get_submodule("channel_embedding")]
    assert all(isinstance(table, LearnedChannelEmbedding) for table in tables)
    assert [table.vocabulary_size for table in tables] == [CHANNELS + 3, CHANNELS]


def dropout_rates(encoder: nn.Module) -> set[float]:
    """The share every dropout layer and every attention of ``encoder`` drops."""
    return {m.p for m in encoder.modules() if isinstance(m, nn.Dropout)} | {
        m.dropout for m in encoder.modules() if isinstance(m, SelfAttention)
    }


def test_both_encoders_drop_what_they_are_asked_to_and_the_stored_weights_stay() -> None:
    store = InMemoryArtifactStore()
    weights, encoder = stored(store)
    backbones = RestoredBackbones(store, weights)

    pretrained = backbones.pretrained(weights, vocabulary_size=CHANNELS, dropout=0.2)
    fresh = backbones.fresh(vocabulary_size=CHANNELS, dropout=0.2)

    assert dropout_rates(pretrained) == dropout_rates(fresh) == {0.2}
    after, before = pretrained.state_dict(), encoder.state_dict()
    assert all(torch.equal(after[name], before[name]) for name in before)
    assert dropout_rates(backbones.pretrained(weights, vocabulary_size=CHANNELS, dropout=0.0)) == {
        0.0
    }


def test_a_fresh_encoder_takes_a_shape_and_a_value_embedding_of_its_own() -> None:
    store = InMemoryArtifactStore()
    weights, _ = stored(store)
    backbones = RestoredBackbones(store, weights)
    shape = EncoderShape(width=8, heads=2, layers=2, feedforward_width=16)

    fresh = backbones.fresh(
        vocabulary_size=CHANNELS,
        dropout=0.0,
        value_embedding=ValueEmbedding.NONLINEAR,
        shape=shape,
    )

    assert isinstance(fresh, SetEncoder)
    built = fresh.architecture
    assert (built.width, built.heads, built.layers, built.feedforward_width) == (8, 2, 2, 16)
    assert built.time_frequencies == TINY.time_frequencies
    assert isinstance(fresh.value_projection, NonlinearValueEmbedding)
    assert len(fresh.blocks) == 2


def test_a_fresh_encoder_takes_a_learnt_time_and_a_wider_feature_axis_of_its_own() -> None:
    store = InMemoryArtifactStore()
    weights, _ = stored(store)
    backbones = RestoredBackbones(store, weights)

    fresh = backbones.fresh(
        vocabulary_size=CHANNELS,
        dropout=0.0,
        time_encoding=TimeEncoding.LEARNED,
        input_features=9,
    )

    assert isinstance(fresh, SetEncoder)
    assert isinstance(fresh.time_encoding, LearnedTimeEncoding)
    assert isinstance(fresh.value_projection, nn.Linear)
    assert fresh.value_projection.in_features == 9
    assert fresh.channel_embedding.vocabulary_size == CHANNELS
    standard = backbones.fresh(vocabulary_size=CHANNELS, dropout=0.0)
    assert isinstance(standard.time_encoding, FourierTimeEncoding)
    assert isinstance(standard.value_projection, nn.Linear)
    assert standard.value_projection.in_features == N_FEATURES
    bent = backbones.fresh(
        vocabulary_size=CHANNELS,
        dropout=0.0,
        value_embedding=ValueEmbedding.NONLINEAR,
        input_features=9,
    )
    assert isinstance(bent.value_projection, NonlinearValueEmbedding)
    assert bent.value_projection.layers[0].in_features == 9


def test_the_bins_are_the_stored_architecture_s() -> None:
    from dataclasses import replace

    from emblema.pretraining.adapters.objective.objective_shape import ObjectiveShape
    from emblema.pretraining.domain.training.objective_target import ObjectiveTarget

    store = InMemoryArtifactStore()
    weights, _ = stored(store)
    assert RestoredBackbones(store, weights).bins == 0

    over_cells = configuration(architecture=replace(TINY, bins=6))
    torch.manual_seed(4)
    model = ObjectiveShape(
        architecture=over_cells.architecture,
        vocabulary_size=CHANNELS,
        decoder_layers=1,
        target=ObjectiveTarget.value(),
    ).build()
    cells = store.put(TrainedModel.of(over_cells, CHANNELS, model).to_bytes())
    backbones = RestoredBackbones(store, cells)

    assert backbones.bins == 6
    pretrained = backbones.pretrained(cells, vocabulary_size=CHANNELS, dropout=0.0)
    assert isinstance(pretrained, SetEncoder)
    assert isinstance(pretrained.value_projection, nn.Linear)
    assert pretrained.value_projection.in_features == 9


def test_a_fresh_encoder_s_input_part_is_the_layout_s_not_the_stored_model_s() -> None:
    from dataclasses import replace

    from emblema.pretraining.adapters.objective.objective_shape import ObjectiveShape
    from emblema.pretraining.domain.training.objective_target import ObjectiveTarget

    store = InMemoryArtifactStore()
    over_cells = configuration(architecture=replace(TINY, bins=6))
    torch.manual_seed(5)
    model = ObjectiveShape(
        architecture=over_cells.architecture,
        vocabulary_size=CHANNELS,
        decoder_layers=1,
        target=ObjectiveTarget.value(),
    ).build()
    backbones = RestoredBackbones(
        store, store.put(TrainedModel.of(over_cells, CHANNELS, model).to_bytes())
    )

    raw = backbones.fresh(vocabulary_size=CHANNELS, dropout=0.0)
    binned = backbones.fresh(vocabulary_size=CHANNELS, dropout=0.0, input_features=9, bins=4)

    assert isinstance(raw, SetEncoder)
    assert isinstance(binned, SetEncoder)
    assert raw.architecture.bins == 0
    assert raw.value_projection.in_features == N_FEATURES
    assert binned.architecture.bins == 4
    assert binned.value_projection.in_features == 9


def stored_over_cells(store: InMemoryArtifactStore) -> tuple[ArtifactRef, TrainedModel]:
    stated = configuration(architecture=CELLS, observation=OBSERVATION)
    torch.manual_seed(6)
    trained = TrainedModel.of(stated, CHANNELS, ObjectiveShape.of(stated, CHANNELS).build())
    return store.put(trained.to_bytes()), trained


def test_the_pretext_is_asked_over_the_draws_the_stored_objective_makes() -> None:
    store = InMemoryArtifactStore()
    weights, trained = stored_over_cells(store)
    # Annotated with the seam it implements, so the type checker holds it to that shape of call.
    source: PretextSource = RestoredBackbones(store, weights)
    read = windows(2, seed=8)

    masked = source.masked(weights, read, [4, 5], answered=True, device="cpu")
    expected = PretextReading.of(TrainedModel.read(trained.to_bytes())).drawn(
        read, [4, 5], answered=True
    )

    assert source.bins == CELLS.bins
    assert source.huber_delta == 0.0
    for got, wanted in zip(masked, expected, strict=True):
        assert np.array_equal(got.hidden, wanted.hidden)
        assert np.array_equal(got.full, wanted.full)
        assert got.answers is not None
        assert wanted.answers is not None
        assert np.array_equal(got.answers, wanted.answers)


def test_the_pretext_of_other_weights_is_refused() -> None:
    store = InMemoryArtifactStore()
    weights, _ = stored_over_cells(store)
    other, _ = stored(store)

    with pytest.raises(UnknownBackboneError, match="serves the backbone"):
        RestoredBackbones(store, weights).masked(
            other, windows(1, seed=8), [1], answered=False, device="cpu"
        )


def test_a_backbone_that_cannot_say_what_it_hid_is_refused_in_this_context_s_terms() -> None:
    store = InMemoryArtifactStore()
    weights, _ = stored(store)

    with pytest.raises(UnreadablePretextError, match="does not say what it hid"):
        RestoredBackbones(store, weights).masked(
            weights, windows(1, seed=8), [1], answered=False, device="cpu"
        )


def test_a_caller_s_error_is_not_passed_off_as_the_backbone_s() -> None:
    store = InMemoryArtifactStore()
    weights, _ = stored_over_cells(store)

    with pytest.raises(ValueError, match="outside") as raised:
        RestoredBackbones(store, weights).masked(
            weights, windows(1, seed=8), [2**64], answered=False, device="cpu"
        )
    assert not isinstance(raised.value, UnreadablePretextError)


def test_the_tail_asked_of_the_source_is_the_reading_s_tail() -> None:
    store = InMemoryArtifactStore()
    weights, trained = stored_over_cells(store)
    read = windows(2, seed=8)

    masked = RestoredBackbones(store, weights).masked(
        weights, read, [4, 5], answered=False, device="cpu", hidden=HiddenCellsChoice.TAIL
    )
    expected = PretextReading.of(TrainedModel.read(trained.to_bytes())).drawn(
        read, [4, 5], answered=False, tail=True
    )

    for got, wanted in zip(masked, expected, strict=True):
        assert np.array_equal(got.hidden, wanted.hidden)
        assert got.hidden.any()
