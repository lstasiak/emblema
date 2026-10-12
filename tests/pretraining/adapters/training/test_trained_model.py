import io

import pytest

from emblema.pretraining.adapters.training.exceptions import UnreadableTrainedModelError
from emblema.pretraining.domain.training.objective_target import ObjectiveTarget
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from tests.support.experiments import CHANNELS, LATENT, configuration, windows

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.pretraining.adapters.objective.latent_prediction import (  # noqa: E402
    LatentPrediction,
)
from emblema.pretraining.adapters.objective.masked_reconstruction import (  # noqa: E402
    MaskedReconstruction,
)
from emblema.pretraining.adapters.objective.objective_shape import ObjectiveShape  # noqa: E402
from emblema.pretraining.adapters.objective.token_masking import TokenMasking  # noqa: E402
from emblema.pretraining.adapters.training.trained_model import TrainedModel  # noqa: E402

pytestmark = pytest.mark.ml


def objective() -> MaskedReconstruction:
    stated = configuration()
    torch.manual_seed(3)
    return MaskedReconstruction(
        SetEncoder.for_vocabulary(stated.architecture, CHANNELS), decoder_layers=1
    ).eval()


def test_a_model_read_back_predicts_what_the_model_stored_predicted() -> None:
    stated = configuration()
    trained = objective()
    batch = TokenTensors.from_windows(windows(2, seed=9))
    masks = TokenMasking(stated.masking).draw(batch, torch.Generator().manual_seed(4))

    read = TrainedModel.read(TrainedModel.of(stated, CHANNELS, trained).to_bytes()).build()

    with torch.no_grad():
        assert torch.equal(read(batch, masks), trained(batch, masks))


def test_the_shape_travels_with_the_weights() -> None:
    stated = configuration()

    read = TrainedModel.read(TrainedModel.of(stated, CHANNELS, objective()).to_bytes())

    assert read.architecture == stated.architecture
    assert (read.vocabulary_size, read.decoder_layers) == (CHANNELS, 1)


def test_bytes_that_are_not_a_model_are_refused() -> None:
    with pytest.raises(UnreadableTrainedModelError, match="not a trained model"):
        TrainedModel.read(b"weights, honestly")


def test_weights_that_do_not_fit_the_shape_stored_are_refused() -> None:
    stored = TrainedModel.of(configuration(), CHANNELS, objective())
    mismatched = TrainedModel(
        architecture=stored.architecture,
        vocabulary_size=CHANNELS + 5,
        decoder_layers=stored.decoder_layers,
        weights=stored.weights,
    )

    with pytest.raises(UnreadableTrainedModelError, match="do not fit"):
        mismatched.build()


def test_a_model_rebuilt_to_go_on_learning_drops_what_it_is_asked_to_and_predicts_the_same() -> (
    None
):
    # Dropout holds no weights and is off in evaluation mode, so the share asked for changes
    # what the model does while it learns and nothing it predicts.
    stated = configuration()
    trained = objective()
    batch = TokenTensors.from_windows(windows(2, seed=9))
    masks = TokenMasking(stated.masking).draw(batch, torch.Generator().manual_seed(4))

    read = TrainedModel.read(TrainedModel.of(stated, CHANNELS, trained).to_bytes()).build(
        dropout=0.3
    )

    rates = {module.p for module in read.modules() if isinstance(module, torch.nn.Dropout)}
    assert rates == {0.3}
    with torch.no_grad():
        assert torch.equal(read(batch, masks), trained(batch, masks))


def test_bytes_written_before_a_target_existed_build_the_objective_that_predicts_values() -> None:
    stated = configuration()
    trained = objective()
    buffer = io.BytesIO()
    stored = torch.load(
        io.BytesIO(TrainedModel.of(stated, CHANNELS, trained).to_bytes()), weights_only=True
    )
    for key in ("target", "target_layers", "momentum", "final_momentum"):
        del stored[key]
    torch.save(stored, buffer)

    read = TrainedModel.read(buffer.getvalue())

    assert read.target == ObjectiveTarget.value()
    assert isinstance(read.build(), MaskedReconstruction)


def test_a_model_that_predicts_states_is_read_back_with_its_teacher_and_its_target() -> None:
    stated = configuration(target=LATENT)
    torch.manual_seed(5)
    trained = ObjectiveShape.of(stated, CHANNELS).build().eval()
    batch = TokenTensors.from_windows(windows(2, seed=9))
    masks = TokenMasking(stated.masking).draw(batch, torch.Generator().manual_seed(4))

    stored = TrainedModel.read(TrainedModel.of(stated, CHANNELS, trained).to_bytes())
    read = stored.build()

    assert stored.target == LATENT
    assert isinstance(read, LatentPrediction)
    assert isinstance(trained, LatentPrediction)
    with torch.no_grad():
        assert torch.equal(read(batch, masks), trained(batch, masks))
        assert torch.equal(read.targets(batch), trained.targets(batch))
