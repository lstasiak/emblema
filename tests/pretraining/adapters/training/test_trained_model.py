import pytest

from emblema.pretraining.adapters.training.exceptions import UnreadableTrainedModelError
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from tests.support.experiments import CHANNELS, configuration, windows

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.pretraining.adapters.objective.masked_reconstruction import (  # noqa: E402
    MaskedReconstruction,
)
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
