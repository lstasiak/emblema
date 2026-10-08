import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402

from emblema.evaluation.adapters.torch.layer_readout import LayerReadout  # noqa: E402
from emblema.evaluation.domain.exceptions import InvalidLayerReadingError  # noqa: E402
from emblema.evaluation.domain.transfer.layer_reading import LayerReading  # noqa: E402
from tests.support.encoders import SMALL, small_encoder  # noqa: E402
from tests.support.token_tensors import random_batch  # noqa: E402

pytestmark = pytest.mark.ml


class StampedLayers(nn.Module):
    """Answers layer ``k`` of every token with the constant ``k``, so a reading shows its layers."""

    def __init__(self, blocks: int, width: int) -> None:
        super().__init__()
        self.blocks, self.width = blocks, width

    def layer_states(self, features: Tensor, *_: Tensor) -> Tensor:
        batch, tokens = features.shape[:2]
        stamps = torch.arange(self.blocks + 1, dtype=torch.float32)
        return stamps[:, None, None, None].expand(-1, batch, tokens, self.width).clone()


def read(reading: str, blocks: int = 3, width: int = 2) -> Tensor:
    readout = LayerReadout(StampedLayers(blocks, width), LayerReading.of(reading), blocks=blocks)
    batch = random_batch(2, 5, seed=1)
    states: Tensor = readout(*batch.args)
    return states[0, 0]


def test_a_layer_read_by_its_number_is_that_layer() -> None:
    assert read("0").tolist() == [0.0, 0.0]
    assert read("2").tolist() == [2.0, 2.0]
    assert read("3").tolist() == [3.0, 3.0]


def test_the_blocks_are_concatenated_in_their_order_and_the_embedding_left_out() -> None:
    assert read("concat").tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_the_mean_is_over_the_blocks_and_leaves_the_embedding_out() -> None:
    assert read("mean").tolist() == [2.0, 2.0]


def test_a_real_encoder_read_at_a_layer_answers_that_layers_states() -> None:
    encoder = small_encoder()
    batch = random_batch(2, 6, seed=3)

    readout = LayerReadout(encoder, LayerReading(layer=1), blocks=SMALL.layers)

    torch.testing.assert_close(readout(*batch.args), encoder.layer_states(*batch.args)[1])


def test_a_layer_above_the_last_block_is_refused() -> None:
    with pytest.raises(InvalidLayerReadingError, match="layer 4 is not one of its layers 0 to 3"):
        LayerReadout(StampedLayers(3, 2), LayerReading(layer=4), blocks=3)


def test_an_encoder_that_answers_with_its_last_layer_only_is_refused() -> None:
    with pytest.raises(InvalidLayerReadingError, match="last layer only"):
        LayerReadout(nn.Linear(2, 2), LayerReading(layer=1), blocks=3)
