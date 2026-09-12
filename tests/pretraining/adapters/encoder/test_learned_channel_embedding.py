import pytest

from emblema.shared.kernel.tokens import PADDING_CHANNEL_ID

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.encoder.learned_channel_embedding import (  # noqa: E402
    LearnedChannelEmbedding,
)
from tests.support.token_tensors import VOCABULARY_SIZE  # noqa: E402

pytestmark = pytest.mark.ml

WIDTH = 16


def test_the_padding_row_stays_at_zero_and_receives_no_gradient() -> None:
    torch.manual_seed(1)
    embedding = LearnedChannelEmbedding(VOCABULARY_SIZE, WIDTH)
    channel_ids = torch.tensor([[PADDING_CHANNEL_ID, 1, 2, PADDING_CHANNEL_ID]])

    embedded = embedding(channel_ids)
    embedded.sum().backward()

    gradient = embedding.table.weight.grad
    assert gradient is not None
    assert torch.equal(embedded[0, 0], torch.zeros(WIDTH))
    assert torch.equal(gradient[PADDING_CHANNEL_ID], torch.zeros(WIDTH))
    assert (gradient[1] != 0).all()


def test_the_table_has_a_row_per_channel_and_one_more_for_padding() -> None:
    embedding = LearnedChannelEmbedding(VOCABULARY_SIZE, WIDTH)

    assert embedding.table.weight.shape == (VOCABULARY_SIZE + 1, WIDTH)
