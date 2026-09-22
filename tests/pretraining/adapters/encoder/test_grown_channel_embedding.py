import pytest

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.encoder.grown_channel_embedding import (  # noqa: E402
    GrownChannelEmbedding,
)
from emblema.pretraining.adapters.encoder.learned_channel_embedding import (  # noqa: E402
    LearnedChannelEmbedding,
)
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.pretraining.domain.exceptions import InvalidEncoderArchitectureError  # noqa: E402
from emblema.shared.adapters.tensors.grown_parameters import GrownParameters  # noqa: E402
from tests.support.encoders import SMALL  # noqa: E402
from tests.support.token_tensors import random_batch  # noqa: E402

pytestmark = pytest.mark.ml

LEARNT = 5
WIDTH = 4


def learnt() -> LearnedChannelEmbedding:
    torch.manual_seed(1)
    return LearnedChannelEmbedding(LEARNT, WIDTH)


def test_learnt_channels_keep_their_rows_and_new_channels_get_rows_of_their_own() -> None:
    table = learnt()
    before = table.table.weight.detach().clone()

    grown = GrownChannelEmbedding(table, LEARNT + 3)
    ids = torch.tensor([[0, 1, LEARNT, LEARNT + 1, LEARNT + 3]])
    out = grown(ids)

    assert grown.vocabulary_size == LEARNT + 3
    assert torch.equal(out[0, 0], torch.zeros(WIDTH))
    assert torch.equal(out[0, 1], before[1])
    assert torch.equal(out[0, 2], before[LEARNT])
    assert torch.equal(out[0, 3], grown.grown.weight[0])
    assert torch.equal(out[0, 4], grown.grown.weight[2])


def test_the_grown_rows_are_the_parameters_it_declares_new() -> None:
    grown = GrownChannelEmbedding(learnt(), LEARNT + 2)

    assert isinstance(grown, GrownParameters)
    assert list(grown.grown_parameters()) == [grown.grown.weight]


def test_gradient_reaches_a_grown_row_and_a_learnt_row_alike() -> None:
    grown = GrownChannelEmbedding(learnt(), LEARNT + 2)

    grown(torch.tensor([[1, LEARNT + 2]])).sum().backward()

    learnt_grad, grown_grad = grown.learnt.table.weight.grad, grown.grown.weight.grad
    assert learnt_grad is not None
    assert grown_grad is not None
    assert learnt_grad[1].abs().sum() > 0
    assert grown_grad[1].abs().sum() > 0
    assert grown_grad[0].abs().sum() == 0


def test_a_table_does_not_grow_to_a_vocabulary_it_already_covers() -> None:
    with pytest.raises(InvalidEncoderArchitectureError):
        GrownChannelEmbedding(learnt(), LEARNT)


def test_an_encoder_grown_to_a_larger_vocabulary_answers_the_old_channels_as_before() -> None:
    torch.manual_seed(2)
    encoder = SetEncoder.for_vocabulary(SMALL, 9).eval()
    batch = random_batch(2, 7, seed=3)
    batch.channel_ids.clamp_(max=9)
    before = encoder(*batch.args)

    same = encoder.grown_to(9)
    grown = encoder.grown_to(15)

    assert same is encoder
    assert grown is encoder
    assert isinstance(encoder.channel_embedding, GrownChannelEmbedding)
    assert torch.equal(encoder(*batch.args), before)


def test_an_encoder_grown_answers_the_new_channels_too() -> None:
    torch.manual_seed(2)
    encoder = SetEncoder.for_vocabulary(SMALL, 9).grown_to(15).eval()
    batch = random_batch(2, 7, seed=3)
    batch.channel_ids[:, :] = 15

    states = encoder(*batch.args)

    assert states.shape == (2, 7, SMALL.width)
    assert torch.isfinite(states).all()


def test_only_a_learnt_table_grows() -> None:
    encoder = SetEncoder.for_vocabulary(SMALL, 9)
    encoder.channel_embedding = torch.nn.Identity()

    with pytest.raises(InvalidEncoderArchitectureError):
        encoder.grown_to(15)
