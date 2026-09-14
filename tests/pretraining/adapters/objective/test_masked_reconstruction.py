import pytest

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.pretraining.adapters.objective.masked_reconstruction import (  # noqa: E402
    MaskedReconstruction,
)
from emblema.pretraining.adapters.objective.reconstruction_decoder import (  # noqa: E402
    ReconstructionDecoder,
)
from emblema.pretraining.adapters.objective.reconstruction_loss import (  # noqa: E402
    ReconstructionLoss,
)
from emblema.pretraining.adapters.objective.token_masking import TokenMasking  # noqa: E402
from emblema.pretraining.adapters.objective.token_masks import TokenMasks  # noqa: E402
from emblema.pretraining.domain.masking_strategy import MaskingStrategy  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from tests.support.encoders import SMALL  # noqa: E402
from tests.support.token_tensors import VOCABULARY_SIZE, random_batch  # noqa: E402

pytestmark = pytest.mark.ml

# The encoder reduces attention in an order the visible set decides, so two runs over the same
# visible tokens agree to a reduction's precision rather than bit for bit.
TOLERANCE = {"atol": 1e-5, "rtol": 1e-4}


def objective(*, seed: int = 1, decoder_layers: int = 1) -> MaskedReconstruction:
    torch.manual_seed(seed)
    encoder = SetEncoder.for_vocabulary(SMALL, VOCABULARY_SIZE)
    return MaskedReconstruction(encoder, decoder_layers=decoder_layers).eval()


def masks_over(batch: TokenTensors, strategy: MaskingStrategy, *, seed: int = 1) -> TokenMasks:
    return TokenMasking(strategy).draw(batch, torch.Generator().manual_seed(seed))


def test_the_prediction_is_one_value_per_position(strategy: MaskingStrategy) -> None:
    batch = random_batch(3, 40, seed=2, padding=7)

    prediction = objective()(batch, masks_over(batch, strategy))

    assert prediction.shape == (3, 40)
    assert torch.isfinite(prediction).all()


def test_the_encoder_never_sees_the_value_of_a_hidden_token(strategy: MaskingStrategy) -> None:
    model = objective()
    batch = random_batch(2, 48, seed=3, padding=5)
    masks = masks_over(batch, strategy)
    other = random_batch(2, 48, seed=4)
    scrambled = TokenTensors(
        torch.where(masks.hidden.unsqueeze(-1), other.features, batch.features),
        batch.channel_ids,
        batch.timestamps,
        batch.timeless,
        batch.padding_mask,
    )

    torch.testing.assert_close(model(scrambled, masks), model(batch, masks), **TOLERANCE)


def test_the_prediction_for_a_hidden_token_depends_on_which_channel_and_when(
    strategy: MaskingStrategy,
) -> None:
    model = objective()
    batch = random_batch(2, 48, seed=5)
    masks = masks_over(batch, strategy)
    other_channels = TokenTensors(
        batch.features,
        torch.where(masks.hidden, (batch.channel_ids % VOCABULARY_SIZE) + 1, batch.channel_ids),
        batch.timestamps,
        batch.timeless,
        batch.padding_mask,
    )

    prediction = model(batch, masks)
    relabelled = model(other_channels, masks)

    assert not torch.allclose(prediction[masks.hidden], relabelled[masks.hidden], **TOLERANCE)


def test_every_parameter_of_encoder_and_decoder_receives_a_gradient(
    strategy: MaskingStrategy,
) -> None:
    model = objective().train()
    batch = random_batch(4, 40, seed=6, padding=3)
    masks = masks_over(batch, strategy)

    ReconstructionLoss()(model(batch, masks), batch, masks).backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_the_padding_row_of_the_channel_table_receives_none(strategy: MaskingStrategy) -> None:
    model = objective().train()
    batch = random_batch(2, 24, seed=7, padding=4)
    masks = masks_over(batch, strategy)

    ReconstructionLoss()(model(batch, masks), batch, masks).backward()

    table = model.encoder.get_parameter("channel_embedding.table.weight").grad
    assert table is not None
    assert not table[0].any()


def test_the_backbone_is_the_encoder_and_nothing_of_the_decoder() -> None:
    model = objective(decoder_layers=2)

    encoder = dict(model.encoder.named_parameters())
    assert len(encoder) == SMALL_PARAMETER_TENSORS
    assert sum(p.numel() for p in model.encoder.parameters()) == SMALL.parameter_count(
        VOCABULARY_SIZE
    )
    assert all(not name.startswith("encoder.") for name, _ in model.decoder.named_parameters())


SMALL_PARAMETER_TENSORS = len(
    dict(SetEncoder.for_vocabulary(SMALL, VOCABULARY_SIZE).named_parameters())
)


def test_a_decoder_without_a_block_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one block"):
        ReconstructionDecoder(SMALL, layers=0)


def test_the_same_seed_builds_the_same_objective(strategy: MaskingStrategy) -> None:
    batch = random_batch(1, 16, seed=8)
    masks = masks_over(batch, strategy)

    torch.testing.assert_close(
        objective(seed=3)(batch, masks), objective(seed=3)(batch, masks), rtol=0, atol=0
    )
