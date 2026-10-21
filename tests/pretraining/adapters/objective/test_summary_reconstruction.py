import copy

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.pretraining.adapters.objective.masked_reconstruction import (  # noqa: E402
    MaskedReconstruction,
)
from emblema.pretraining.adapters.objective.objective_score import RECONSTRUCTION  # noqa: E402
from emblema.pretraining.adapters.objective.summary_loss import (  # noqa: E402
    SummaryLoss,
    SummaryUnits,
)
from emblema.pretraining.adapters.objective.summary_reconstruction import (  # noqa: E402
    SUMMARY,
    SummaryReconstruction,
)
from emblema.pretraining.adapters.objective.token_masking import TokenMasking  # noqa: E402
from emblema.pretraining.adapters.objective.token_masks import TokenMasks  # noqa: E402
from emblema.pretraining.domain.masking_strategy import MaskingStrategy  # noqa: E402
from emblema.pretraining.domain.training.objective_loss import (  # noqa: E402
    LossKind,
    ObjectiveLoss,
)
from emblema.pretraining.domain.training.summary_term import SummaryTerm  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from tests.support.encoders import SMALL  # noqa: E402
from tests.support.token_tensors import VOCABULARY_SIZE, random_batch  # noqa: E402

pytestmark = pytest.mark.ml

SQUARED = ObjectiveLoss(kind=LossKind.MSE)
TERM = SummaryTerm.of(weight=0.5)
TOLERANCE = {"atol": 1e-5, "rtol": 1e-4}


def objective(*, seed: int = 1, term: SummaryTerm = TERM) -> SummaryReconstruction:
    torch.manual_seed(seed)
    encoder = SetEncoder.for_vocabulary(SMALL, VOCABULARY_SIZE)
    return SummaryReconstruction(encoder, decoder_layers=1, summary=term).eval()


def masks_over(batch: TokenTensors, strategy: MaskingStrategy, *, seed: int = 1) -> TokenMasks:
    return TokenMasking(strategy).draw(batch, torch.Generator().manual_seed(seed))


def masks_hiding(hidden: Tensor) -> TokenMasks:
    return TokenMasks(
        channel=torch.zeros_like(hidden),
        block=hidden,
        token=torch.zeros_like(hidden),
        horizon=torch.zeros_like(hidden),
    )


def by_hand(batch: TokenTensors, hidden: Tensor) -> dict[tuple[int, int], list[float]]:
    """Each window and channel's mean, minimum and maximum of its hidden timed values."""
    found: dict[tuple[int, int], list[float]] = {}
    for row in range(batch.batch_size):
        chosen = hidden[row] & ~batch.padding_mask[row] & ~batch.timeless[row]
        for channel in batch.channel_ids[row, chosen].unique().tolist():
            values = batch.features[row, chosen & (batch.channel_ids[row] == channel), 0]
            found[(row, channel)] = [
                float(values.mean()),
                float(values.min()),
                float(values.max()),
            ]
    return found


def test_each_unit_is_one_channel_of_one_window_with_its_hidden_values_summarised(
    strategy: MaskingStrategy,
) -> None:
    batch = random_batch(3, 40, seed=2, padding=5)
    masks = masks_over(batch, strategy)
    units = SummaryUnits.of(batch, masks.hidden)
    expected = by_hand(batch, masks.hidden)

    summaries = units.summaries(batch.features[..., 0])

    assert units.count == len(expected)
    for (row, channel), wanted in expected.items():
        tokens = (batch.channel_ids[row] == channel) & (units.unit[row] >= 0)
        [unit] = units.unit[row, tokens].unique().tolist()
        torch.testing.assert_close(summaries[unit], torch.tensor(wanted), **TOLERANCE)


def test_padding_timeless_and_visible_tokens_join_no_unit(strategy: MaskingStrategy) -> None:
    batch = random_batch(2, 30, seed=3, padding=4)
    masks = masks_over(batch, strategy)

    units = SummaryUnits.of(batch, masks.hidden)

    joined = units.unit >= 0
    assert not (joined & batch.padding_mask).any()
    assert not (joined & batch.timeless).any()
    assert not (joined & ~masks.hidden).any()


def test_a_unit_weighs_one_however_many_tokens_it_hid() -> None:
    batch = random_batch(1, 12, seed=4)
    channels = torch.tensor([[1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]])
    batch = TokenTensors(
        batch.features,
        channels,
        batch.timestamps,
        torch.zeros_like(batch.timeless),
        batch.padding_mask,
    )
    hidden = torch.ones(1, 12, dtype=torch.bool)
    units = SummaryUnits.of(batch, hidden)
    target = units.summaries(batch.features[..., 0])
    answered = target.clone()
    answered[0] += 1.0

    summed, scored = SummaryLoss(SQUARED).summed(answered, batch, units)

    assert int(scored) == 3
    torch.testing.assert_close(summed, torch.tensor(1.0), **TOLERANCE)


def test_the_trivial_answer_is_the_channel_s_mean_for_every_summary(
    strategy: MaskingStrategy,
) -> None:
    batch = random_batch(3, 40, seed=5, padding=3)
    units = SummaryUnits.of(batch, masks_over(batch, strategy).hidden)
    target = units.summaries(batch.features[..., 0])

    trivial = SummaryLoss(SQUARED).summed_trivial(batch, units)

    torch.testing.assert_close(trivial, (target**2).mean(dim=-1).sum(), **TOLERANCE)


def test_the_prediction_is_the_reconstruction_s_and_the_decoder_is_drawn_as_without_the_term(
    strategy: MaskingStrategy,
) -> None:
    batch = random_batch(3, 40, seed=2, padding=7)
    masks = masks_over(batch, strategy)
    torch.manual_seed(1)
    plain = MaskedReconstruction(
        SetEncoder.for_vocabulary(SMALL, VOCABULARY_SIZE), decoder_layers=1
    ).eval()

    summarised = objective(seed=1)

    torch.testing.assert_close(summarised(batch, masks), plain(batch, masks), rtol=0, atol=0)
    for (name, mine), (_, theirs) in zip(
        [(n, p) for n, p in summarised.named_parameters() if not n.startswith("summary_head.")],
        plain.named_parameters(),
        strict=True,
    ):
        assert torch.equal(mine, theirs), name


def test_the_score_is_the_reconstruction_plus_the_weighed_summaries_over_the_same_tokens(
    strategy: MaskingStrategy,
) -> None:
    model = objective()
    batch = random_batch(4, 40, seed=6, padding=3)
    masks = masks_over(batch, strategy)
    torch.manual_seed(1)
    plain = MaskedReconstruction(
        SetEncoder.for_vocabulary(SMALL, VOCABULARY_SIZE), decoder_layers=1
    ).eval()
    reconstruction = plain.score(batch, masks, SQUARED)
    answered, units = model.summarised(batch, masks)
    summarising = SummaryLoss(SQUARED)
    summary_sum, scored = summarising.summed(answered, batch, units)
    trivial = summarising.summed_trivial(batch, units)
    scale = TERM.weight * float(reconstruction.tokens) / float(scored)

    score = model.score(batch, masks, SQUARED)

    assert int(score.tokens) == int(reconstruction.tokens)
    torch.testing.assert_close(score.terms[RECONSTRUCTION], reconstruction.error, **TOLERANCE)
    torch.testing.assert_close(score.terms[SUMMARY], summary_sum * scale, **TOLERANCE)
    torch.testing.assert_close(score.error, reconstruction.error + summary_sum * scale, **TOLERANCE)
    torch.testing.assert_close(score.trivial_terms[SUMMARY], trivial * scale, **TOLERANCE)
    torch.testing.assert_close(score.trivial, reconstruction.trivial + trivial * scale, **TOLERANCE)
    # Divided by the tokens, the loss is the mean reconstruction plus the weight times the mean
    # summary loss per unit.
    torch.testing.assert_close(
        score.error / score.tokens,
        reconstruction.error / reconstruction.tokens + TERM.weight * summary_sum / scored,
        **TOLERANCE,
    )
    assert not score.trivial.requires_grad
    assert all(not term.requires_grad for term in score.terms.values())


def test_a_batch_the_masks_hid_nothing_of_costs_nothing() -> None:
    model = objective()
    batch = random_batch(2, 20, seed=7)
    masks = masks_hiding(torch.zeros(2, 20, dtype=torch.bool))

    score = model.score(batch, masks, SQUARED)

    assert float(score.error.detach()) == 0.0
    assert float(score.trivial) == 0.0


def test_every_parameter_of_encoder_decoder_and_head_receives_a_gradient(
    strategy: MaskingStrategy,
) -> None:
    model = objective().train()
    batch = random_batch(4, 40, seed=6, padding=3)
    masks = masks_over(batch, strategy)

    score = model.score(batch, masks, SQUARED)
    (score.error / score.tokens).backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_the_summaries_alone_reach_the_encoder_but_not_the_value_head(
    strategy: MaskingStrategy,
) -> None:
    model = objective().train()
    batch = random_batch(4, 40, seed=9, padding=3)
    masks = masks_over(batch, strategy)
    answered, units = model.summarised(batch, masks)

    summed, _ = SummaryLoss(SQUARED).summed(answered, batch, units)
    summed.backward()

    for name, parameter in model.named_parameters():
        if name.startswith(("decoder.head.", "decoder.mask_token")):
            continue
        assert parameter.grad is not None, name
    assert model.decoder.head.weight.grad is None


def test_the_backbone_is_the_encoder_and_nothing_of_the_decoder_or_the_head() -> None:
    model = objective()

    assert {name.split(".")[0] for name, _ in model.named_parameters()} == {
        "encoder",
        "decoder",
        "summary_head",
    }
    assert model.term == TERM


def test_a_loss_needs_a_stated_term() -> None:
    with pytest.raises(ValueError, match="needs a term"):
        objective(term=SummaryTerm.none())


def test_the_same_seed_builds_the_same_objective(strategy: MaskingStrategy) -> None:
    batch = random_batch(2, 16, seed=8)
    masks = masks_over(batch, strategy)

    first, second = (
        objective(seed=3).score(batch, masks, SQUARED),
        objective(seed=3).score(batch, masks, SQUARED),
    )

    torch.testing.assert_close(first.error, second.error, rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple-silicon MPS")
def test_the_objective_scores_the_same_on_the_accelerator_as_on_the_host(
    strategy: MaskingStrategy,
) -> None:
    model = objective()
    batch = random_batch(4, 32, seed=11, padding=3)
    masks = masks_over(batch, strategy)
    on_host = model.score(batch, masks, SQUARED)

    moved = copy.deepcopy(model).to("mps")
    on_device = moved.score(batch.to("mps"), masks.to("mps"), SQUARED)

    # Off the accelerator first, then widened: the other order reads zeros.
    for host, device in (
        (on_host.error, on_device.error),
        (on_host.trivial, on_device.trivial),
        (on_host.terms[SUMMARY], on_device.terms[SUMMARY]),
    ):
        torch.testing.assert_close(
            device.detach().to("cpu").to(torch.float64),
            host.detach().to(torch.float64),
            rtol=1e-3,
            atol=1e-3,
        )
