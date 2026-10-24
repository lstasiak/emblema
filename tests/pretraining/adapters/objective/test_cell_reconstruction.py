"""The objective over cells: what it hides, what it scores and what it leaks nothing of."""

import copy
import math
from collections.abc import Sequence
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.pretraining.adapters.objective.cell_reconstruction import (  # noqa: E402
    COUNT,
    PRESENCE,
    CellReconstruction,
)
from emblema.pretraining.adapters.objective.cell_tensors import CellTensors  # noqa: E402
from emblema.pretraining.adapters.objective.objective_score import RECONSTRUCTION  # noqa: E402
from emblema.pretraining.adapters.objective.token_masking import TokenMasking  # noqa: E402
from emblema.pretraining.adapters.objective.token_masks import TokenMasks  # noqa: E402
from emblema.pretraining.adapters.training.cell_batches import CellBatches  # noqa: E402
from emblema.pretraining.domain.masking_strategy import MaskingStrategy  # noqa: E402
from emblema.pretraining.domain.training.objective_loss import (  # noqa: E402
    LossKind,
    ObjectiveLoss,
)
from emblema.pretraining.domain.training.observation_term import ObservationTerm  # noqa: E402
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from emblema.shared.kernel.tokens import Token, TokenWindow  # noqa: E402
from tests.support.encoders import SMALL  # noqa: E402
from tests.support.experiments import irregular_windows  # noqa: E402

pytestmark = pytest.mark.ml

SQUARED = ObjectiveLoss(kind=LossKind.MSE)
BINS = 4
CELLS = replace(SMALL, bins=BINS)
VOCABULARY = 8
BOTH = ObservationTerm.of(presence_weight=1.0, count_weight=0.5)
LAYOUT = CellBatches(BinnedTokens(BINS))


def objective(
    *, seed: int = 1, observation: ObservationTerm = BOTH, decoder_layers: int = 1
) -> CellReconstruction:
    torch.manual_seed(seed)
    encoder = SetEncoder.for_vocabulary(
        CELLS, VOCABULARY, value_embedding=nn.Linear(CELLS.input_features, CELLS.width)
    )
    return CellReconstruction(
        encoder, decoder_layers=decoder_layers, observation=observation
    ).eval()


def laid(
    windows: Sequence[TokenWindow], strategy: MaskingStrategy, *, seed: int = 1
) -> tuple[CellTensors, TokenMasks]:
    collated = LAYOUT.collate(windows)
    masks = TokenMasking(strategy).draw(collated, torch.Generator().manual_seed(seed))
    return LAYOUT.scored(collated, masks), masks


def irregular(count: int, *, seed: int) -> list[TokenWindow]:
    return irregular_windows(count, seed=seed, channels=VOCABULARY - 1)


def masks_hiding(hidden: Tensor) -> TokenMasks:
    nothing = torch.zeros_like(hidden)
    return TokenMasks(channel=nothing, block=nothing, token=hidden, horizon=nothing)


def test_the_prediction_answers_seven_things_per_position(strategy: MaskingStrategy) -> None:
    batch, masks = laid(irregular(3, seed=2), strategy)

    prediction = objective()(batch, masks)

    assert prediction.shape == (3, batch.token_count, CellReconstruction.OUTPUTS)
    assert torch.isfinite(prediction).all()


def test_the_encoder_never_reads_a_hidden_cell_nor_the_full_layout(
    strategy: MaskingStrategy,
) -> None:
    model = objective()
    batch, masks = laid(irregular(2, seed=3), strategy)
    hidden = masks.hidden.unsqueeze(-1)

    with torch.no_grad():
        before = model(batch, masks)
        scrambled_hidden = replace(
            batch,
            visible=torch.where(hidden, torch.randn_like(batch.visible), batch.visible),
        )
        after_hidden = model(scrambled_hidden, masks)
        scrambled_full = replace(batch, features=torch.randn_like(batch.features))
        after_full = model(scrambled_full, masks)

    torch.testing.assert_close(after_hidden, before, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(after_full, before, atol=1e-5, rtol=1e-4)


def test_a_visible_cell_s_features_do_change_the_prediction(strategy: MaskingStrategy) -> None:
    model = objective()
    batch, masks = laid(irregular(2, seed=3), strategy)
    visible = ~masks.hidden & ~batch.padding_mask

    with torch.no_grad():
        before = model(batch, masks)
        bent = replace(
            batch,
            visible=torch.where(visible.unsqueeze(-1), batch.visible + 1.0, batch.visible),
        )
        after = model(bent, masks)

    assert not torch.allclose(after, before)


def test_the_score_counts_the_hidden_cells_and_scores_the_summaries_where_a_reading_was(
    strategy: MaskingStrategy,
) -> None:
    model = objective(observation=ObservationTerm.none())
    batch, masks = laid(irregular(4, seed=4), strategy)
    hidden = masks.hidden & ~batch.padding_mask
    held = hidden & (batch.features[..., 0] > 0.5)

    score = model.score(batch, masks, SQUARED)

    assert int(score.tokens) == int(hidden.sum())
    with torch.no_grad():
        prediction = model(batch, masks)[..., 2:7]
        per_cell = ((prediction - batch.features[..., 2:7]) ** 2).mean(-1)
        expected = float((per_cell * held).sum())
        trivial = float(((batch.features[..., 2:7] ** 2).mean(-1) * held).sum())
    assert float(score.error.detach()) == pytest.approx(expected, rel=1e-5)
    assert float(score.trivial) == pytest.approx(trivial, rel=1e-5)
    assert score.terms == {}


def test_hidden_cells_that_held_nothing_cost_no_reconstruction_but_a_presence() -> None:
    # Channel 1 is read in its first bin only; its last bin is empty, and is the one hidden.
    window = TokenWindow.of(
        [Token(1, 1.0, 0.1, 0.1), Token(2, 2.0, 0.1, 0.1), Token(2, 3.0, 0.9, 0.8)]
    )
    collated = LAYOUT.collate([window])
    hidden = torch.zeros_like(collated.padding_mask)
    hidden[0, collated.token_count - 2] = True
    masks = masks_hiding(hidden)
    batch = LAYOUT.scored(collated, masks)
    assert batch.channel_ids[0, collated.token_count - 2] == 1
    assert batch.features[0, collated.token_count - 2, 0] == 0.0

    score = objective().score(batch, masks, SQUARED)

    assert int(score.tokens) == 1
    assert float(score.terms[RECONSTRUCTION]) == 0.0
    assert float(score.terms[PRESENCE]) > 0.0
    assert float(score.terms[COUNT]) == 0.0


def test_the_trivial_presence_is_the_channel_s_visible_share_or_the_window_s() -> None:
    # Two channels over two bins, no statics: the cells stand c1b1, c2b1, c1b2, c2b2. Channel 1
    # is read in both bins, channel 2 in the first only.
    window = TokenWindow.of(
        [Token(1, 1.0, 0.2, 0.2), Token(2, 2.0, 0.2, 0.2), Token(1, 3.0, 0.7, 0.5)]
    )
    layout = CellBatches(BinnedTokens(2))
    collated = layout.collate([window])
    assert collated.channel_ids.tolist() == [[1, 2, 1, 2]]
    model = objective(observation=ObservationTerm.of(presence_weight=1.0))

    # Hide c1b2 (held a reading): channel 1's one visible cell held one, which with one imagined
    # cell of either kind is a share of two thirds.
    masks = masks_hiding(torch.tensor([[False, False, True, False]]))
    score = model.score(layout.scored(collated, masks), masks, SQUARED)
    assert float(score.trivial_terms[PRESENCE]) == pytest.approx(-math.log(2 / 3), rel=1e-4)

    # Hide c2b2 (empty): channel 2's visible cell held one, so two thirds against a zero.
    masks = masks_hiding(torch.tensor([[False, False, False, True]]))
    score = model.score(layout.scored(collated, masks), masks, SQUARED)
    assert float(score.trivial_terms[PRESENCE]) == pytest.approx(-math.log(1 / 3), rel=1e-4)

    # Hide both cells of channel 2: no visible cell of its own, so the window's share, which is
    # channel 1's two cells both held, three quarters smoothed, against a one and a zero.
    masks = masks_hiding(torch.tensor([[False, True, False, True]]))
    score = model.score(layout.scored(collated, masks), masks, SQUARED)
    assert float(score.trivial_terms[PRESENCE]) == pytest.approx(
        -math.log(3 / 4) - math.log(1 / 4), rel=1e-4
    )


def test_the_terms_add_up_to_the_error_and_name_the_hidden_cells_own(
    strategy: MaskingStrategy,
) -> None:
    batch, masks = laid(irregular(3, seed=5), strategy)

    score = objective().score(batch, masks, SQUARED)

    assert set(score.terms) == {RECONSTRUCTION, PRESENCE, COUNT}
    assert float(score.error.detach()) == pytest.approx(
        sum(float(term) for term in score.terms.values()), rel=1e-5
    )
    assert float(score.trivial) == pytest.approx(
        sum(float(term) for term in score.trivial_terms.values()), rel=1e-5
    )
    reconstruction, trivial = score.reconstruction
    assert float(reconstruction) == float(score.terms[RECONSTRUCTION])
    assert float(trivial) == float(score.trivial_terms[RECONSTRUCTION])


def test_the_weights_scale_the_terms_they_name(strategy: MaskingStrategy) -> None:
    batch, masks = laid(irregular(3, seed=5), strategy)
    unit = objective(observation=ObservationTerm.of(presence_weight=1.0, count_weight=1.0))
    doubled = objective(observation=ObservationTerm.of(presence_weight=2.0, count_weight=0.5))

    one, two = unit.score(batch, masks, SQUARED), doubled.score(batch, masks, SQUARED)

    assert float(two.terms[PRESENCE]) == pytest.approx(2 * float(one.terms[PRESENCE]), rel=1e-5)
    assert float(two.terms[COUNT]) == pytest.approx(0.5 * float(one.terms[COUNT]), rel=1e-5)
    assert float(two.terms[RECONSTRUCTION]) == pytest.approx(
        float(one.terms[RECONSTRUCTION]), rel=1e-5
    )


def test_every_parameter_of_encoder_and_decoder_receives_a_gradient(
    strategy: MaskingStrategy,
) -> None:
    model = objective().train()
    batch, masks = laid(irregular(4, seed=6), strategy)

    score = model.score(batch, masks, SQUARED)
    (score.error / score.tokens).backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_the_backbone_is_the_encoder_and_nothing_of_the_decoder() -> None:
    model = objective()

    assert {name.split(".")[0] for name, _ in model.named_parameters()} == {"encoder", "decoder"}
    assert model.encoder.architecture == CELLS
    assert model.decoder.outputs == CellReconstruction.OUTPUTS
    assert model.term == BOTH


def test_a_batch_without_the_visible_layout_is_refused(strategy: MaskingStrategy) -> None:
    collated = LAYOUT.collate(irregular(2, seed=7))
    masks = TokenMasking(strategy).draw(collated, torch.Generator().manual_seed(1))
    plain = TokenTensors(*collated.args)

    with pytest.raises(ValueError, match="one layout"):
        objective().score(plain, masks, SQUARED)


def test_an_encoder_over_readings_is_refused() -> None:
    with pytest.raises(ValueError, match="one token per reading"):
        CellReconstruction(
            SetEncoder.for_vocabulary(SMALL, VOCABULARY), decoder_layers=1, observation=BOTH
        )


def test_the_same_seed_builds_the_same_objective(strategy: MaskingStrategy) -> None:
    batch, masks = laid(irregular(2, seed=8), strategy)

    first = objective(seed=3).score(batch, masks, SQUARED)
    second = objective(seed=3).score(batch, masks, SQUARED)

    torch.testing.assert_close(first.error, second.error, rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple-silicon MPS")
def test_the_objective_scores_the_same_on_the_accelerator_as_on_the_host(
    strategy: MaskingStrategy,
) -> None:
    model = objective()
    batch, masks = laid(irregular(4, seed=11), strategy)
    on_host = model.score(batch, masks, SQUARED)

    moved = copy.deepcopy(model).to("mps")
    on_device = moved.score(batch.to("mps"), masks.to("mps"), SQUARED)

    # Off the accelerator first, then widened: the other order reads zeros.
    for name in (RECONSTRUCTION, PRESENCE, COUNT):
        for host, device in (
            (on_host.terms[name], on_device.terms[name]),
            (on_host.trivial_terms[name], on_device.trivial_terms[name]),
        ):
            torch.testing.assert_close(
                device.detach().to("cpu").to(torch.float64),
                host.detach().to(torch.float64),
                rtol=1e-3,
                atol=1e-3,
            )
