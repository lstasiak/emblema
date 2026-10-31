"""How a run lays windows out for its masks and for its objective, over readings and over cells."""

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
from torch import Tensor  # noqa: E402

from emblema.pretraining.adapters.objective.cell_tensors import CellTensors  # noqa: E402
from emblema.pretraining.adapters.objective.token_masking import TokenMasking  # noqa: E402
from emblema.pretraining.adapters.objective.token_masks import TokenMasks  # noqa: E402
from emblema.pretraining.adapters.training.cell_batches import CellBatches  # noqa: E402
from emblema.pretraining.adapters.training.cell_windows import CellWindows  # noqa: E402
from emblema.pretraining.adapters.training.raw_batches import RawBatches  # noqa: E402
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from emblema.shared.kernel.tokens import Token, TokenWindow  # noqa: E402
from tests.support.experiments import MIXTURE, irregular_windows  # noqa: E402

pytestmark = pytest.mark.ml

LAYOUT = BinnedTokens(4)
# Channel 1 read in its first bin twice and in its last bin once; channel 2 in its second bin.
WINDOW = TokenWindow.of(
    [
        Token(1, 3.0, 0.10, 0.10),
        Token(1, 5.0, 0.20, 0.10),
        Token(2, -1.0, 0.30, 0.30),
        Token(1, 7.0, 0.90, 0.70),
    ]
)
SHORT = TokenWindow.of([Token(1, 0.5, 0.5, 0.5)])
PRESENCE, LAST, SINCE = BinnedTokens.PRESENCE, BinnedTokens.VALUE_FEATURES.stop - 1, 8


def masks_hiding(hidden: Tensor) -> TokenMasks:
    nothing = torch.zeros_like(hidden)
    return TokenMasks(channel=nothing, block=nothing, token=hidden, horizon=nothing)


def test_collated_cells_are_the_layout_s_batch_and_keep_their_windows() -> None:
    windows = irregular_windows(3, seed=1, channels=4)

    collated = CellBatches(LAYOUT).collate(windows)

    expected = TokenTensors.of(LAYOUT.batch(LAYOUT.lay(windows)))
    assert isinstance(collated, CellWindows)
    assert collated.windows == tuple(windows)
    for ours, theirs in zip(collated.args, expected.args, strict=True):
        assert torch.equal(ours, theirs)


def test_cells_laid_out_once_batch_again_as_the_windows_collated_afresh() -> None:
    windows = irregular_windows(3, seed=1, channels=4)
    laid = LAYOUT.lay(windows)
    batches = CellBatches(LAYOUT)

    first = batches.collate_laid([*windows, windows[0]], [*laid, laid[0]])
    again = batches.collate_laid([*windows, windows[0]], [*laid, laid[0]])

    expected = TokenTensors.of(LAYOUT.batch(LAYOUT.lay([*windows, windows[0]])))
    for batch in (first, again):
        assert batch.windows == (*windows, windows[0])
        for ours, theirs in zip(batch.args, expected.args, strict=True):
            assert torch.equal(ours, theirs)


def test_cells_that_are_not_one_layout_per_window_are_refused() -> None:
    windows = irregular_windows(2, seed=1, channels=4)

    with pytest.raises(ValueError, match="1 layouts were given for 2 windows"):
        CellBatches(LAYOUT).collate_laid(windows, LAYOUT.lay(windows[:1]))


def test_moving_the_collated_cells_keeps_the_windows_and_retypes_the_tensors() -> None:
    collated = CellBatches(LAYOUT).collate([WINDOW, SHORT])

    moved = collated.to("cpu", torch.float64)

    assert moved.windows == (WINDOW, SHORT)
    assert moved.features.dtype == torch.float64
    assert moved.channel_ids.dtype == torch.int64


def test_the_scored_batch_holds_the_full_cells_and_the_cells_without_the_hidden_readings() -> None:
    batches = CellBatches(LAYOUT)
    collated = batches.collate([WINDOW, SHORT])
    # Cell 0 of the first window is channel 1's first bin, which held 3.0 and 5.0.
    hidden = torch.zeros_like(collated.padding_mask)
    hidden[0, 0] = True

    scored = batches.scored(collated, masks_hiding(hidden))

    assert isinstance(scored, CellTensors)
    assert torch.equal(scored.features, collated.features)
    assert torch.equal(scored.padding_mask, collated.padding_mask)
    assert scored.visible.shape == scored.features.shape
    first = scored.features[0], scored.visible[0]
    # The full layout carries 5.0 forward into channel 1's empty second bin (cell 2); the
    # visible layout has nothing to carry, and counts the bins since from the window's start.
    assert first[0][2, LAST] == 5.0
    assert first[1][2, LAST] == 0.0
    assert first[0][2, SINCE] == pytest.approx(0.25)
    assert first[1][2, SINCE] == pytest.approx(0.5)
    # The hidden cell itself stands empty in the visible layout.
    assert first[1][0, PRESENCE] == 0.0
    # The short window, nothing hidden, is laid out the same twice, padding included.
    assert torch.equal(scored.visible[1], scored.features[1])


def test_hiding_nothing_lays_the_visible_cells_out_as_the_full_ones() -> None:
    batches = CellBatches(LAYOUT)
    windows = irregular_windows(3, seed=2, channels=3)
    collated = batches.collate(windows)

    scored = batches.scored(collated, masks_hiding(torch.zeros_like(collated.padding_mask)))

    assert torch.equal(scored.visible, scored.features)


def test_the_drawn_masks_of_a_strategy_hide_whole_cells_and_the_visible_layout_follows() -> None:
    batches = CellBatches(LAYOUT)
    windows = irregular_windows(4, seed=3, channels=5)
    collated = batches.collate(windows)
    masks = TokenMasking(MIXTURE).draw(collated, torch.Generator().manual_seed(1))

    scored = batches.scored(collated, masks)

    hidden = masks.hidden & ~collated.padding_mask
    assert hidden.any()
    # Every hidden cell stands empty to the encoder, whatever the full layout held there.
    assert (scored.visible[..., PRESENCE][hidden] == 0.0).all()
    # A visible cell never reads more than the full layout did: its count is at most the full one.
    visible = ~hidden & ~collated.padding_mask
    assert (scored.visible[..., 1][visible] <= scored.features[..., 1][visible] + 1e-6).all()


def test_the_visible_layout_is_each_window_laid_without_its_hidden_cells() -> None:
    batches = CellBatches(LAYOUT)
    windows = irregular_windows(2, seed=4, channels=3)
    collated = batches.collate(windows)
    masks = TokenMasking(MIXTURE).draw(collated, torch.Generator().manual_seed(2))

    scored = batches.scored(collated, masks)

    for row, window in enumerate(windows):
        cells = int((~collated.padding_mask[row]).sum())
        expected = LAYOUT.without(window, masks.hidden[row, :cells].numpy().astype(np.bool_))
        np.testing.assert_allclose(
            scored.visible[row, :cells].numpy(), expected.features.astype(np.float32)
        )


def test_a_batch_not_collated_here_cannot_be_scored() -> None:
    plain = TokenTensors.from_windows([WINDOW])

    with pytest.raises(ValueError, match="laid from"):
        CellBatches(LAYOUT).scored(plain, masks_hiding(torch.zeros_like(plain.padding_mask)))


def test_raw_batches_are_the_readings_and_are_scored_as_collated() -> None:
    batches = RawBatches()
    windows = [WINDOW, SHORT]

    collated = batches.collate(windows)

    expected = TokenTensors.from_windows(windows)
    for ours, theirs in zip(collated.args, expected.args, strict=True):
        assert torch.equal(ours, theirs)
    masks = masks_hiding(torch.zeros_like(collated.padding_mask))
    assert batches.scored(collated, masks) is collated
