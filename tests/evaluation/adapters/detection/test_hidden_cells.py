import numpy as np
import pytest

from emblema.evaluation.adapters.detection.hidden_cells import HiddenCells
from emblema.evaluation.adapters.detection.masked_cells import MaskedCells


def cell(value: float | None) -> list[float]:
    """A cell's nine features: held with ``value`` at every summary, or empty."""
    if value is None:
        return [0.0] * 9
    return [1.0, np.log1p(1.0), value, value, value, value, value, 0.5, 0.0]


def masked(full: list[list[float | None]], hidden: list[list[bool]]) -> MaskedCells:
    """Two channels over the bins of ``full`` (rows of bins, one value per channel)."""
    rows = [cell(value) for per_bin in full for value in per_bin]
    shown = [
        cell(None if hide else value)
        for per_bin, hides in zip(full, hidden, strict=True)
        for value, hide in zip(per_bin, hides, strict=True)
    ]
    return MaskedCells(
        full=np.asarray(rows),
        visible=np.asarray(shown),
        hidden=np.asarray([hide for hides in hidden for hide in hides], dtype=np.bool_),
        channel_ids=np.asarray([1, 2] * len(full), dtype=np.int64),
        timeless=np.zeros(2 * len(full), dtype=np.bool_),
        answers=None,
    )


# Channel 1 is read at every bin and hidden in the middle one; channel 2 is hidden in the first.
CELLS = masked(
    [[1.0, 4.0], [7.0, 5.0], [3.0, 6.0]],
    [[False, True], [True, False], [False, False]],
)


def test_the_timed_cells_fold_into_bins_by_channels_and_the_hidden_held_ones_are_scored() -> None:
    hidden = HiddenCells.of(CELLS, bins=3)

    assert hidden.channels.tolist() == [1, 2]
    assert hidden.scored.tolist() == [[False, True], [True, False], [False, False]]
    assert hidden.targets[1, 0].tolist() == [7.0] * 5


def test_a_hidden_cell_is_answered_by_the_line_between_its_channel_s_visible_neighbours() -> None:
    hidden = HiddenCells.of(CELLS, bins=3)

    # Channel 1, bin 1: halfway between 1 before and 3 after.
    assert hidden.interpolated[1, 0] == pytest.approx(2.0)
    assert hidden.carried[1, 0] == pytest.approx(1.0)
    assert (hidden.before[1, 0], hidden.either[1, 0]) == (True, True)
    # Channel 2, bin 0: nothing before, so the first visible value after it.
    assert hidden.interpolated[0, 1] == pytest.approx(5.0)
    assert hidden.carried[0, 1] == 0.0
    assert (hidden.before[0, 1], hidden.either[0, 1]) == (False, True)


def test_a_channel_with_nothing_visible_is_answered_by_its_mean() -> None:
    alone = masked([[1.0, 4.0], [2.0, 5.0]], [[True, False], [True, False]])

    hidden = HiddenCells.of(alone, bins=2)

    assert hidden.interpolated[:, 0].tolist() == [0.0, 0.0]
    assert hidden.either[:, 0].tolist() == [False, False]
    assert hidden.interpolation()[0, 0].tolist() == [0.0] * 5


def test_a_hidden_cell_that_held_nothing_is_not_scored_and_static_cells_are_left_out() -> None:
    empty = masked([[None, 4.0], [7.0, 5.0]], [[True, False], [False, False]])
    with_static = MaskedCells(
        full=np.vstack((np.asarray([cell(9.0)]), empty.full)),
        visible=np.vstack((np.asarray([cell(None)]), empty.visible)),
        hidden=np.concatenate(([True], empty.hidden)),
        channel_ids=np.concatenate(([3], empty.channel_ids)),
        timeless=np.concatenate(([True], empty.timeless)),
        answers=None,
    )

    hidden = HiddenCells.of(with_static, bins=2)

    assert not hidden.scored.any()
    assert hidden.channels.tolist() == [1, 2]


def test_timed_cells_that_do_not_fill_the_bins_are_refused() -> None:
    with pytest.raises(ValueError, match="do not lie on 4 bins"):
        HiddenCells.of(CELLS, bins=4)


def spanning(first: float, last: float) -> list[float]:
    """A held cell whose first and last readings differ, its mean between them."""
    middle = (first + last) / 2
    return [1.0, np.log1p(2.0), middle, min(first, last), max(first, last), first, last, 1.0, 0.0]


def test_a_gap_of_several_bins_runs_from_the_last_reading_before_to_the_first_after() -> None:
    # One channel over five bins: read in the first and the last, hidden in the three between.
    full = np.asarray([spanning(0.0, 1.0), *[spanning(4.0, 4.0)] * 3, spanning(9.0, 10.0)])
    hidden = np.asarray([False, True, True, True, False], dtype=np.bool_)
    visible = full.copy()
    visible[hidden] = 0.0
    cells = MaskedCells(
        full=full,
        visible=visible,
        hidden=hidden,
        channel_ids=np.ones(5, dtype=np.int64),
        timeless=np.zeros(5, dtype=np.bool_),
        answers=None,
    )

    grid = HiddenCells.of(cells, bins=5)

    assert grid.interpolated[1:4, 0].tolist() == pytest.approx([3.0, 5.0, 7.0])
    assert grid.carried[1:4, 0].tolist() == pytest.approx([1.0, 1.0, 1.0])
