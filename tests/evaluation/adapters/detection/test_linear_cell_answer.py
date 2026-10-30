import numpy as np
import pytest

from emblema.evaluation.adapters.detection.hidden_cells import HiddenCells
from emblema.evaluation.adapters.detection.linear_cell_answer import LinearCellAnswer
from tests.evaluation.adapters.detection.test_hidden_cells import masked


def draws(count: int, *, seed: int) -> list[tuple[HiddenCells, int]]:
    """Windows whose channel 1 is always twice channel 2 in the same bin, channel 1 hidden once."""
    generator = np.random.default_rng(seed)
    drawn = []
    for at in range(count):
        values = generator.normal(size=3)
        hidden_bin = at % 3
        full: list[list[float | None]] = [[2.0 * value, value] for value in values]
        hidden = [[b == hidden_bin, False] for b in range(3)]
        drawn.append((HiddenCells.of(masked(full, hidden), bins=3), at % 2))
    return drawn


def test_a_ridge_learns_what_the_other_channels_in_the_bin_say_about_a_hidden_cell() -> None:
    answer = LinearCellAnswer.fitted([1, 2], 3, draws(400, seed=1))
    cells, _ = draws(1, seed=2)[0]

    answered = answer.answers(cells)

    at = np.nonzero(cells.scored)
    assert np.allclose(answered[at], cells.targets[at], atol=0.05)
    assert set(answer.weights) == {1}


def test_the_ridge_beats_the_line_where_the_other_channels_carry_the_answer() -> None:
    answer = LinearCellAnswer.fitted([1, 2], 3, draws(400, seed=1))

    ridge, line = 0.0, 0.0
    for cells, _ in draws(50, seed=3):
        at = np.nonzero(cells.scored)
        ridge += float(((answer.answers(cells)[at] - cells.targets[at]) ** 2).sum())
        line += float(((cells.interpolation()[at] - cells.targets[at]) ** 2).sum())

    assert ridge < 0.01 * line


def test_a_channel_the_fitting_side_never_hid_is_answered_by_its_line() -> None:
    answer = LinearCellAnswer.fitted([1, 2], 3, draws(20, seed=1))
    cells = HiddenCells.of(
        masked(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[False, False], [False, True], [False, False]],
        ),
        bins=3,
    )

    answered = answer.answers(cells)

    assert 2 not in answer.weights
    assert answered[1, 1].tolist() == pytest.approx([cells.interpolated[1, 1]] * 5)


def test_a_window_with_a_channel_no_fit_laid_out_is_refused() -> None:
    answer = LinearCellAnswer.fitted([1], 3, [])
    cells, _ = draws(1, seed=2)[0]

    with pytest.raises(ValueError, match="which no fit laid out"):
        answer.answers(cells)


def test_the_squared_error_read_from_the_sums_is_the_one_over_the_rows() -> None:
    from emblema.evaluation.adapters.detection.linear_cell_answer import _Statistics

    generator = np.random.default_rng(4)
    described = generator.normal(size=(200, 6))
    targets = generator.normal(size=(200, 5))
    weights = generator.normal(size=(6, 5))
    sums = _Statistics(6)
    sums.add(described[:120], targets[:120])
    sums.add(described[120:], targets[120:])

    direct = float(((described @ weights - targets) ** 2).sum())
    assert sums.missed(weights) == pytest.approx(direct)
