import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from emblema.evaluation.adapters.torch.summary_branch import SummaryBranch  # noqa: E402

pytestmark = pytest.mark.ml

NAN = math.nan
# Columns: varying, never defined, one value everywhere, two values and a gap, one value once.
ROWS = np.array(
    [
        [1.0, NAN, 3.0, 2.0, NAN],
        [2.0, NAN, 3.0, NAN, 7.0],
        [3.0, NAN, 3.0, 6.0, NAN],
    ]
)


def branch() -> SummaryBranch:
    torch.manual_seed(0)
    return SummaryBranch.fitted(ROWS, width=5)


def test_the_branch_keeps_only_the_columns_the_drawn_windows_define_twice_or_more() -> None:
    assert branch().columns.tolist() == [0, 3]


def test_a_column_is_standardised_by_the_values_the_drawn_windows_define() -> None:
    fitted = branch()

    assert fitted.centre.tolist() == pytest.approx([2.0, 4.0])
    assert fitted.scale.tolist() == pytest.approx([math.sqrt(2.0 / 3.0), 2.0])
    assert (fitted.lower.tolist(), fitted.upper.tolist()) == ([1.0, 2.0], [3.0, 6.0])


def test_a_statistic_a_window_cannot_support_lands_on_the_drawn_windows_mean() -> None:
    fitted = branch()
    gap = torch.tensor([[1.0, NAN, 3.0, NAN, NAN]])
    at_the_mean = torch.tensor([[1.0, NAN, 3.0, 4.0, NAN]])

    assert torch.equal(fitted(gap), fitted(at_the_mean))
    assert torch.isfinite(fitted(gap)).all()


def test_a_scored_window_beyond_the_drawn_range_reads_as_the_range_s_edge() -> None:
    fitted = branch()
    beyond = torch.tensor([[1000.0, NAN, 3.0, -1e12, 5.0]])
    at_the_edges = torch.tensor([[3.0, NAN, 3.0, 2.0, 5.0]])

    assert torch.equal(fitted(beyond), fitted(at_the_edges))


def test_the_branch_answers_one_state_of_its_width_per_window() -> None:
    assert branch()(torch.from_numpy(ROWS).to(torch.float32)).shape == (3, 5)


def test_what_the_branch_was_prepared_by_is_kept_with_its_weights() -> None:
    state = branch().state_dict()

    assert {"columns", "lower", "upper", "centre", "scale"} <= state.keys()
    assert state["hidden.weight"].shape == (5, 2)
