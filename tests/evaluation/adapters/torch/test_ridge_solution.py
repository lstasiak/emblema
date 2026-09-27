"""The closed-form head against the reference regressor, and what it writes into a head."""

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

from emblema.evaluation.adapters.torch.regression_head import RegressionHead  # noqa: E402
from emblema.evaluation.adapters.torch.ridge_solution import RidgeSolution  # noqa: E402
from emblema.evaluation.domain.exceptions import UnsolvableHeadError  # noqa: E402
from emblema.evaluation.domain.heads.ridge_penalties import RidgePenalties  # noqa: E402

pytestmark = pytest.mark.ml

PENALTIES = RidgePenalties((0.01, 0.1, 1.0, 10.0, 100.0))


def drawn(count: int = 40, width: int = 6, seed: int = 3) -> tuple[Tensor, Tensor]:
    """States of unequal spread and targets that are a noisy linear read of them."""
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn(count, width, generator=generator) * torch.arange(1, width + 1)
    true = torch.randn(width, generator=generator)
    targets = states @ true + 3.0 + 0.5 * torch.randn(count, generator=generator)
    return states, targets


def test_the_solution_is_the_reference_regressors_to_working_precision() -> None:
    sklearn_linear_model = pytest.importorskip("sklearn.linear_model")
    states, targets = drawn()

    solved = RidgeSolution.fitted(states, targets, PENALTIES)

    # The reference is fitted on the same scaled columns, since scaling is ours, not its.
    spread = states.double().std(dim=0, correction=0)
    scaled = (states.double() / spread).numpy()
    reference = sklearn_linear_model.RidgeCV(alphas=PENALTIES.values).fit(
        scaled, targets.double().numpy()
    )
    assert solved.penalty == reference.alpha_
    assert torch.allclose(
        solved.weights, torch.as_tensor(reference.coef_) / spread, rtol=1e-8, atol=1e-8
    )
    assert solved.intercept == pytest.approx(reference.intercept_, rel=1e-8)


def test_the_penalty_chosen_is_the_one_with_the_smallest_leave_one_out_error() -> None:
    states, targets = drawn()

    solved = RidgeSolution.fitted(states, targets, PENALTIES)

    assert len(solved.leave_one_out) == len(PENALTIES.values)
    assert solved.penalty == PENALTIES.values[solved.leave_one_out.index(min(solved.leave_one_out))]


def test_the_solution_written_into_a_head_is_what_the_head_then_answers() -> None:
    states, targets = drawn()
    head = RegressionHead(states.shape[1], starting_at=0.0)

    RidgeSolution.fitted(states, targets, PENALTIES).applied_to(head)

    answered = head(states)
    expected = states.double() @ RidgeSolution.fitted(states, targets, PENALTIES).weights
    assert torch.allclose(answered.double(), expected + head.linear.bias.item(), atol=1e-4)


def test_a_column_that_never_varies_is_left_unscaled_and_carries_no_weight() -> None:
    states, targets = drawn()
    states[:, 2] = 7.0

    solved = RidgeSolution.fitted(states, targets, PENALTIES)

    assert solved.weights[2].item() == pytest.approx(0.0)
    assert torch.isfinite(solved.weights).all()


def test_fewer_than_two_windows_leave_nothing_to_leave_out() -> None:
    states, targets = drawn(count=1)

    with pytest.raises(UnsolvableHeadError, match="at least two windows"):
        RidgeSolution.fitted(states, targets, PENALTIES)
