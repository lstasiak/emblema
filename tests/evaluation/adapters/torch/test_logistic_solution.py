"""The head over outcomes against the reference classifier, and what it writes into a head."""

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

from emblema.evaluation.adapters.torch.logistic_solution import LogisticSolution  # noqa: E402
from emblema.evaluation.adapters.torch.regression_head import RegressionHead  # noqa: E402
from emblema.evaluation.domain.exceptions import UnfoldableOutcomesError  # noqa: E402
from emblema.evaluation.domain.heads.outcome_folds import OutcomeFolds  # noqa: E402
from emblema.evaluation.domain.heads.ridge_penalties import RidgePenalties  # noqa: E402

pytestmark = pytest.mark.ml

PENALTIES = RidgePenalties((0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0))


def drawn(
    seed: int, count: int = 50, width: int = 32, signal: float = 0.1
) -> tuple[Tensor, Tensor]:
    """States of unequal spread and rare outcomes they tell apart as weakly as ``signal`` says."""
    generator = torch.Generator().manual_seed(seed)
    spread = torch.arange(1, width + 1, dtype=torch.float64)
    states = torch.randn(count, width, generator=generator, dtype=torch.float64) * spread
    read = (states / spread) @ torch.randn(width, generator=generator, dtype=torch.float64)
    noise = torch.randn(count, generator=generator, dtype=torch.float64)
    return states, (signal * read + noise > 1.2).to(torch.float64)


def scaled(states: Tensor) -> Tensor:
    spread = states.std(dim=0, correction=0)
    return states / torch.where(spread > 0.0, spread, torch.ones_like(spread))


def test_the_head_at_one_penalty_is_the_reference_classifiers_to_working_precision() -> None:
    sklearn_linear_model = pytest.importorskip("sklearn.linear_model")
    states, outcomes = drawn(seed=1, count=200, signal=0.5)

    solved = LogisticSolution.fitted(states, outcomes, RidgePenalties((3.0,)))

    # The reference is fitted on the same scaled columns, since scaling is ours, not its; its
    # inverse strength is the reciprocal of the penalty.
    spread = states.std(dim=0, correction=0)
    reference = sklearn_linear_model.LogisticRegression(
        C=1.0 / 3.0, solver="newton-cholesky", tol=1e-12, max_iter=1000
    ).fit(scaled(states).numpy(), outcomes.numpy())
    assert torch.allclose(
        solved.weights, torch.as_tensor(reference.coef_[0]) / spread, rtol=1e-6, atol=1e-8
    )
    assert solved.intercept == pytest.approx(float(reference.intercept_[0]), rel=1e-6)


def test_the_penalty_chosen_gives_the_folds_the_smallest_log_loss() -> None:
    sklearn_linear_model = pytest.importorskip("sklearn.linear_model")
    states, outcomes = drawn(seed=2, count=200, signal=0.5)
    folds = OutcomeFolds.of(outcomes.tolist())
    splits = [(list(folds.kept(f)), list(folds.held_out(f))) for f in range(folds.count)]

    solved = LogisticSolution.fitted(states, outcomes, PENALTIES)

    reference = sklearn_linear_model.LogisticRegressionCV(
        Cs=[1.0 / penalty for penalty in PENALTIES.values],
        l1_ratios=(0.0,),
        cv=splits,
        scoring="neg_log_loss",
        solver="newton-cholesky",
        tol=1e-12,
        max_iter=1000,
        use_legacy_attributes=False,
    ).fit(scaled(states).numpy(), outcomes.numpy())
    assert len(solved.held_out_loss) == len(PENALTIES.values)
    assert solved.penalty == PENALTIES.values[solved.held_out_loss.index(min(solved.held_out_loss))]
    assert solved.penalty == pytest.approx(1.0 / float(reference.C_))


def test_outcomes_the_states_separate_still_give_a_finite_head() -> None:
    states, _ = drawn(seed=3, count=40)
    outcomes = (states[:, 0] > 0.0).to(torch.float64)

    solved = LogisticSolution.fitted(states, outcomes, RidgePenalties((0.001,)))

    answered = states @ solved.weights + solved.intercept
    assert torch.isfinite(solved.weights).all()
    assert bool((answered[outcomes == 1.0].min() > answered[outcomes == 0.0].max()).item())


@pytest.mark.parametrize("seed", range(10))
def test_outcomes_the_states_barely_tell_apart_are_still_ranked_by_the_head(seed: int) -> None:
    # Fifty windows and a weak signal: where a least-squares head calibrated afterwards flattened
    # or reversed its ranking, the likelihood's own head answers every window apart, and at its
    # optimum its log-odds lean towards the outcomes it was fitted on, never against them.
    states, outcomes = drawn(seed)
    fresh, _ = drawn(seed + 100, count=500)

    solved = LogisticSolution.fitted(states, outcomes, PENALTIES)

    answered = fresh @ solved.weights + solved.intercept
    fitted = states @ solved.weights + solved.intercept
    assert torch.unique(answered).numel() == len(answered)
    assert fitted[outcomes == 1.0].mean() > fitted[outcomes == 0.0].mean()


def test_the_solution_written_into_a_head_is_what_the_head_then_answers() -> None:
    states, outcomes = drawn(seed=4, count=200, signal=0.5)
    solved = LogisticSolution.fitted(states, outcomes, PENALTIES)
    head = RegressionHead(states.shape[1], starting_at=0.0)

    solved.applied_to(head)

    expected = states @ solved.weights + solved.intercept
    assert torch.allclose(head(states.float()).double(), expected, atol=1e-3)


def test_outcomes_too_few_to_fold_are_refused() -> None:
    states, _ = drawn(seed=5, count=10)
    outcomes = torch.tensor([1.0] + [0.0] * 9, dtype=torch.float64)

    with pytest.raises(UnfoldableOutcomesError, match="got 1 of outcome 1"):
        LogisticSolution.fitted(states, outcomes, PENALTIES)
