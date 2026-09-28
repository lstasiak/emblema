from dataclasses import dataclass
from math import log
from statistics import fmean
from typing import ClassVar, Self

import torch
from torch import Tensor
from torch.nn.functional import softplus

from emblema.evaluation.adapters.torch.regression_head import RegressionHead
from emblema.evaluation.domain.exceptions import UnsolvableHeadError
from emblema.evaluation.domain.heads.outcome_folds import OutcomeFolds
from emblema.evaluation.domain.heads.ridge_penalties import RidgePenalties


@dataclass(frozen=True)
class LogisticSolution:
    """A linear head over pooled states answering an outcome's log-odds, by penalised likelihood.

    The head a network over outcomes learns by binary cross-entropy, solved to its optimum
    instead of stepped towards it: logistic regression with an unpenalised intercept and a
    penalty of half the given strength on the squared weights, the columns scaled by their
    spread as the least-squares head's are. The answer is the log-odds itself, so the head
    ranks the windows by the fit and a sigmoid reads it as a probability, with nothing fitted
    afterwards that could reorder or flatten it.

    The penalty is the one whose heads, fitted without each of ``OutcomeFolds`` in turn, give
    the folds the smallest mean log-loss, the smaller one on a tie; the head is then fitted on
    every window. The optimiser is Newton's method with a backtracking line search, in double
    precision on the host, stopped once half the squared Newton decrement — a bound on how far
    the objective still is from its minimum — falls below the tolerance, and finished with one
    full step.

    Attributes:
        weights: The linear map over the states as they were given, one weight per column.
        intercept: What the map adds.
        penalty: The penalty the fit chose.
        held_out_loss: Mean log-loss over the folds under each penalty, in their order.
    """

    weights: Tensor
    intercept: float
    penalty: float
    held_out_loss: tuple[float, ...]

    _TOLERANCE: ClassVar[float] = 1e-10
    _ITERATIONS: ClassVar[int] = 100
    # A step is accepted when it buys at least this share of the decrease its gradient promises,
    # halved down to the smallest step.
    _ARMIJO: ClassVar[float] = 1e-4
    _SMALLEST_STEP: ClassVar[float] = 1e-10

    @classmethod
    def fitted(cls, states: Tensor, outcomes: Tensor, penalties: RidgePenalties) -> Self:
        """The head over ``states`` answering ``outcomes``, choosing among ``penalties``.

        Raises:
            UnfoldableOutcomesError: If an outcome is not zero or one, or either has fewer than
                two windows.
            UnsolvableHeadError: If a fit does not converge.
        """
        x = states.detach().to("cpu").to(torch.float64)
        y = outcomes.detach().to("cpu").to(torch.float64)
        folds = OutcomeFolds.of(y.tolist())
        spread = x.std(dim=0, correction=0)
        scale = torch.where(spread > 0.0, spread, torch.ones_like(spread))
        scaled = x / scale
        losses: list[float] = []
        for penalty in penalties.values:
            per_fold: list[float] = []
            for fold in range(folds.count):
                kept, held = list(folds.kept(fold)), list(folds.held_out(fold))
                weights, intercept = cls._maximised(scaled[kept], y[kept], penalty)
                answered = scaled[held] @ weights + intercept
                per_fold.append(float((softplus(answered) - y[held] * answered).mean()))
            losses.append(fmean(per_fold))
        chosen = min(range(len(losses)), key=lambda index: (losses[index], index))
        weights, intercept = cls._maximised(scaled, y, penalties.values[chosen])
        return cls(
            weights=weights / scale,
            intercept=intercept,
            penalty=penalties.values[chosen],
            held_out_loss=tuple(losses),
        )

    def applied_to(self, head: RegressionHead) -> None:
        """Write the solution into ``head``, in the head's precision on the head's device."""
        with torch.no_grad():
            head.linear.weight.copy_(self.weights.to(head.linear.weight).unsqueeze(0))
            head.linear.bias.fill_(self.intercept)

    @classmethod
    def _maximised(cls, x: Tensor, y: Tensor, penalty: float) -> tuple[Tensor, float]:
        """The weights and intercept of largest penalised likelihood of ``y`` over ``x``.

        Raises:
            UnsolvableHeadError: If Newton's method does not converge.
        """
        count, width = x.shape
        design = torch.cat([x, torch.ones(count, 1, dtype=x.dtype)], dim=1)
        shrunk = torch.full((width + 1,), penalty, dtype=x.dtype)
        shrunk[width] = 0.0
        share = float(y.mean())
        theta = torch.zeros(width + 1, dtype=x.dtype)
        theta[width] = log(share / (1.0 - share))
        objective = cls._objective(design, y, shrunk, theta)
        for _ in range(cls._ITERATIONS):
            probability = torch.sigmoid(design @ theta)
            gradient = design.T @ (probability - y) + shrunk * theta
            curvature = design.T @ (design * (probability * (1.0 - probability)).unsqueeze(1))
            curvature += torch.diag(shrunk)
            solved, singular = torch.linalg.solve_ex(curvature, gradient)
            if int(singular) != 0:
                raise UnsolvableHeadError("the likelihood has no curvature to step by")
            step = -solved
            promised = float(gradient @ step)
            if -promised / 2.0 <= cls._TOLERANCE:
                # Close enough that a full step squares what error is left, where a line search
                # could no longer tell a decrease from rounding.
                theta = theta + step
                return theta[:width], float(theta[width])
            size = 1.0
            while size >= cls._SMALLEST_STEP:
                trial = theta + size * step
                reached = cls._objective(design, y, shrunk, trial)
                if reached <= objective + cls._ARMIJO * size * promised:
                    theta, objective = trial, reached
                    break
                size /= 2.0
            else:
                raise UnsolvableHeadError("the line search found no step that lowers the loss")
        raise UnsolvableHeadError(f"the fit did not converge in {cls._ITERATIONS} iterations")

    @staticmethod
    def _objective(design: Tensor, y: Tensor, shrunk: Tensor, theta: Tensor) -> float:
        """Negative log-likelihood plus the penalty, computed without overflow at any log-odds."""
        answered = design @ theta
        return float((softplus(answered) - y * answered).sum() + 0.5 * (shrunk * theta**2).sum())
