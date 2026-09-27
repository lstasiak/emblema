from dataclasses import dataclass, replace
from typing import Self

import torch
from torch import Tensor

from emblema.evaluation.adapters.torch.regression_head import RegressionHead
from emblema.evaluation.adapters.torch.target_link import TargetLink
from emblema.evaluation.domain.exceptions import UnsolvableHeadError
from emblema.evaluation.domain.heads.logistic_calibration import LogisticCalibration
from emblema.evaluation.domain.heads.ridge_penalties import RidgePenalties
from emblema.evaluation.domain.labels.target_kind import TargetKind


@dataclass(frozen=True)
class RidgeSolution:
    """A linear head solved in closed form over pooled states, its penalty chosen leave-one-out.

    Ridge with an unpenalised intercept, the columns scaled by their spread so that one penalty
    means the same for every state, solved once through a singular value decomposition and
    read off for every penalty in turn: the leave-one-out residuals of a linear smoother are
    the residuals divided by one minus the leverage, so the choice costs no refit. The penalty
    with the smallest mean squared leave-one-out error wins, the smaller one on a tie. Solved
    on the host in double precision, since the states come off an accelerator that may have
    none, and moved there before they are widened.

    Over outcomes of zero and one the solution ranks them as well as any linear head but does not
    answer probabilities; the head then answers log-odds, so the solution is scaled by a logistic
    calibration fitted on the leave-one-out answers — answers each window got from a head that
    had not seen it — rather than on the in-sample ones, which would calibrate the head to its
    own overconfidence. The scaling is monotone, so the ranking is the ridge's.

    Attributes:
        weights: The linear map over the states as they were given, one weight per column.
        intercept: What the map adds.
        penalty: The penalty the fit chose.
        leave_one_out: Mean squared leave-one-out error under each penalty, in their order.
        left_out_answers: What each window was answered under the chosen penalty by the head
            fitted without it.
    """

    weights: Tensor
    intercept: float
    penalty: float
    leave_one_out: tuple[float, ...]
    left_out_answers: Tensor

    @classmethod
    def fitted(cls, states: Tensor, targets: Tensor, penalties: RidgePenalties) -> Self:
        """The head over ``states`` answering ``targets``, choosing among ``penalties``.

        Raises:
            UnsolvableHeadError: If there are fewer than two windows, which leaves nothing to
                leave out.
        """
        x = states.detach().to("cpu").to(torch.float64)
        y = targets.detach().to("cpu").to(torch.float64)
        count = x.shape[0]
        if count < 2:
            raise UnsolvableHeadError(
                f"a head chosen by leave-one-out error needs at least two windows, got {count}"
            )
        spread = x.std(dim=0, correction=0)
        scale = torch.where(spread > 0.0, spread, torch.ones_like(spread))
        scaled = x / scale
        x_mean, y_mean = scaled.mean(dim=0), y.mean()
        centred, residual_targets = scaled - x_mean, y - y_mean
        left, singular, right_t = torch.linalg.svd(centred, full_matrices=False)
        projected = left.T @ residual_targets
        squared = singular**2
        errors: list[float] = []
        coefficients: list[Tensor] = []
        left_out: list[Tensor] = []
        for penalty in penalties.values:
            shrink = squared / (squared + penalty)
            fitted = left @ (shrink * projected)
            # The intercept is unpenalised, so its leverage of one in ``count`` stays whole.
            leverage = (left**2 * shrink).sum(dim=1) + 1.0 / count
            loo = (residual_targets - fitted) / (1.0 - leverage)
            errors.append(float((loo**2).mean()))
            left_out.append(y - loo)
            coefficients.append(right_t.T @ (singular / (squared + penalty) * projected))
        chosen = min(range(len(errors)), key=lambda index: (errors[index], index))
        coefficient = coefficients[chosen]
        return cls(
            weights=coefficient / scale,
            intercept=float(y_mean - (x_mean * coefficient).sum()),
            penalty=penalties.values[chosen],
            leave_one_out=tuple(errors),
            left_out_answers=left_out[chosen],
        )

    def linked(self, link: TargetLink, taught: Tensor) -> Self:
        """The solution as ``link`` reads the head: as it is, or scaled to answer log-odds.

        Raises:
            UncalibratableScoresError: If the outcomes the head was taught cannot be calibrated.
        """
        match link.kind:
            case TargetKind.CONTINUOUS:
                return self
            case TargetKind.BINARY:
                calibration = LogisticCalibration.fitted(
                    self.left_out_answers.tolist(), taught.detach().to("cpu").tolist()
                )
                return replace(
                    self,
                    weights=self.weights * calibration.slope,
                    intercept=calibration.log_odds(self.intercept),
                )

    def applied_to(self, head: RegressionHead) -> None:
        """Write the solution into ``head``, in the head's precision on the head's device."""
        with torch.no_grad():
            head.linear.weight.copy_(self.weights.to(head.linear.weight).unsqueeze(0))
            head.linear.bias.fill_(self.intercept)
