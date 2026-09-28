from dataclasses import dataclass
from math import log
from typing import ClassVar, Self

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss

from emblema.evaluation.domain.labels.label_scheme import LabelScheme
from emblema.evaluation.domain.labels.target_kind import TargetKind


@dataclass(frozen=True)
class TargetLink:
    """How a network's linear head reaches the task's answer: a generalised linear model's link.

    Every network of this context ends in one linear layer; what its output means is fixed by
    the kind of the target. A quantity is learnt divided by the scheme's scale under squared
    error, and the answer is multiplied back. An outcome is learnt as the log-odds of the
    positive one under binary cross-entropy, and the answer is the sigmoid: the probability a
    campaign ranks and a consumer reads. The bias starts at the mean label in the head's unit,
    or at the log-odds of the prevalence. No class is weighted: the area under the ROC curve
    does not depend on the prevalence, and weighting would move the probabilities off the
    outcomes.

    Attributes:
        kind: What kind of number the target is.
        scale: What a quantity is divided by to be learnt; one for an outcome.
    """

    kind: TargetKind
    scale: float

    # How close to certainty a head may start. A sample of one outcome is refused before it
    # gets here; this keeps the starting log-odds finite all the same.
    _SMALLEST_SHARE: ClassVar[float] = 1e-4

    @classmethod
    def of(cls, scheme: LabelScheme) -> Self:
        """The link for targets read by ``scheme``."""
        return cls(kind=scheme.kind, scale=scheme.scale)

    @classmethod
    def named(cls, name: str, scale: float) -> Self:
        """The link a kept candidate recorded under ``name``.

        Raises:
            ValueError: If no kind of target is called that.
        """
        return cls(kind=TargetKind(name), scale=scale)

    def learnt(self, target: float) -> float:
        """What the head is taught for ``target``."""
        match self.kind:
            case TargetKind.CONTINUOUS:
                return target / self.scale
            case TargetKind.BINARY:
                return target

    def starting_at(self, mean_target: float) -> float:
        """Where the head's bias starts, for labels whose mean is ``mean_target``."""
        match self.kind:
            case TargetKind.CONTINUOUS:
                return mean_target / self.scale
            case TargetKind.BINARY:
                share = min(max(mean_target, self._SMALLEST_SHARE), 1.0 - self._SMALLEST_SHARE)
                return log(share / (1.0 - share))

    def loss(self, raw: Tensor, taught: Tensor) -> Tensor:
        """What the network descends: the negative log-likelihood of the target's family."""
        match self.kind:
            case TargetKind.CONTINUOUS:
                return mse_loss(raw, taught)
            case TargetKind.BINARY:
                return binary_cross_entropy_with_logits(raw, taught)

    def answered(self, raw: Tensor) -> Tensor:
        """The head's output as the task's answer: in the target's unit, or as a probability."""
        match self.kind:
            case TargetKind.CONTINUOUS:
                return raw * self.scale
            case TargetKind.BINARY:
                return torch.sigmoid(raw)
