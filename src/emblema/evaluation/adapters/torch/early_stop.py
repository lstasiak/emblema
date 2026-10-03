from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn

from emblema.evaluation.adapters.torch.target_link import TargetLink
from emblema.evaluation.domain.labels.target_kind import TargetKind


class EarlyStop:
    """Keeps the weights of the best epoch on held-out windows and says when to give up.

    Scored after every epoch on windows the run does not learn from; a score that betters the
    best so far is kept with a copy of the model's weights on the host, and the run is told to
    stop once ``patience`` epochs pass without one. The score is the sum of the areas under the
    ROC and the precision-recall curves for an outcome, as the published network stops on, and
    the negative mean squared error for a quantity; higher is better in both.

    Attributes:
        patience: Epochs without a better score before the run stops.
        best: The best score seen; minus infinity before the first.
        best_epoch: Index of the epoch that scored it; ``None`` before the first.
    """

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("-inf")
        self.best_epoch: int | None = None
        self._since_best = 0
        self._weights: dict[str, Any] | None = None  # a state dict: torch's own mapping

    @staticmethod
    def score(link: TargetLink, answers: Tensor, targets: Tensor) -> float:
        """How well ``answers`` fit ``targets`` under the task's link; higher is better.

        For an outcome, the areas under the ROC and the precision-recall curves summed, over the
        answers as probabilities; ties count half in the ROC area. For a quantity, the negative
        mean squared error in the unit the head is taught in.
        """
        answers = answers.detach().to("cpu", torch.float64)
        targets = targets.detach().to("cpu", torch.float64)
        match link.kind:
            case TargetKind.CONTINUOUS:
                return -float(((answers - targets) ** 2).mean())
            case TargetKind.BINARY:
                return _roc_area(answers, targets) + _precision_recall_area(answers, targets)

    def observe(self, epoch: int, score: float, model: nn.Module) -> bool:
        """Record the epoch's score and the weights if it is the best; whether to stop now."""
        if score > self.best:
            self.best, self.best_epoch, self._since_best = score, epoch, 0
            self._weights = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
        else:
            self._since_best += 1
        return self._since_best >= self.patience

    def restore(self, model: nn.Module) -> None:
        """Put the best epoch's weights back into ``model``; a model never scored is left alone."""
        if self._weights is not None:
            model.load_state_dict(deepcopy(self._weights))


def _roc_area(answers: Tensor, targets: Tensor) -> float:
    positive = targets > 0.5
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(answers)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _precision_recall_area(answers: Tensor, targets: Tensor) -> float:
    """Average precision, the precision-recall area as the published network's code reads it."""
    positive = targets > 0.5
    n_pos = int(positive.sum())
    if n_pos == 0:
        return 0.0
    order = torch.argsort(answers, descending=True, stable=True)
    hits = positive[order].to(torch.float64)
    precision = hits.cumsum(0) / torch.arange(1, len(hits) + 1, dtype=torch.float64)
    return float((precision * hits).sum() / n_pos)


def _average_ranks(values: Tensor) -> Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(1, len(values) + 1, dtype=values.dtype)
    sorted_values = values[order]
    # Ties take the mean of the ranks they span.
    unique, inverse = torch.unique(sorted_values, return_inverse=True)
    sums = torch.zeros(len(unique), dtype=values.dtype).index_add_(0, inverse, ranks[order])
    counts = torch.zeros(len(unique), dtype=values.dtype).index_add_(
        0, inverse, torch.ones_like(ranks)
    )
    averaged = (sums / counts)[inverse]
    ranks[order] = averaged
    return ranks
