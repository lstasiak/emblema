import math

import torch
from torch import Tensor, nn
from torch.nn import functional

from emblema.pretraining.domain.training.contrastive_term import ContrastiveTerm


class ContrastiveLoss(nn.Module):
    """The normalised-temperature cross-entropy over the two halves of every window of a batch.

    Each half of each window is an anchor whose positive is the other half of the same window
    and whose distractors are both halves of every other window of the batch (Chen et al. 2020,
    NT-Xent). The similarity is the cosine divided by the term's temperature; the loss of an
    anchor is the cross-entropy of picking its positive among every candidate but itself. The
    windows are of one corpus, as a micro-batch is, so what tells them apart is what varies
    within a corpus, not which corpus they come from.

    A window whose half holds no token is no pair: it is left out as an anchor and as a
    distractor, since the state of nothing is not a window's. One window alone has no
    distractor and costs nothing, as ``ContrastiveTerm.chance_of`` states. Summed rather than
    averaged, as the reconstruction's scorer sums, so whoever weighs the term divides once.

    The arithmetic is done in single precision at least, whatever the projections came in, and
    outside any autocast: a softmax over similarities divided by a tenth is where half precision
    saturates first.
    """

    def __init__(self, term: ContrastiveTerm) -> None:
        """Score the pairs at the term's temperature.

        Raises:
            ValueError: If the term is not stated, since it then has no temperature.
        """
        super().__init__()
        if not term.is_stated:
            raise ValueError("a contrastive loss needs a stated term, which has a temperature")
        self.term = term

    def summed(self, first: Tensor, second: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """The loss summed over the anchors of the valid windows, and how many anchors there were.

        ``first`` and ``second`` are the projections of the two halves, ``[batch, width]`` each;
        ``valid`` says which windows are pairs, ``[batch]``.
        """
        pairs = int(valid.sum())
        if pairs < 2:
            zero = torch.zeros((), dtype=torch.float32, device=first.device)
            return zero, torch.zeros((), dtype=torch.int64, device=first.device)
        anchors = 2 * pairs
        halves = torch.cat([first[valid], second[valid]]).to(
            torch.promote_types(first.dtype, torch.float32)
        )
        with torch.autocast(device_type=halves.device.type, enabled=False):
            normalised = functional.normalize(halves, dim=-1)
            similarities = normalised @ normalised.T / self.term.temperature
            # An anchor is not its own candidate: its similarity to itself is the largest of
            # all and would be the easiest answer.
            own = torch.eye(anchors, dtype=torch.bool, device=halves.device)
            candidates = similarities.masked_fill(own, -math.inf)
            positives = torch.arange(anchors, device=halves.device).roll(pairs)
            losses = (
                torch.logsumexp(candidates, dim=1)
                - candidates[torch.arange(anchors, device=halves.device), positives]
            )
        return losses.sum(), torch.tensor(anchors, device=first.device)

    def chance(self, valid: Tensor) -> Tensor:
        """What the same anchors cost a model whose similarities are all equal, summed."""
        pairs = int(valid.sum())
        anchors = 2 * pairs if pairs >= 2 else 0
        return torch.tensor(
            anchors * ContrastiveTerm.chance_of(pairs), dtype=torch.float32, device=valid.device
        )
