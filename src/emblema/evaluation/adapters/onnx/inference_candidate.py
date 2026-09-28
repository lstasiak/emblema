from torch import Tensor, nn

from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone
from emblema.evaluation.adapters.torch.target_link import TargetLink
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class InferenceCandidate(nn.Module):
    """A fitted candidate as an inference graph computes it: five tensors in, two answers out.

    The candidate is called with a batch value and answers through its head, in units of the
    label scale or as log-odds, which suits a training loop and not a graph: a graph is traced
    over positional tensors and is run by a context that knows nothing of the task. So this
    takes the tensors in the order the batch declares them and returns the pooled state beside
    the answer the task's link makes of the head — a quantity in the task's unit, or a
    probability. One graph serves the embedding and the prediction, and a runtime that asks for
    one output computes only that.

    Attributes:
        candidate: The fitted candidate: encoder, pooling and head.
        link: How the head's output becomes the task's answer.
    """

    def __init__(self, candidate: AdaptedBackbone, *, link: TargetLink) -> None:
        super().__init__()
        self.candidate = candidate
        self.link = link
        # A graph is traced in evaluation mode; the composition puts itself there so that no
        # caller has to remember, and the exporter has nothing to warn about.
        self.eval()

    def forward(
        self,
        features: Tensor,
        channel_ids: Tensor,
        timestamps: Tensor,
        timeless: Tensor,
        padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch = TokenTensors(features, channel_ids, timestamps, timeless, padding_mask)
        pooled = self.candidate.embed(batch)
        return pooled, self.link.answered(self.candidate.head(pooled))
