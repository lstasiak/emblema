from torch import Tensor, nn

from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class InferenceCandidate(nn.Module):
    """A fitted candidate as an inference graph computes it: five tensors in, two answers out.

    The candidate is called with a batch value and answers in units of the label ceiling, which
    suits a training loop and not a graph: a graph is traced over positional tensors and is run
    by a context that knows nothing of the ceiling. So this takes the tensors in the order the
    batch declares them and returns the pooled state beside the answer multiplied back into the
    task's unit. One graph serves the embedding and the prediction, and a runtime that asks for
    one output computes only that.

    Attributes:
        candidate: The fitted candidate: encoder, pooling and head.
        target_scale: What the head's answer is multiplied by to read in the task's unit.
    """

    def __init__(self, candidate: AdaptedBackbone, *, target_scale: float) -> None:
        super().__init__()
        self.candidate = candidate
        self.target_scale = target_scale
        # A graph is traced in evaluation mode; the composition puts itself there so that no
        # caller has to remember, and the exporter has nothing to warn about.
        self.eval()

    @property
    def width(self) -> int:
        return self.candidate.head.linear.in_features

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
        return pooled, self.candidate.head(pooled) * self.target_scale
