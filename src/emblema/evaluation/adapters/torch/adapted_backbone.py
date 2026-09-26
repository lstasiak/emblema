from typing import Self

from torch import Tensor, nn

from emblema.evaluation.adapters.torch.backbone_factory import BackboneFactory
from emblema.evaluation.adapters.torch.low_rank_adaptation import LowRankAdaptation
from emblema.evaluation.adapters.torch.pooling import pooling_module
from emblema.evaluation.adapters.torch.regression_head import RegressionHead
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.shared.adapters.tensors.grown_parameters import GrownParameters
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class AdaptedBackbone(nn.Module):
    """The candidate a plan makes: encoder, pooling and head, the mode's weights left free.

    The head is new in every mode and the candidate is called the same way whatever the mode, so
    a comparison between modes is a comparison between backbones. The pooling is the plan's: a
    variant that turns it is the same backbone under another head, and whatever weights the
    pooling has of its own train under every mode, as the head does.

    Attributes:
        encoder: The backbone, with whatever the mode left trainable.
        pooling: One state per window out of the states per token, as the plan named it.
        head: The task's answer out of the pooled state.
    """

    def __init__(self, encoder: nn.Module, pooling: nn.Module, head: RegressionHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.head = head

    @classmethod
    def under(
        cls,
        plan: AdaptationPlan,
        backbones: BackboneFactory,
        *,
        vocabulary_size: int,
        starting_at: float,
    ) -> Self:
        """The candidate ``plan`` describes, built on the host from torch's current generator.

        The head is drawn first, so under one seed every mode starts it from the same weights,
        whatever the encoder draws after it; its bias starts at ``starting_at``, the mean of the
        labels the run holds in the task's scale. The encoder is asked for over
        ``vocabulary_size``, the channels of the task's corpus; rows it grew for channels it was
        never trained on are trained under every mode, as the head is, because there is nothing
        pretrained in them to freeze.

        Raises:
            UnknownBackboneError: If the plan names pretrained weights the factory does not
                serve.
            LoraTargetNotFoundError: If the plan's low-rank updates name a layer the backbone
                does not have.
        """
        head = RegressionHead(backbones.width, starting_at=starting_at)
        pooling = pooling_module(plan.pooling, width=backbones.width)
        if plan.backbone is None:
            encoder = backbones.fresh(vocabulary_size=vocabulary_size)
        else:
            encoder = backbones.pretrained(plan.backbone, vocabulary_size=vocabulary_size)
        encoder.requires_grad_(plan.mode.trains_backbone_weights)
        for module in encoder.modules():
            if isinstance(module, GrownParameters):
                for parameter in module.grown_parameters():
                    parameter.requires_grad_(True)
        if plan.lora is not None:
            LowRankAdaptation(plan.lora).applied_to(encoder)
        return cls(encoder, pooling, head)

    def embed(self, batch: TokenTensors) -> Tensor:
        """One state per window, ``[batch, width]``."""
        pooled: Tensor = self.pooling(
            self.encoder(*batch.args), batch.padding_mask, batch.timestamps, batch.timeless
        )
        return pooled

    def forward(self, batch: TokenTensors) -> Tensor:
        return self.head(self.embed(batch))

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]
