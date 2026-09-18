from typing import Self

from torch import Tensor, nn

from emblema.evaluation.adapters.torch.backbone_factory import BackboneFactory
from emblema.evaluation.adapters.torch.low_rank_adaptation import LowRankAdaptation
from emblema.evaluation.adapters.torch.regression_head import RegressionHead
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.shared.adapters.tensors.masked_mean_pooling import MaskedMeanPooling
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class AdaptedBackbone(nn.Module):
    """The candidate a plan makes: encoder, pooling and head, the mode's weights left free.

    The head is new in every mode and the candidate is called the same way whatever the mode, so
    a comparison between modes is a comparison between backbones.

    Attributes:
        encoder: The backbone, with whatever the mode left trainable.
        pooling: One state per window out of the states per token.
        head: The task's answer out of the pooled state.
    """

    def __init__(self, encoder: nn.Module, head: RegressionHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = MaskedMeanPooling()
        self.head = head

    @classmethod
    def under(cls, plan: AdaptationPlan, backbones: BackboneFactory) -> Self:
        """The candidate ``plan`` describes, built on the host from torch's current generator.

        The head is drawn first, so under one seed every mode starts it from the same weights,
        whatever the encoder draws after it.

        Raises:
            UnknownBackboneError: If the plan names pretrained weights the factory does not
                serve.
            LoraTargetNotFoundError: If the plan's low-rank updates name a layer the backbone
                does not have.
        """
        head = RegressionHead(backbones.width)
        if plan.backbone is None:
            encoder = backbones.fresh()
        else:
            encoder = backbones.pretrained(plan.backbone)
        encoder.requires_grad_(plan.mode.trains_backbone_weights)
        if plan.lora is not None:
            LowRankAdaptation(plan.lora).applied_to(encoder)
        return cls(encoder, head)

    def embed(self, batch: TokenTensors) -> Tensor:
        """One state per window, ``[batch, width]``."""
        return self.pooling(self.encoder(*batch.args), batch.padding_mask)

    def forward(self, batch: TokenTensors) -> Tensor:
        return self.head(self.embed(batch))

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]
