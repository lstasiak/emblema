from typing import Any, Self

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from emblema.evaluation.adapters.torch.backbone_factory import BackboneFactory
from emblema.evaluation.adapters.torch.clipped_values import ClippedValues
from emblema.evaluation.adapters.torch.layer_readout import LayerReadout
from emblema.evaluation.adapters.torch.low_rank_adaptation import LowRankAdaptation
from emblema.evaluation.adapters.torch.pooling import pooling_module
from emblema.evaluation.adapters.torch.regression_head import RegressionHead
from emblema.evaluation.adapters.torch.summary_branch import SummaryBranch
from emblema.evaluation.domain.exceptions import InvalidAdaptationPlanError
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.arrays.raw_tokens import RawTokens
from emblema.shared.adapters.arrays.window_layout import WindowLayout
from emblema.shared.adapters.tensors.grown_parameters import GrownParameters
from emblema.shared.adapters.tensors.token_tensors import TokenTensors

# One group of an optimiser's parameters and the rate they learn at: torch takes a group as an
# untyped mapping of the parameters and their options, so Any is what it is.
ParameterGroup = dict[str, Any]


class AdaptedBackbone(nn.Module):
    """The candidate a plan makes: encoder, pooling and head, the mode's weights left free.

    The head is new in every mode and the candidate is called the same way whatever the mode, so
    a comparison between modes is a comparison between backbones. The pooling is the plan's: a
    variant that turns it is the same backbone under another head, and whatever weights the
    pooling has of its own train under every mode, as the head does.

    Where the plan reads the window's statistics per channel, a branch turns them into a state
    the head reads beside the pooled one, or in its place; the encoder is then built as ever but
    never run, and none of its weights train.

    Attributes:
        encoder: The backbone, with whatever the mode left trainable; fed bounded values where
            the plan clips them, and read at the layer the plan names.
        pooling: One state per window out of the states per token, as the plan named it.
        head: The task's answer out of the pooled state.
        summaries: The branch over the statistics per channel, where the plan reads them.
        reads_tokens: Whether the encoder is run at all.
    """

    def __init__(
        self,
        encoder: nn.Module,
        pooling: nn.Module,
        head: RegressionHead,
        *,
        summaries: SummaryBranch | None = None,
        reads_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.head = head
        self.summaries = summaries
        self.reads_tokens = reads_tokens

    @classmethod
    def under(
        cls,
        plan: AdaptationPlan,
        backbones: BackboneFactory,
        *,
        vocabulary_size: int,
        starting_at: float,
        summary_rows: NDArray[np.float64] | None = None,
        layout: WindowLayout | None = None,
    ) -> Self:
        """The candidate ``plan`` describes, built on the host from torch's current generator.

        The head is drawn first, so under one seed every mode starts it from the same weights,
        whatever the encoder draws after it; its bias starts at ``starting_at``, where the task's
        link places a predictor that knows nothing of the labels the run holds. The encoder is
        asked for over ``vocabulary_size``, the channels of the task's corpus, and over the
        feature axis of ``layout``, the way the run lays windows out, which also says which
        features a clip bounds; the raw readings unless given. Rows it grew for channels it was
        never trained on are trained under every mode, as the head is, because
        there is nothing pretrained in them to freeze. A branch over the statistics per channel is
        drawn after everything else and prepared over ``summary_rows``, the statistics of the
        windows the run draws, so a plan that reads none draws exactly what it always did.

        Raises:
            InvalidAdaptationPlanError: If the plan reads the statistics per channel and none are
                given, or some are given to a plan that does not read them.
            UnknownBackboneError: If the plan names pretrained weights the factory does not
                serve.
            LoraTargetNotFoundError: If the plan's low-rank updates name a layer the backbone
                does not have.
            InvalidLayerReadingError: If the plan reads a layer the encoder does not have.
        """
        shape = plan.encoder.shape
        blocks = backbones.layers if shape is None else shape.layers
        width = (backbones.width if shape is None else shape.width) * (
            plan.encoder.layer.width_factor(blocks)
        )
        reading = plan.encoder
        if reading.reads_summaries != (summary_rows is not None):
            raise InvalidAdaptationPlanError(
                f"a plan reading summaries {reading.summaries} is given "
                + ("none" if summary_rows is None else "some")
            )
        laid_out = RawTokens() if layout is None else layout
        read = width * plan.pooling.width_factor if reading.reads_tokens else 0
        head = RegressionHead(
            read + (width if reading.reads_summaries else 0), starting_at=starting_at
        )
        pooling = pooling_module(plan.pooling, width=width)
        dropout = plan.encoder.dropout
        if plan.backbone is None:
            encoder = backbones.fresh(
                vocabulary_size=vocabulary_size,
                dropout=dropout,
                value_embedding=plan.encoder.value_embedding,
                shape=shape,
                time_encoding=plan.encoder.time_encoding,
                input_features=laid_out.features,
                bins=laid_out.bins if isinstance(laid_out, BinnedTokens) else 0,
            )
        else:
            encoder = backbones.pretrained(
                plan.backbone, vocabulary_size=vocabulary_size, dropout=dropout
            )
        encoder.requires_grad_(plan.mode.trains_backbone_weights)
        for module in encoder.modules():
            if isinstance(module, GrownParameters):
                for parameter in module.grown_parameters():
                    parameter.requires_grad_(True)
        if plan.lora is not None:
            LowRankAdaptation(plan.lora).applied_to(encoder)
        if not plan.encoder.layer.is_last:
            encoder = LayerReadout(encoder, plan.encoder.layer, blocks=blocks)
        if plan.encoder.value_clip is not None:
            encoder = ClippedValues(encoder, plan.encoder.value_clip, laid_out.value_features)
        if not reading.reads_tokens:
            encoder.requires_grad_(False)
        return cls(
            encoder,
            pooling,
            head,
            summaries=(
                None if summary_rows is None else SummaryBranch.fitted(summary_rows, width=width)
            ),
            reads_tokens=reading.reads_tokens,
        )

    def embed(self, batch: TokenTensors, summaries: Tensor | None = None) -> Tensor:
        """One state per window: the pooled states, the statistics' state, or both side by side.

        Args:
            batch: The windows' tokens.
            summaries: The windows' statistics per channel, one row each, where the candidate
                reads them.

        Raises:
            InvalidAdaptationPlanError: If the candidate reads the statistics and is not given
                them.
        """
        if self.summaries is None:
            return self._pooled(batch)
        if summaries is None:
            raise InvalidAdaptationPlanError(
                "this candidate reads the windows' statistics and was given none"
            )
        read: Tensor = self.summaries(summaries)
        if not self.reads_tokens:
            return read
        return torch.cat((self._pooled(batch), read), dim=-1)

    def forward(self, batch: TokenTensors, summaries: Tensor | None = None) -> Tensor:
        return self.head(self.embed(batch, summaries))

    def _pooled(self, batch: TokenTensors) -> Tensor:
        pooled: Tensor = self.pooling(
            self.encoder(*batch.args), batch.padding_mask, batch.timestamps, batch.timeless
        )
        return pooled

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def parameter_groups(self, rate: float, *, encoder_share: float) -> list[ParameterGroup]:
        """The trainable parameters as the optimiser takes them, in two groups by their rate.

        The head, the pooling's own weights and the rows grown for the task's new channels learn
        at ``rate``; the encoder's other weights at ``rate × encoder_share``, where the plan
        slows them. One group holds everything where the share is one, so a run under the
        standard regime hands the optimiser what it always did.
        """
        trainable = self.trainable_parameters()
        if encoder_share >= 1.0:
            return [{"params": trainable, "lr": rate}]
        grown = {
            id(parameter)
            for module in self.encoder.modules()
            if isinstance(module, GrownParameters)
            for parameter in module.grown_parameters()
        }
        slowed = {id(p) for p in self.encoder.parameters() if p.requires_grad} - grown
        groups: list[ParameterGroup] = [
            {"params": [p for p in trainable if id(p) not in slowed], "lr": rate},
            {"params": [p for p in trainable if id(p) in slowed], "lr": rate * encoder_share},
        ]
        # The optimiser refuses a group with nothing in it.
        return [group for group in groups if group["params"]]
