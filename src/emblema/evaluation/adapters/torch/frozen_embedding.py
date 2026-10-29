from typing import Self

import torch
from torch import Tensor, nn

from emblema.evaluation.adapters.torch.backbone_factory import BackboneFactory
from emblema.evaluation.adapters.torch.layer_readout import LayerReadout
from emblema.evaluation.adapters.torch.pooling import pooling_module
from emblema.evaluation.domain.detection.encoder_reading import EncoderReading
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.arrays.window_layout import WindowLayout
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class FrozenEmbedding(nn.Module):
    """A frozen encoder and a pooling with no weights: one state per window, nothing trained.

    What a detector reads a window through: the encoder a reading names, pretrained or drawn
    at its initialisation, read at the layer the setting names, its states per token pooled
    by the scheme the reading names. No head, since a detector learns none, and no weight left
    trainable, since nothing here descends a loss.

    Attributes:
        encoder: The backbone, read at the reading's layer.
        pooling: One state per window out of the states per token.
    """

    def __init__(self, encoder: nn.Module, pooling: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling

    @classmethod
    def under(
        cls,
        reading: EncoderReading,
        backbones: BackboneFactory,
        *,
        vocabulary_size: int,
        layout: WindowLayout,
    ) -> Self:
        """The embedding ``reading`` describes, built on the host from torch's current generator.

        The encoder is asked for over ``vocabulary_size``, the channels of the task's corpus,
        and over the feature axis of ``layout``, the way the run lays windows out; a pretrained
        one grows rows for channels it never saw, drawn from the generator, as a probe's does.

        Raises:
            UnknownBackboneError: If the reading names pretrained weights the factory does not
                serve.
            InvalidLayerReadingError: If the reading reads a layer the encoder does not have.
        """
        setting = reading.encoder
        shape = setting.shape
        blocks = backbones.layers if shape is None else shape.layers
        width = (backbones.width if shape is None else shape.width) * setting.layer.width_factor(
            blocks
        )
        if reading.backbone is None:
            encoder = backbones.fresh(
                vocabulary_size=vocabulary_size,
                dropout=0.0,
                value_embedding=setting.value_embedding,
                shape=shape,
                time_encoding=setting.time_encoding,
                input_features=layout.features,
                bins=layout.bins if isinstance(layout, BinnedTokens) else 0,
            )
        else:
            encoder = backbones.pretrained(
                reading.backbone, vocabulary_size=vocabulary_size, dropout=0.0
            )
        encoder.requires_grad_(False)
        if not setting.layer.is_last:
            encoder = LayerReadout(encoder, setting.layer, blocks=blocks)
        return cls(encoder, pooling_module(reading.pooling, width=width))

    def forward(self, batch: TokenTensors) -> Tensor:
        pooled: Tensor = self.pooling(
            self.encoder(*batch.args), batch.padding_mask, batch.timestamps, batch.timeless
        )
        return pooled

    def embed(self, batch: TokenTensors) -> Tensor:
        """One state per window of ``batch``, with no gradient recorded."""
        with torch.no_grad():
            return self.forward(batch)
