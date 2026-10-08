import torch
from torch import Tensor, nn

from emblema.evaluation.adapters.torch.layered_encoder import LayeredEncoder
from emblema.evaluation.domain.exceptions import InvalidLayerReadingError
from emblema.evaluation.domain.transfer.layer_reading import LayerCombination, LayerReading


class LayerReadout(nn.Module):
    """An encoder read at the layer a plan names, called as the encoder is.

    It holds no weights and changes nothing the encoder computes, so a candidate holds it where
    it held the encoder and the pooling and head downstream see one state per token, as wide as
    the reading makes it. Every layer is computed and one or all are kept, so it is meant for an
    encoder that does not train.

    Attributes:
        encoder: The encoder read.
        reading: Which layer, or which combination of the blocks, is read.
    """

    def __init__(self, encoder: nn.Module, reading: LayerReading, *, blocks: int) -> None:
        """Read ``encoder`` of ``blocks`` blocks as ``reading`` says.

        Raises:
            InvalidLayerReadingError: If the encoder cannot answer with its layers, or the layer
                named lies above its last block.
        """
        super().__init__()
        if not isinstance(encoder, LayeredEncoder):
            raise InvalidLayerReadingError(
                f"a {type(encoder).__name__} answers with its last layer only"
            )
        if reading.layer is not None and reading.layer > blocks:
            raise InvalidLayerReadingError(
                f"the encoder has {blocks} blocks, so layer {reading.layer} is not one of its "
                f"layers 0 to {blocks}"
            )
        self.encoder = encoder
        self.reading = reading

    def forward(
        self,
        features: Tensor,
        channel_ids: Tensor,
        timestamps: Tensor,
        timeless: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        every: Tensor = self.encoder.layer_states(
            features, channel_ids, timestamps, timeless, padding_mask
        )
        match self.reading.combination:
            case LayerCombination.MEAN:
                return every[1:].mean(dim=0)
            case LayerCombination.CONCATENATION:
                return torch.cat(every[1:].unbind(dim=0), dim=-1)
            case None:
                return every[-1 if self.reading.layer is None else self.reading.layer]
