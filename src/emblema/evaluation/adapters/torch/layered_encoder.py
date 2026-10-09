from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class LayeredEncoder(Protocol):
    """An encoder that also answers with every layer's states, not only its last.

    Evaluation may not import the encoder's class, so a head that reads another layer asks the
    module for this by name. The states come normalised as the last is, the embedding first,
    ``[blocks + 1, batch, tokens, width]``.
    """

    def layer_states(
        self,
        features: Tensor,
        channel_ids: Tensor,
        timestamps: Tensor,
        timeless: Tensor,
        padding_mask: Tensor,
    ) -> Tensor: ...
