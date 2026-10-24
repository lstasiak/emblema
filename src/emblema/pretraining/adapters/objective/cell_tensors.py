from dataclasses import dataclass

from torch import Tensor

from emblema.shared.adapters.tensors.token_tensors import TokenTensors


@dataclass(frozen=True)
class CellTensors(TokenTensors):
    """A batch of windows laid out as cells twice: from every reading, and from the visible ones.

    The five tensors of a ``TokenTensors`` hold the full layout, cell for cell: what the masks
    were drawn over and what a hidden cell's prediction is scored against. ``visible`` holds the
    features the same cells have when the readings of the hidden cells are left out, so an empty
    cell beside a hidden one carries forward and counts from what remains visible. The two stand
    position for position, and the encoder is fed ``visible`` with the hidden positions as
    padding; it never reads the full features. Moved and retyped as one, so a run on any device
    scores the cells it laid out on the host.

    Attributes:
        visible: ``[batch, tokens, features]`` — the cells' features with the hidden cells'
            readings left out of the layout.
    """

    visible: Tensor
