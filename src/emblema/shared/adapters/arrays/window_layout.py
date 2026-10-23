from collections.abc import Sequence
from typing import Protocol

from numpy.typing import DTypeLike

from emblema.shared.adapters.arrays.laid_window import LaidWindow
from emblema.shared.adapters.arrays.token_batch import TokenBatch
from emblema.shared.kernel.tokens import TokenWindow


class WindowLayout(Protocol):
    """How a run lays token windows out as the batch an encoder is called with.

    A seam between the adapters that read windows and the encoder, not a port: nothing in a use
    case sees it. The raw layout hands the encoder one token per reading; another may hand it
    tokens of its own making, such as one per channel and span of time, as long as the batch
    keeps the five arrays and their order. The width of the feature axis is the layout's to say.
    A window is laid once and batched as often as a run asks.
    """

    @property
    def features(self) -> int:
        """How wide the feature axis of a batch this lays out is."""
        ...

    @property
    def value_features(self) -> slice:
        """Which features of a token hold its values, for whatever bounds them."""
        ...

    def lay(self, windows: Sequence[TokenWindow]) -> list[LaidWindow]:
        """Every window as the columns this layout gives it, in the order given.

        Raises:
            ValueError: If a window holds something the layout refuses.
        """
        ...

    def batch(self, laid: Sequence[LaidWindow], *, dtype: DTypeLike = ...) -> TokenBatch:
        """``laid`` row by row, padded to the longest.

        Raises:
            ValueError: If no window is given.
        """
        ...
