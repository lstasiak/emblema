from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from torch import nn


@runtime_checkable
class GrownParameters(Protocol):
    """A module that grew parameters for a task beyond the ones it was trained with.

    The context that adapts an encoder freezes or frees its weights by transfer mode, and may not
    import the context that built it; what it can ask any module is whether some of its parameters
    are new. A parameter grown for the task carries nothing from pretraining, so there is nothing
    in it to keep or to freeze: like the head, it is new in every mode and trained in every mode.
    """

    def grown_parameters(self) -> Iterable[nn.Parameter]:
        """The parameters grown for the task, none of which pretraining set."""
        ...
