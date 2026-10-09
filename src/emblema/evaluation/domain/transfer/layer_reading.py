from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Self

from emblema.evaluation.domain.exceptions import InvalidLayerReadingError


class LayerCombination(StrEnum):
    """How the states of every block are made into one state per token.

    Attributes:
        MEAN: The mean of the blocks' states, as wide as one of them.
        CONCATENATION: The blocks' states side by side, as many times wider as there are blocks.
    """

    MEAN = "mean"
    CONCATENATION = "concat"


@dataclass(frozen=True)
class LayerReading:
    """Which of the encoder's layers a head reads its states from.

    A probe has only ever read the last block. What a pretext teaches need not lie there: a
    block that predicts hidden readings may hold in its middle what a task wants and spend its
    last layers on the reconstruction. Every layer is read through the encoder's final
    normalisation, so the last one read by index is the state every head read before, and a
    head solved over states scaled per column sees no difference between that normalisation and
    one without weights. Layer zero is the tokens' embedding, before any block; a combination
    reads the blocks, one to the last, and leaves the embedding out.

    Invariants: at most one of a layer and a combination; a layer is not negative.

    Attributes:
        layer: The layer read, zero for the embedding; ``None`` with no combination for the
            last.
        combination: How every block is read together; ``None`` for one layer.
    """

    layer: int | None = None
    combination: LayerCombination | None = None

    LAST: ClassVar[str] = "last"

    def __post_init__(self) -> None:
        if self.layer is not None and self.combination is not None:
            raise InvalidLayerReadingError(
                f"a reading names one layer or a combination, not layer {self.layer} and "
                f"{self.combination}"
            )
        if self.layer is not None and self.layer < 0:
            raise InvalidLayerReadingError(f"a layer is not negative, got {self.layer}")

    @classmethod
    def last(cls) -> Self:
        """The last block's states, as every head read before this was a knob."""
        return cls()

    @classmethod
    def of(cls, text: str) -> Self:
        """The reading ``text`` names: ``last``, a layer's number, ``mean`` or ``concat``.

        A number is read only as it is written back, so a name stored with a reading is the name
        it was asked for.

        Raises:
            InvalidLayerReadingError: If ``text`` names none of them.
        """
        if text == cls.LAST:
            return cls.last()
        if text in tuple(LayerCombination):
            return cls(combination=LayerCombination(text))
        if text.isascii() and text.isdigit() and str(int(text)) == text:
            return cls(layer=int(text))
        raise InvalidLayerReadingError(
            f"a layer is read as {cls.LAST!r}, a layer's number or one of "
            f"{[str(c) for c in LayerCombination]}, not {text!r}"
        )

    @property
    def is_last(self) -> bool:
        return self.layer is None and self.combination is None

    def width_factor(self, blocks: int) -> int:
        """How many times wider than one layer's state this reading is, over ``blocks`` blocks."""
        return blocks if self.combination is LayerCombination.CONCATENATION else 1

    def __str__(self) -> str:
        if self.combination is not None:
            return str(self.combination)
        return self.LAST if self.layer is None else str(self.layer)
