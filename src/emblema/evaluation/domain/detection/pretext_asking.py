from dataclasses import dataclass
from typing import ClassVar

from emblema.evaluation.domain.detection.hidden_cells_choice import HiddenCellsChoice
from emblema.evaluation.domain.exceptions import InvalidDetectorRecipeError
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class PretextAsking:
    """Which backbone a window is hidden for, which of its cells, and how many times.

    A detector scored by the error of what is filled in asks under a backbone: the pretrained
    backbone fills the hidden cells in, and an answer that learnt nothing fills the same cells in
    from the same visible ones. Under the backbone's own masks a window is hidden ``draws``
    times and its error is the mean over them, since one draw hides too few cells to rank a
    window by; with its tail hidden there is nothing to draw, so it is hidden once.

    Invariants: a window is hidden at least once; a tail is hidden exactly once.

    Attributes:
        backbone: Artifact of the pretrained weights the cells are filled in by.
        draws: How many draws of the masks each window is hidden under.
        hidden: Which cells are hidden: the backbone's own draw, or the tail.
    """

    STANDARD_DRAWS: ClassVar[int] = 8

    backbone: ArtifactRef
    draws: int = STANDARD_DRAWS
    hidden: HiddenCellsChoice = HiddenCellsChoice.OWN

    def __post_init__(self) -> None:
        if self.draws < 1:
            raise InvalidDetectorRecipeError(
                f"a window is hidden at least once, got {self.draws} draws"
            )
        if self.hidden is HiddenCellsChoice.TAIL and self.draws != 1:
            raise InvalidDetectorRecipeError(
                f"a tail is hidden once, there being nothing to draw, got {self.draws} draws"
            )
