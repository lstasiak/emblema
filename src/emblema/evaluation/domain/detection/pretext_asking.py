from dataclasses import dataclass
from typing import ClassVar

from emblema.evaluation.domain.exceptions import InvalidDetectorRecipeError
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class PretextAsking:
    """Which backbone's masks a window is hidden under, and how many times.

    A detector scored by the error of what is filled in asks under a backbone's own masks: the
    pretrained backbone fills the hidden cells in, and an answer that learnt nothing fills the
    same cells in from the same visible ones. A window is hidden ``draws`` times and its error
    is the mean over them, since one draw hides too few cells to rank a window by.

    Invariants: a window is hidden at least once.

    Attributes:
        backbone: Artifact of the pretrained weights whose masks hide the cells.
        draws: How many draws of the masks each window is hidden under.
    """

    STANDARD_DRAWS: ClassVar[int] = 8

    backbone: ArtifactRef
    draws: int = STANDARD_DRAWS

    def __post_init__(self) -> None:
        if self.draws < 1:
            raise InvalidDetectorRecipeError(
                f"a window is hidden at least once, got {self.draws} draws"
            )
