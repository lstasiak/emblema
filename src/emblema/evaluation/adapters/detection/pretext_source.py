from collections.abc import Sequence
from typing import Protocol

from emblema.evaluation.adapters.detection.masked_cells import MaskedCells
from emblema.evaluation.domain.detection.hidden_cells_choice import HiddenCellsChoice
from emblema.shared.adapters.arrays.joined_windows import JoinedWindow
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow


class PretextSource(Protocol):
    """Where a detector asks a pretrained backbone what it would fill in, not knowing whose it is.

    A seam of the adapter beside the factory of encoders, implemented by the process that
    composes both contexts: Evaluation may not import the objective that trained the backbone,
    nor read the register that says how it was trained. What crosses is a window's cells under a
    draw of the backbone's own masks and, where asked, its decoder's answers; never a checkpoint.
    A draw is seeded by the caller, one seed per window, and depends on nothing else, so a
    detector that learnt nothing can be asked over exactly the cells the backbone was. Asked
    for the tail, the source hides the last span of every channel instead of drawing. A
    backbone over a history is asked over windows joined with theirs, as its run joined them.
    """

    @property
    def bins(self) -> int:
        """Spans the backbone lays a channel over; zero for one token per reading."""
        ...

    @property
    def history(self) -> int:
        """Published windows one of the backbone's windows spans, its own the last."""
        ...

    @property
    def huber_delta(self) -> float:
        """The knee of the loss the backbone scored a miss by; zero for the squared error."""
        ...

    def masked(
        self,
        weights: ArtifactRef,
        windows: Sequence[TokenWindow | JoinedWindow],
        seeds: Sequence[int],
        *,
        answered: bool,
        device: str,
        hidden: HiddenCellsChoice = HiddenCellsChoice.OWN,
    ) -> list[MaskedCells]:
        """One draw of the backbone's masks over each window, from the seed beside it, or its tail.

        Raises:
            UnknownBackboneError: If this source does not serve those weights.
            UnreadablePretextError: If the backbone reads no cells, its stored model does not say
                what it hid, a window holds a channel it learnt no row for, or the windows are
                not joined with their history where the backbone read one, or the reverse.
            ValueError: If the windows and seeds are not paired, or a seed lies outside
                ``[0, 2**63)``.
        """
        ...
