import io
import pickle
import zipfile
from dataclasses import dataclass
from typing import Any, ClassVar, Self

import torch

from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone
from emblema.evaluation.adapters.torch.target_link import TargetLink
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError
from emblema.evaluation.domain.labels.target_kind import TargetKind
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan

# What comes back from handing torch bytes it did not write. The reasons differ and to a caller
# they are one thing: this artifact is not ours. Stated again rather than shared with the
# Pretraining reader of stored models, because a context imports no other context's adapters.
UNREADABLE_BYTES = (
    KeyError,
    IndexError,
    TypeError,
    ValueError,
    RuntimeError,
    EOFError,
    pickle.UnpicklingError,
    zipfile.BadZipFile,
)


@dataclass(frozen=True)
class FittedCandidate:
    """A candidate as one run left it: its weights, and what it takes to build it again.

    The measured form of what a campaign keeps: the state whose answers the campaign scored, and
    the one the inference graph is derived from. The weights alone would not be enough — the same
    numbers mean different things under a different head, a different channel vocabulary or a
    different link to the target — so the plan's parameters travel with them, flattened to scalars
    because that is how they survive a store. A kept candidate's manifest names this form
    ``FORMAT``.

    Attributes:
        parameters: The plan the candidate was made under, flattened to scalars.
        vocabulary_size: How many channels the corpus it was fitted on has.
        link: How the head's output reaches the task's answer; a candidate kept before the
            link was recorded answered a quantity in its scale.
        weights: State of the whole candidate, head included, on the host.
        cell_channels: The measured channels a candidate over the corpus's cells gets a row
            for; empty for one over the window's own channels or over the raw readings.
        cell_statics: The static features such a candidate gets a cell for; empty likewise.
        cell_bins: The spans a candidate over cells cut the window into, which a plan beside a
            backbone over cells does not state; zero over the raw readings.
    """

    FORMAT: ClassVar[str] = "torch-state"

    parameters: dict[str, str | int | float]
    vocabulary_size: int
    link: TargetLink
    weights: dict[str, Any]
    cell_channels: tuple[int, ...] = ()
    cell_statics: tuple[int, ...] = ()
    cell_bins: int = 0

    @classmethod
    def of(
        cls,
        plan: AdaptationPlan,
        candidate: AdaptedBackbone,
        *,
        vocabulary_size: int,
        link: TargetLink,
        cell_channels: tuple[int, ...] = (),
        cell_statics: tuple[int, ...] = (),
        cell_bins: int = 0,
    ) -> Self:
        """The candidate as it ended, with its weights copied to the host.

        The move off the accelerator comes before anything else is done to them: a widening cast
        made on a device without double precision returns zeros rather than failing.
        """
        return cls(
            parameters=plan.parameters(),
            vocabulary_size=vocabulary_size,
            link=link,
            weights={
                key: value.detach().to("cpu") for key, value in candidate.state_dict().items()
            },
            cell_channels=cell_channels,
            cell_statics=cell_statics,
            cell_bins=cell_bins,
        )

    def to_bytes(self) -> bytes:
        """The candidate as the bytes the artifact store keeps."""
        buffer = io.BytesIO()
        torch.save(
            {
                "parameters": self.parameters,
                "vocabulary_size": self.vocabulary_size,
                "target_kind": str(self.link.kind),
                "target_scale": self.link.scale,
                "weights": self.weights,
                "cell_channels": list(self.cell_channels),
                "cell_statics": list(self.cell_statics),
                "cell_bins": self.cell_bins,
            },
            buffer,
        )
        return buffer.getvalue()

    @classmethod
    def read(cls, content: bytes) -> Self:
        """The candidate those bytes hold.

        Raises:
            UnreadableFittedCandidateError: If the bytes are not a fitted candidate of ours.
        """
        try:
            stored = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
            return cls(
                parameters=stored["parameters"],
                vocabulary_size=stored["vocabulary_size"],
                link=TargetLink.named(
                    stored.get("target_kind", TargetKind.CONTINUOUS), stored["target_scale"]
                ),
                weights=stored["weights"],
                cell_channels=tuple(stored.get("cell_channels", ())),
                cell_statics=tuple(stored.get("cell_statics", ())),
                cell_bins=int(stored.get("cell_bins", 0)),
            )
        except UNREADABLE_BYTES as error:
            raise UnreadableFittedCandidateError(
                f"not a fitted candidate this can read: {error}"
            ) from error
