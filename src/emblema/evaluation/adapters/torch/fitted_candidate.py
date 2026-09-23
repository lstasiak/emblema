import io
import pickle
import zipfile
from dataclasses import dataclass
from typing import Any, Self

import torch

from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError
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

    This is what a campaign keeps when it keeps anything, and what another context would serve.
    The weights alone would not be enough — the same numbers mean different things under a
    different head, a different channel vocabulary or a different target scale — so the plan's
    parameters travel with them, flattened to scalars because that is how they survive a store.

    Attributes:
        parameters: The plan the candidate was made under, flattened to scalars.
        vocabulary_size: How many channels the corpus it was fitted on has.
        target_scale: What the targets were divided by, so an answer can be read back in the
            task's own unit.
        weights: State of the whole candidate, head included, on the host.
    """

    parameters: dict[str, str | int | float]
    vocabulary_size: int
    target_scale: float
    weights: dict[str, Any]

    @classmethod
    def of(
        cls,
        plan: AdaptationPlan,
        candidate: AdaptedBackbone,
        *,
        vocabulary_size: int,
        target_scale: float,
    ) -> Self:
        """The candidate as it ended, with its weights copied to the host.

        The move off the accelerator comes before anything else is done to them: a widening cast
        made on a device without double precision returns zeros rather than failing.
        """
        return cls(
            parameters=plan.parameters(),
            vocabulary_size=vocabulary_size,
            target_scale=target_scale,
            weights={
                key: value.detach().to("cpu") for key, value in candidate.state_dict().items()
            },
        )

    def to_bytes(self) -> bytes:
        """The candidate as the bytes the artifact store keeps."""
        buffer = io.BytesIO()
        torch.save(
            {
                "parameters": self.parameters,
                "vocabulary_size": self.vocabulary_size,
                "target_scale": self.target_scale,
                "weights": self.weights,
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
                target_scale=stored["target_scale"],
                weights=stored["weights"],
            )
        except UNREADABLE_BYTES as error:
            raise UnreadableFittedCandidateError(
                f"not a fitted candidate this can read: {error}"
            ) from error
