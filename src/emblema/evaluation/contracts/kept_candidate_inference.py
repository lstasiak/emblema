from collections.abc import Sequence
from typing import Protocol

from emblema.evaluation.contracts.kept_representation import KeptRepresentation
from emblema.shared.kernel.tokens import TokenWindow


class KeptCandidateInference(Protocol):
    """Answers windows with a candidate a campaign kept, from a form only this context reads.

    An open host service of the Evaluation context. A classical candidate is kept in the form
    its runtime fitted — trees over a window's statistics, convolutions over a grid — and the
    reading of a window that form expects is this context's method, stated in its adapters and
    nowhere else. Another context that serves such a candidate cannot import those adapters and
    must not re-implement the reading, so this context publishes the service and the process
    that serves composes the adapter in. The neural candidate needs none of this: the graph it
    is kept as carries its own reading and runs wherever a graph runs.

    The bytes arrive already read from the store and verified, so what this takes is what the
    manifest says the form is, the bytes, and the windows to answer.
    """

    def reads(self, format: str) -> bool:
        """Whether this service answers from a form stored as ``format``."""
        ...

    def answer(
        self,
        form: KeptRepresentation,
        content: bytes,
        windows: Sequence[TokenWindow],
        *,
        channels: int,
    ) -> tuple[float, ...]:
        """The candidate's answer for every window, in the task's unit, in the order given.

        Args:
            form: The form the bytes are, as the kept candidate's manifest names it.
            content: The bytes of that form.
            windows: What to answer, at least one.
            channels: How many channels the vocabulary of the candidate's corpus runs to, which
                a reading laid out channel by channel is as wide as.

        Raises:
            InvalidKeptRepresentationError: If the form is not one this reads, or the bytes
                are not what the form says they are.
        """
        ...
