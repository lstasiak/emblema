from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.contracts.exceptions import InvalidKeptRepresentationError
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class KeptRepresentation:
    """One form a kept candidate is stored in, and how far it strays from the form measured.

    A candidate a campaign keeps is one thing stored several ways: the state its runtime fitted,
    and whatever was derived from that state for another context to run. Each form is named by
    the format of its bytes, so a reader picks the one it has a reader for and passes over the
    rest without opening them. The deviation belongs to a derived form: the largest absolute
    difference, in the task's unit, between its answers and the measured form's over the windows
    the campaign scored. The measured form carries none.

    Invariants: the format is named; a deviation, where there is one, is finite and not negative.

    Attributes:
        format: What the bytes are, as the code that wrote them names itself.
        artifact: Where the bytes are, and what they hash to.
        deviation: How far this form's answers strayed from the measured form's; ``None`` for
            the measured form itself.
    """

    format: str
    artifact: ArtifactRef
    deviation: float | None

    def __post_init__(self) -> None:
        if not self.format or self.format != self.format.strip():
            raise InvalidKeptRepresentationError("a representation's format must be named")
        if self.deviation is not None and (not isfinite(self.deviation) or self.deviation < 0.0):
            raise InvalidKeptRepresentationError(
                f"a deviation must be finite and not negative, got {self.deviation}"
            )
