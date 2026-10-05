from dataclasses import dataclass
from math import isfinite

from emblema.pretraining.domain.exceptions import InvalidCorpusFractionError


@dataclass(frozen=True, kw_only=True)
class CorpusFraction:
    """How much of one corpus's training side a mixed run reads, where it differs from the rest.

    A mixture of corpora of very different sizes is not read whole on every side: a corpus of a
    billion values is read in part and a corpus of a million in full, or the small one is a
    rounding error of every epoch. The fraction stated here replaces, for its corpus alone, the
    share the configuration states for the mixture; a corpus left unstated is read at that share,
    so a run without this knob is the run it always was.

    Invariants: the corpus is named without surrounding whitespace; the fraction lies in
    ``(0, 1]`` — a whole corpus may be stated, where the mixture's share is less than whole.

    Attributes:
        corpus: The corpus, as the experiment names it.
        fraction: Share of its training units the run reads.
    """

    corpus: str
    fraction: float

    def __post_init__(self) -> None:
        if not self.corpus or self.corpus != self.corpus.strip():
            raise InvalidCorpusFractionError("corpus must be named without surrounding whitespace")
        if not isfinite(self.fraction) or not 0.0 < self.fraction <= 1.0:
            raise InvalidCorpusFractionError(f"fraction must lie in (0, 1], got {self.fraction}")
