from dataclasses import dataclass

from emblema.pretraining.domain.exceptions import InvalidCorpusPassesError


@dataclass(frozen=True, kw_only=True)
class CorpusPasses:
    """How many times an epoch of a mixed run reads one of its corpora.

    A corpus weighs its share of the optimiser's steps, which is its share of the windows; a
    corpus the mixture reads too rarely to learn is read more than once an epoch, each pass in
    an order of its own, so its share grows by whole passes while every other corpus is read as
    it was. One pass is what every corpus gets and is not stated, so a run without this knob
    is the run it always was.

    Invariants: the corpus is named without surrounding whitespace; the passes are at least two.

    Attributes:
        corpus: The corpus, as the experiment names it.
        passes: How many times an epoch reads it.
    """

    corpus: str
    passes: int

    def __post_init__(self) -> None:
        if not self.corpus or self.corpus != self.corpus.strip():
            raise InvalidCorpusPassesError("corpus must be named without surrounding whitespace")
        if self.passes < 2:
            raise InvalidCorpusPassesError(
                f"a corpus read once is not stated; passes must be at least 2, got {self.passes}"
            )
