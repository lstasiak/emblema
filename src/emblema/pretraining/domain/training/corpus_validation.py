from dataclasses import dataclass
from math import isfinite

from emblema.pretraining.domain.exceptions import InvalidCorpusValidationError


@dataclass(frozen=True, kw_only=True)
class CorpusValidation:
    """What one corpus of a run's held-out side cost an epoch, beside what nothing to learn costs.

    A run over several corpora scored by one number is scored by the corpus whose values are
    largest: the satellite corpus's held-out months err five times worse under the channel mean
    than a classical corpus's, so a mixed loss is theirs and a run stopped by it is stopped by
    them. Each corpus is therefore reported on its own, and with it the trivial predictor's loss
    over the same tokens under the same reading, so that the relative loss — one wherever nothing
    was learnt — is a division here rather than a lookup in a note.

    Invariants: the corpus is named without surrounding whitespace; the epoch scored a token of
    it; the losses are finite and not negative; the trivial predictor's is positive, since a
    held-out side it gets exactly right holds nothing to learn and no relative loss over it is a
    number.

    Attributes:
        corpus: The corpus the windows were published from.
        tokens: Hidden tokens of that corpus the epoch scored.
        loss: The run's reading over them.
        trivial: The channel-mean predictor's loss over the same tokens, under the same reading.
    """

    corpus: str
    tokens: int
    loss: float
    trivial: float

    def __post_init__(self) -> None:
        if not self.corpus or self.corpus != self.corpus.strip():
            raise InvalidCorpusValidationError(
                "corpus must be non-empty without surrounding whitespace"
            )
        if self.tokens < 1:
            raise InvalidCorpusValidationError(
                f"an epoch that scored no token of {self.corpus!r} validates nothing of it, "
                f"got {self.tokens}"
            )
        if not isfinite(self.loss) or self.loss < 0.0:
            raise InvalidCorpusValidationError(
                f"loss must be finite and not negative, got {self.loss}"
            )
        if not isfinite(self.trivial) or self.trivial <= 0.0:
            raise InvalidCorpusValidationError(
                f"the trivial predictor's loss must be finite and positive, got {self.trivial}"
            )

    @property
    def relative(self) -> float:
        """The run's loss over the trivial predictor's; one is nothing learnt, and less is some."""
        return self.loss / self.trivial
