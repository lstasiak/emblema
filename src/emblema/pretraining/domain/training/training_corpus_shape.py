from dataclasses import dataclass

from emblema.pretraining.domain.exceptions import InvalidTrainingCorpusError
from emblema.shared.kernel.checksums import Checksum


@dataclass(frozen=True, kw_only=True)
class TrainingCorpusShape:
    """What a run's corpus is, without the windows: enough to tell two runs' data apart.

    A run is identified by its configuration and its corpus, and the corpus by the checksum of
    the artifact its windows came out of and by how much of it was read — the same block read for
    half its units is another run. This is that description on its own, so that a run that
    happened on another machine can be held against the corpus this process would have read
    without the windows being here.

    Invariants: the name is non-empty and carries no surrounding whitespace; both sides hold at
    least one window; the vocabulary holds at least one channel.

    Attributes:
        name: The corpus the windows were published from.
        checksum: Checksum of the artifact the windows were read out of.
        training_windows: Windows the gradient is taken over.
        validation_windows: Windows the run is scored on.
        vocabulary_size: Channels the published vocabulary holds.
    """

    name: str
    checksum: Checksum
    training_windows: int
    validation_windows: int
    vocabulary_size: int

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise InvalidTrainingCorpusError(
                "name must be non-empty without surrounding whitespace"
            )
        for label, count in (
            ("training", self.training_windows),
            ("validation", self.validation_windows),
        ):
            if count < 1:
                raise InvalidTrainingCorpusError(f"the {label} side must hold a window")
        if self.vocabulary_size < 1:
            raise InvalidTrainingCorpusError(
                f"vocabulary_size must be positive, got {self.vocabulary_size}"
            )
