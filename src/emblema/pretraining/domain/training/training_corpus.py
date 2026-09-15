from collections.abc import Sequence
from dataclasses import dataclass

from emblema.pretraining.domain.exceptions import InvalidTrainingCorpusError
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow


@dataclass(frozen=True, kw_only=True)
class TrainingCorpus:
    """The windows a run learns from and the windows it is scored on, under one vocabulary.

    Both sides are named here rather than split by whoever trains, so that a run cannot read the
    validation side by forgetting an argument. The vocabulary size is carried because it is what
    the channel table is built for, and it belongs to the corpus the windows were published with,
    not to the model.

    Invariants: the name is non-empty and carries no surrounding whitespace; both sides hold at
    least one window; the vocabulary holds at least one channel. That every token's channel is
    within the vocabulary is not checked: the scan would cost the pass it protects, and a window
    that says otherwise is refused by the model's channel table on the first batch.

    Attributes:
        name: The corpus the windows were published from.
        checksum: Checksum of the artifact the windows were read out of. What the windows are,
            rather than how many of them there are: two publications of one corpus that differ in
            a window, a normalisation or a split are different data under the same name, and a
            run of one must not be picked up on the other.
        training: Windows the gradient is taken over.
        validation: Windows the run is scored on and never trains on.
        vocabulary_size: Channels the published vocabulary holds.
    """

    name: str
    checksum: Checksum
    training: Sequence[TokenWindow]
    validation: Sequence[TokenWindow]
    vocabulary_size: int

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise InvalidTrainingCorpusError(
                "name must be non-empty without surrounding whitespace"
            )
        for label, windows in (("training", self.training), ("validation", self.validation)):
            if not windows:
                raise InvalidTrainingCorpusError(f"the {label} side must hold a window")
        if self.vocabulary_size < 1:
            raise InvalidTrainingCorpusError(
                f"vocabulary_size must be positive, got {self.vocabulary_size}"
            )
