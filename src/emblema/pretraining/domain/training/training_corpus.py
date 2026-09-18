from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from emblema.pretraining.domain.exceptions import InvalidTrainingCorpusError
from emblema.pretraining.domain.training.training_corpus_shape import TrainingCorpusShape
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow


@dataclass(frozen=True, kw_only=True)
class TrainingCorpus:
    """The windows a run learns from and the windows it is scored on, under one vocabulary.

    Both sides are named here rather than split by whoever trains, so that a run cannot read the
    validation side by forgetting an argument. The vocabulary is carried by the names of its
    channels, in identifier order, because it is what the channel table is built for and what
    says whether two corpora can be trained on together: it belongs to the corpus the windows
    were published with, not to the model.

    Invariants: those of its shape — the name is non-empty and carries no surrounding whitespace,
    both sides hold at least one window — and the vocabulary holds at least one channel, each
    named without surrounding whitespace and none twice. That every token's channel is within
    the vocabulary is not checked: the scan would cost the pass it protects, and a window that
    says otherwise is refused by the model's channel table on the first batch.

    Attributes:
        name: The corpus the windows were published from.
        checksum: Checksum of the artifact the windows were read out of. What the windows are,
            rather than how many of them there are: two publications of one corpus that differ in
            a window, a normalisation or a split are different data under the same name, and a
            run of one must not be picked up on the other.
        training: Windows the gradient is taken over.
        validation: Windows the run is scored on and never trains on.
        channels: Names of the published vocabulary's channels, entry ``i`` being the channel
            the tokens carry as identifier ``i + 1``.
    """

    name: str
    checksum: Checksum
    training: Sequence[TokenWindow]
    validation: Sequence[TokenWindow]
    channels: tuple[str, ...]

    def __post_init__(self) -> None:
        for channel in self.channels:
            if not channel or channel != channel.strip():
                raise InvalidTrainingCorpusError(
                    f"every channel must be named without surrounding whitespace, got {channel!r}"
                )
        if len(set(self.channels)) != len(self.channels):
            raise InvalidTrainingCorpusError(
                f"a channel is named twice in the vocabulary of {self.name!r}"
            )
        # Building the shape runs the invariants: a corpus whose shape is not a shape is no corpus.
        _ = self.shape

    @property
    def vocabulary_size(self) -> int:
        """Channels the published vocabulary holds."""
        return len(self.channels)

    def continues(self, earlier: Self) -> bool:
        """Whether this corpus was published continuing the vocabulary of ``earlier``.

        True when the earlier corpus's channel names are a prefix of this one's, which is what a
        publication that took the earlier manifest as the vocabulary to continue leaves behind;
        a corpus published from scratch, or one that renamed an earlier channel, does not.
        """
        return self.channels[: len(earlier.channels)] == earlier.channels

    @property
    def shape(self) -> TrainingCorpusShape:
        """The corpus without its windows: what a run's signature is computed from.

        Raises:
            InvalidTrainingCorpusError: If the name is blank, a side is empty or the vocabulary
                holds no channel.
        """
        return TrainingCorpusShape(
            name=self.name,
            checksum=self.checksum,
            training_windows=len(self.training),
            validation_windows=len(self.validation),
            vocabulary_size=self.vocabulary_size,
        )
