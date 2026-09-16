from dataclasses import dataclass

from emblema.pretraining.domain.backbone.backbone import Backbone
from emblema.pretraining.domain.exceptions import (
    InvalidPretrainingResultError,
    PretrainingResultRejectedError,
)
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.run_signature import RunSignature
from emblema.pretraining.domain.training.training_corpus_shape import TrainingCorpusShape
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class PretrainingResult:
    """What a machine outside this system reports back: the order it took and what the run made.

    It states what the run actually did — the configuration it trained under, the shape of the
    corpus it read, the revision of the code, the checkpoint it was picked up from — rather than
    claiming it did what was ordered, so that acceptance compares facts with the order and says
    which of them differs. A result that differs in the configuration and one that differs in
    the data are refused with different words, because they are fixed in different places.

    Invariants: the commit is non-empty without surrounding whitespace.

    Attributes:
        order: The order fulfilled.
        backbone: The backbone the order was for.
        configuration: The experiment the run trained under.
        corpus: The corpus the run read, as its shape.
        git_commit: Revision of the code the run was made with.
        resumed_from: Checkpoint the run was picked up from; ``None`` for a run from the start.
        outcome: What the run produced.
    """

    order: ArtifactRef
    backbone: BackboneId
    configuration: ExperimentConfiguration
    corpus: TrainingCorpusShape
    git_commit: str
    resumed_from: ArtifactRef | None
    outcome: TrainingOutcome

    def __post_init__(self) -> None:
        if not self.git_commit or self.git_commit != self.git_commit.strip():
            raise InvalidPretrainingResultError(
                "git_commit must be non-empty without surrounding whitespace"
            )

    @property
    def signature(self) -> RunSignature:
        """What the run was, computed from what the result states rather than copied."""
        return RunSignature.of_shape(self.configuration, self.corpus)

    def require_run_of(
        self,
        configuration: ExperimentConfiguration,
        corpus: TrainingCorpusShape,
        resume_from: ArtifactRef | None,
    ) -> None:
        """Refuse the result unless it is the run of ``configuration`` over ``corpus``.

        Raises:
            PretrainingResultRejectedError: Naming what differs — the configuration, the data or
                the checkpoint the run was picked up from.
        """
        self._require_configuration(configuration)
        if self.corpus != corpus:
            raise PretrainingResultRejectedError(
                f"the result read corpus {self.corpus.name!r} with checksum "
                f"{self.corpus.checksum}, {self.corpus.training_windows} training and "
                f"{self.corpus.validation_windows} validation windows over "
                f"{self.corpus.vocabulary_size} channels; the data ordered is "
                f"{corpus.name!r} with checksum {corpus.checksum}, {corpus.training_windows} "
                f"training and {corpus.validation_windows} validation windows over "
                f"{corpus.vocabulary_size} channels"
            )
        if self.resumed_from != resume_from:
            raise PretrainingResultRejectedError(
                f"the result was picked up from {self._picked_up_from(self.resumed_from)}, "
                f"the run asked for is picked up from {self._picked_up_from(resume_from)}"
            )

    def require_delivery_for(self, backbone: Backbone) -> None:
        """Refuse the result unless it is the delivery ``backbone`` is waiting for.

        What differs is named in the order it is fixed in: the backbone, the code, the
        configuration by parameter, the data by checksum, and last the signature, which by then
        can differ only in how many windows each side of the corpus held when it was read.

        Raises:
            PretrainingResultRejectedError: If the result is for another backbone, was made with
                other code, trained under another configuration, read other data, or is not the
                run the backbone was ordered as.
        """
        if self.backbone != backbone.id:
            raise PretrainingResultRejectedError(
                f"the result is for backbone {self.backbone}, not {backbone.id}"
            )
        if self.git_commit != backbone.git_commit:
            raise PretrainingResultRejectedError(
                f"the result was made at commit {self.git_commit}, "
                f"the backbone was ordered at {backbone.git_commit}"
            )
        self._require_configuration(backbone.configuration)
        ordered = backbone.input
        if (self.corpus.name, self.corpus.checksum, self.corpus.vocabulary_size) != (
            ordered.corpus,
            ordered.block_checksum,
            ordered.vocabulary_size,
        ):
            raise PretrainingResultRejectedError(
                f"the result read corpus {self.corpus.name!r} with checksum "
                f"{self.corpus.checksum} over {self.corpus.vocabulary_size} channels; the "
                f"backbone was ordered on {ordered.corpus!r} with checksum "
                f"{ordered.block_checksum} over {ordered.vocabulary_size} channels"
            )
        if self.signature != backbone.signature:
            raise PretrainingResultRejectedError(
                f"the result signs run {self.signature}, the backbone was ordered as run "
                f"{backbone.signature}: the same configuration and data, read as "
                f"{self.corpus.training_windows} training and "
                f"{self.corpus.validation_windows} validation windows where the order was "
                f"signed over other counts"
            )

    def _require_configuration(self, expected: ExperimentConfiguration) -> None:
        if self.configuration == expected:
            return
        stated, ordered = self.configuration.parameters(), expected.parameters()
        differing = ", ".join(
            f"{key}={stated[key]!r} for {ordered[key]!r}"
            for key in self.configuration.differences_from(expected)
        )
        raise PretrainingResultRejectedError(
            f"the result trained under configuration {self.configuration.name!r} "
            f"with parameters that differ from the ones ordered: {differing}"
        )

    @staticmethod
    def _picked_up_from(ref: ArtifactRef | None) -> str:
        return "the start" if ref is None else ref.key
