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
from emblema.pretraining.domain.training.training_mixture_shape import TrainingMixtureShape
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class PretrainingResult:
    """What a machine outside this system reports back: the order it took and what the run made.

    It states what the run actually did — the configuration it trained under, the shape of each
    corpus it read, the revision of the code, the checkpoint it was picked up from — rather than
    claiming it did what was ordered, so that acceptance compares facts with the order and says
    which of them differs. A result that differs in the configuration and one that differs in
    the data are refused with different words, because they are fixed in different places.

    Invariants: the commit is non-empty without surrounding whitespace.

    Attributes:
        order: The order fulfilled.
        backbone: The backbone the order was for.
        configuration: The experiment the run trained under.
        mixture: The corpora the run read, as their shapes, in the order read.
        git_commit: Revision of the code the run was made with.
        resumed_from: Checkpoint the run was picked up from; ``None`` for a run from the start.
        outcome: What the run produced.
    """

    order: ArtifactRef
    backbone: BackboneId
    configuration: ExperimentConfiguration
    mixture: TrainingMixtureShape
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
        return RunSignature.of_shape(self.configuration, self.mixture)

    def require_run_of(
        self,
        configuration: ExperimentConfiguration,
        mixture: TrainingMixtureShape,
        resume_from: ArtifactRef | None,
    ) -> None:
        """Refuse the result unless it is the run of ``configuration`` over ``mixture``.

        Raises:
            PretrainingResultRejectedError: Naming what differs — the configuration, the data or
                the checkpoint the run was picked up from.
        """
        self._require_configuration(configuration)
        if self.mixture != mixture:
            raise PretrainingResultRejectedError(
                f"the result read {self._describe(self.mixture)}; the data ordered is "
                f"{self._describe(mixture)}"
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
        can differ only in how many windows each side of a corpus held when it was read.

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
        trained_on = [
            (corpus.name, corpus.checksum, corpus.vocabulary_size)
            for corpus in self.mixture.corpora
        ]
        ordered = [
            (given.corpus, given.block_checksum, given.vocabulary_size) for given in backbone.inputs
        ]
        if trained_on != ordered:
            raise PretrainingResultRejectedError(
                f"the result read {self._describe(self.mixture)}; the backbone was ordered on "
                + ", ".join(
                    f"{name!r} with checksum {checksum} over {size} channels"
                    for name, checksum, size in ordered
                )
            )
        if self.signature != backbone.signature:
            raise PretrainingResultRejectedError(
                f"the result signs run {self.signature}, the backbone was ordered as run "
                f"{backbone.signature}: the same configuration and data, read as "
                + ", ".join(
                    f"{corpus.training_windows} training and {corpus.validation_windows} "
                    f"validation windows of {corpus.name!r}"
                    for corpus in self.mixture.corpora
                )
                + " where the order was signed over other counts"
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

    @classmethod
    def _describe(cls, mixture: TrainingMixtureShape) -> str:
        return ", ".join(cls._describe_corpus(corpus) for corpus in mixture.corpora)

    @staticmethod
    def _describe_corpus(corpus: TrainingCorpusShape) -> str:
        return (
            f"corpus {corpus.name!r} with checksum {corpus.checksum}, "
            f"{corpus.training_windows} training and {corpus.validation_windows} validation "
            f"windows over {corpus.vocabulary_size} channels"
        )

    @staticmethod
    def _picked_up_from(ref: ArtifactRef | None) -> str:
        return "the start" if ref is None else ref.key
