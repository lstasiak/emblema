from dataclasses import dataclass
from typing import Self

from emblema.pretraining.domain.backbone.backbone import Backbone
from emblema.pretraining.domain.exceptions import (
    InvalidPretrainingOrderError,
    PretrainingOrderRejectedError,
)
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.run_signature import RunSignature
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class PretrainingOrder:
    """What a machine outside this system is asked to run, complete enough to run and to prove.

    Everything the run needs — the configuration and the manifest of the corpus — and everything
    the result is held to: which backbone it is for, the revision the code is to be at, and the
    signature the machine must find when it reads the corpus, so that one running other code or
    reading the wrong data stops before it trains rather than after.

    Invariants: the run name and the commit are non-empty without surrounding whitespace.

    Attributes:
        backbone: The backbone the run is for.
        configuration: The experiment to run.
        manifest: The published corpus to read.
        run: Name of the run within the experiment.
        git_commit: Revision of the code the run is to be made with.
        signature: What the run is: the configuration over the corpus as read when ordered.
    """

    backbone: BackboneId
    configuration: ExperimentConfiguration
    manifest: ArtifactRef
    run: str
    git_commit: str
    signature: RunSignature

    def __post_init__(self) -> None:
        for label, text in (("run", self.run), ("git_commit", self.git_commit)):
            if not text or text != text.strip():
                raise InvalidPretrainingOrderError(
                    f"{label} must be non-empty without surrounding whitespace"
                )

    @classmethod
    def of(cls, backbone: Backbone) -> Self:
        """The order that produces ``backbone``: its run, as it was registered."""
        return cls(
            backbone=backbone.id,
            configuration=backbone.configuration,
            manifest=backbone.input.manifest,
            run=backbone.run,
            git_commit=backbone.git_commit,
            signature=backbone.signature,
        )

    def require_read(self, corpus: TrainingCorpus) -> None:
        """Stop a run over a corpus other than the one ordered, before it trains.

        Raises:
            PretrainingOrderRejectedError: If the configuration over ``corpus`` is not the run
                the order signed.
        """
        found = RunSignature.of(self.configuration, corpus)
        if found != self.signature:
            raise PretrainingOrderRejectedError(
                f"the corpus read signs run {found}, the order is for run {self.signature}"
            )

    def require_commit(self, git_commit: str) -> None:
        """Stop a run on code other than the code ordered, before it trains.

        Raises:
            PretrainingOrderRejectedError: If ``git_commit`` is not the revision the order names.
        """
        if git_commit != self.git_commit:
            raise PretrainingOrderRejectedError(
                f"this machine runs commit {git_commit}, the order is for commit {self.git_commit}"
            )
