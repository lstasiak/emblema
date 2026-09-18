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
from emblema.pretraining.domain.training.training_mixture import TrainingMixture
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class PretrainingOrder:
    """What a machine outside this system is asked to run, complete enough to run and to prove.

    Everything the run needs — the configuration and the manifests of the corpora, in the order
    they are read — and everything the result is held to: which backbone it is for, the revision
    the code is to be at, and the signature the machine must find when it reads the corpora, so
    that one running other code or reading the wrong data stops before it trains rather than
    after.

    Invariants: the run name and the commit are non-empty without surrounding whitespace; at
    least one manifest is named, none twice.

    Attributes:
        backbone: The backbone the run is for.
        configuration: The experiment to run.
        manifests: The published corpora to read, in the order their vocabulary was chained.
        run: Name of the run within the experiment.
        git_commit: Revision of the code the run is to be made with.
        signature: What the run is: the configuration over the corpora as read when ordered.
    """

    backbone: BackboneId
    configuration: ExperimentConfiguration
    manifests: tuple[ArtifactRef, ...]
    run: str
    git_commit: str
    signature: RunSignature

    def __post_init__(self) -> None:
        for label, text in (("run", self.run), ("git_commit", self.git_commit)):
            if not text or text != text.strip():
                raise InvalidPretrainingOrderError(
                    f"{label} must be non-empty without surrounding whitespace"
                )
        if not self.manifests:
            raise InvalidPretrainingOrderError("an order names at least one manifest")
        if len(set(self.manifests)) != len(self.manifests):
            raise InvalidPretrainingOrderError("an order names no manifest twice")

    @classmethod
    def of(cls, backbone: Backbone) -> Self:
        """The order that produces ``backbone``: its run, as it was registered."""
        return cls(
            backbone=backbone.id,
            configuration=backbone.configuration,
            manifests=tuple(read.manifest for read in backbone.inputs),
            run=backbone.run,
            git_commit=backbone.git_commit,
            signature=backbone.signature,
        )

    def require_read(self, mixture: TrainingMixture) -> None:
        """Stop a run over data other than the data ordered, before it trains.

        Raises:
            PretrainingOrderRejectedError: If the configuration over ``mixture`` is not the run
                the order signed.
        """
        found = RunSignature.of(self.configuration, mixture)
        if found != self.signature:
            raise PretrainingOrderRejectedError(
                f"the corpora read sign run {found}, the order is for run {self.signature}"
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
