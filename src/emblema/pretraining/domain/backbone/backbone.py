from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Self

from emblema.pretraining.domain.backbone.pretraining_input import PretrainingInput
from emblema.pretraining.domain.exceptions import (
    BackboneAlreadyDeliveredError,
    InvalidBackboneError,
)
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.run_signature import RunSignature
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.timestamps import UtcDateTime


@dataclass(frozen=True, kw_only=True)
class Backbone:
    """An encoder pretrained under one configuration over published corpora, and its weights.

    Ordered before it is trained and ready once its weights are registered, because the run that
    produces them may happen on a machine this system only hands work to: the order is what that
    machine is asked to do, and what comes back is held to it. Immutable — delivering returns the
    backbone with its artifact, and a backbone that is ready never changes; another run is another
    backbone.

    Invariants: the run name and the commit are non-empty without surrounding whitespace; at
    least one corpus was read, none twice, and their vocabularies do not shrink along the order,
    since each continues the one before; the result, the artifact and the time of delivery come
    together or not at all; a delivery is never earlier than the order.

    Attributes:
        id: Identity of the backbone.
        configuration: The experiment it was trained under.
        inputs: The published corpora it was trained on, in the order their vocabulary was
            chained; a backbone over one corpus has one.
        run: Name of the run within the experiment; with the experiment's name, what the backbone
            is called.
        git_commit: Revision of the code the run was ordered at, and made with.
        signature: What the run is: the configuration over the corpora as read at ordering.
        ordered_at: When the run was ordered.
        result: The result the weights were delivered with, which names the order it fulfilled,
            the checkpoint the run was picked up from and every epoch; ``None`` while the order
            is open.
        artifact: Where the weights are, once delivered; ``None`` while the order is open.
        delivered_at: When the weights were registered; ``None`` while the order is open.
    """

    id: BackboneId
    configuration: ExperimentConfiguration
    inputs: tuple[PretrainingInput, ...]
    run: str
    git_commit: str
    signature: RunSignature
    ordered_at: UtcDateTime
    result: ArtifactRef | None = None
    artifact: ArtifactRef | None = None
    delivered_at: UtcDateTime | None = None

    def __post_init__(self) -> None:
        for label, text in (("run", self.run), ("git_commit", self.git_commit)):
            if not text or text != text.strip():
                raise InvalidBackboneError(
                    f"{label} must be non-empty without surrounding whitespace"
                )
        if not self.inputs:
            raise InvalidBackboneError("a backbone is trained on at least one corpus")
        names = [read.corpus for read in self.inputs]
        if len(set(names)) != len(names):
            raise InvalidBackboneError(f"a corpus is read twice: {names}")
        sizes = [read.vocabulary_size for read in self.inputs]
        if any(later < earlier for earlier, later in pairwise(sizes)):
            raise InvalidBackboneError(
                f"the vocabularies of the corpora do not shrink along the order, got {sizes}"
            )
        delivery = (self.result is None, self.artifact is None, self.delivered_at is None)
        if any(delivery) and not all(delivery):
            raise InvalidBackboneError(
                "the result, the artifact and the time of delivery come together"
            )
        if self.delivered_at is not None and self.delivered_at < self.ordered_at:
            raise InvalidBackboneError("a backbone is not delivered before it is ordered")

    @property
    def name(self) -> str:
        """What the backbone is called: the experiment's name and the run's."""
        return f"{self.configuration.name}/{self.run}"

    @property
    def is_ready(self) -> bool:
        return self.artifact is not None

    @property
    def seed(self) -> int:
        return self.configuration.budget.seed

    @property
    def vocabulary_size(self) -> int:
        """Channels of the vocabulary the run trained under: the last corpus's."""
        return self.inputs[-1].vocabulary_size

    @property
    def parameter_count(self) -> int:
        """Parameters of the encoder over the vocabulary the run trained under."""
        return self.configuration.architecture.parameter_count(self.vocabulary_size)

    def require_open(self) -> None:
        """Refuse a backbone that already holds its weights.

        Raises:
            BackboneAlreadyDeliveredError: If the backbone already has its artifact.
        """
        if self.artifact is not None:
            raise BackboneAlreadyDeliveredError(
                f"backbone {self.name} already holds {self.artifact.key}"
            )

    def deliver(self, result: ArtifactRef, artifact: ArtifactRef, at: UtcDateTime) -> Self:
        """The backbone with its weights registered, which makes it ready.

        Args:
            result: The result the weights came with.
            artifact: Where the weights are.
            at: When they were registered.

        Raises:
            BackboneAlreadyDeliveredError: If the backbone already has its artifact.
            InvalidBackboneError: If the delivery is earlier than the order.
        """
        self.require_open()
        return replace(self, result=result, artifact=artifact, delivered_at=at)
