from dataclasses import dataclass

from emblema.pretraining.domain.backbone.backbone import Backbone
from emblema.pretraining.domain.exceptions import (
    IncompatibleStartError,
    PretrainingOrderRejectedError,
)
from emblema.pretraining.domain.handoff.pretraining_order import PretrainingOrder
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.run_signature import RunSignature
from emblema.pretraining.domain.training.training_mixture import TrainingMixture
from emblema.pretraining.ports.backbone_repository import BackboneRepository
from emblema.pretraining.ports.handoff_exchange import HandoffExchange
from emblema.pretraining.ports.training_corpus_reader import TrainingCorpusReader
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.id_generator import IdGenerator


@dataclass(frozen=True, kw_only=True)
class OrderPretrainingCommand:
    """One run to be made, on this machine or another: what, on which corpora, by which code.

    Attributes:
        configuration: The experiment to run.
        corpora: The corpora to train on, in the order their vocabulary was chained: each the
            name the experiment gives it and the manifest of the publication to read it from,
            which must publish the corpus of that name.
        run: What this run is called within the experiment.
        git_commit: Revision of the code the run is to be made with.
    """

    configuration: ExperimentConfiguration
    corpora: tuple[tuple[str, ArtifactRef], ...]
    run: str
    git_commit: str


@dataclass(frozen=True, kw_only=True)
class PlacedPretrainingOrder:
    """What ordering leaves behind: the backbone registered and the order the trainer is handed.

    Attributes:
        backbone: The backbone, registered as ordered.
        order: Reference to the order in the exchange.
    """

    backbone: BackboneId
    order: ArtifactRef


class OrderPretraining:
    """Registers a backbone as ordered and places the order the machine that trains is handed.

    The corpora are read here rather than only described: the signature the result will be
    held to covers how much of each corpus a run reads, and that is known from the windows, not
    from the manifests. Reading them together is also what checks that they can be trained on
    together — that each continues the vocabulary of the one before. A run that starts from
    stored weights is traced to the ready backbone holding them, which becomes the parent, and
    is refused where those weights would not fit: another shape or objective, or a vocabulary
    the corpora ordered do not chain with, since a row of the channel table is carried over by
    its identifier alone. The backbone is saved before the order is placed, because an order in
    the exchange for a backbone the registry does not know is a delivery nobody could accept.
    """

    def __init__(
        self,
        reader: TrainingCorpusReader,
        backbones: BackboneRepository,
        exchange: HandoffExchange,
        ids: IdGenerator,
        clock: Clock,
    ) -> None:
        self._reader = reader
        self._backbones = backbones
        self._exchange = exchange
        self._ids = ids
        self._clock = clock

    def __call__(self, command: OrderPretrainingCommand) -> PlacedPretrainingOrder:
        """Register the backbone and place its order.

        Raises:
            PretrainingOrderRejectedError: If a manifest publishes a corpus other than the one
                it is named for, or the configuration states passes or fractions for a corpus
                the order does not read.
            InvalidTrainingMixtureError: If the corpora read are not a chain of one vocabulary.
            BackboneNotFoundError: If the configuration starts from weights no ready backbone
                of the registry holds.
            IncompatibleStartError: If the backbone holding those weights is of another shape,
                objective or decoder, or its vocabulary and the corpora's do not lie on one
                chain of publications.
            UnreadablePublishedCorpusError: If a manifest or its block cannot be read.
        """
        named = {name for name, _ in command.corpora}
        for knob, stated_for in (
            ("passes", command.configuration.passes),
            ("fractions", command.configuration.fractions),
        ):
            unread = [stated.corpus for stated in stated_for if stated.corpus not in named]
            if unread:
                raise PretrainingOrderRejectedError(
                    f"the configuration states {knob} for corpora the order does not read: {unread}"
                )
        described = tuple(self._reader.describe(manifest) for _, manifest in command.corpora)
        for (expected, _), found in zip(command.corpora, described, strict=True):
            if found.corpus != expected:
                raise PretrainingOrderRejectedError(
                    f"the experiment names corpus {expected!r}, "
                    f"the manifest publishes {found.corpus!r}"
                )
        mixture = TrainingMixture(
            corpora=tuple(
                self._reader.read(manifest, command.configuration.corpus_share_of(name))
                for name, manifest in command.corpora
            )
        )
        backbone = Backbone(
            id=self._ids.generate(BackboneId),
            configuration=command.configuration,
            inputs=described,
            run=command.run,
            git_commit=command.git_commit,
            signature=RunSignature.of(command.configuration, mixture),
            ordered_at=self._clock.now(),
            parent=self._parent_of(command.configuration, mixture),
        )
        self._backbones.save(backbone)
        return PlacedPretrainingOrder(
            backbone=backbone.id, order=self._exchange.place(PretrainingOrder.of(backbone))
        )

    def _parent_of(
        self, configuration: ExperimentConfiguration, mixture: TrainingMixture
    ) -> BackboneId | None:
        """The backbone whose weights the run starts from, held to the run; none from the seed."""
        if configuration.start is None:
            return None
        parent = self._backbones.get_by_artifact(configuration.start)
        parent.require_continuable_by(configuration)
        # The parent's vocabulary is the last corpus's of its chain; the run's corpora must lie
        # on that chain, before or after it, or the rows carried over would name other channels.
        learnt_under = self._reader.channels(parent.inputs[-1].manifest)
        if not mixture.chains_with(learnt_under):
            raise IncompatibleStartError(
                f"the corpora ordered ({mixture.name}) are not published on the chain backbone "
                f"{parent.name} was pretrained on: its {len(learnt_under)} channels and their "
                f"{mixture.vocabulary_size} are not one a prefix of the other"
            )
        return parent.id
