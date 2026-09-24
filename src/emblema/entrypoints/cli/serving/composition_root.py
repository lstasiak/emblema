from emblema.config.settings import Settings
from emblema.entrypoints.cli.serving.adapters import Adapters
from emblema.entrypoints.cli.serving.services import Services
from emblema.entrypoints.configured import configured_engine, configured_store, settings_for
from emblema.serving.adapters.persistence.promotable_artifact_repository import (
    SqlAlchemyPromotableArtifactRepository,
)
from emblema.serving.adapters.persistence.served_model_repository import (
    SqlAlchemyServedModelRepository,
)
from emblema.serving.application.use_cases.promote_artifact import PromoteArtifact
from emblema.serving.application.use_cases.withdraw_served_model import WithdrawServedModel
from emblema.serving.ports.promotable_artifact_repository import PromotableArtifactRepository
from emblema.serving.ports.served_model_repository import ServedModelRepository
from emblema.shared.adapters.system.clock import SystemClock
from emblema.shared.adapters.system.id_generator import Uuid4IdGenerator
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.id_generator import IdGenerator


class CompositionRoot:
    """Assembles the process that puts artifacts into service and takes them out.

    It reads the projection a campaign process fed and never feeds it: what is promotable is
    decided where campaigns close, so this process registers no handler and holds nothing of
    Evaluation. All lifetimes are process-scoped; one invocation runs one use case.

    Attributes:
        adapters: The port implementations the process runs on.
        services: The use cases, each already holding its dependencies.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: ArtifactStore | None = None,
        promotables: PromotableArtifactRepository | None = None,
        served: ServedModelRepository | None = None,
        clock: Clock | None = None,
        ids: IdGenerator | None = None,
    ) -> None:
        """Assemble the process.

        Args:
            settings: Values read from the environment; only the root and the adapters see them.
            store: Artifact store; the configured S3-compatible bucket unless given.
            promotables: What finished campaigns kept; the configured database unless given.
            served: Registry of served models; the same database unless given.
            clock: Source of the current instant; the system clock unless given.
            ids: Source of new identifiers; random UUIDs unless given.

        Raises:
            ValueError: If an adapter is left to the root without settings to build it from.
        """
        chosen_promotables, chosen_served = self._registries(settings, promotables, served)
        self.adapters = Adapters(
            store=configured_store(settings_for(settings, "store")) if store is None else store,
            promotables=chosen_promotables,
            served=chosen_served,
            clock=SystemClock() if clock is None else clock,
            ids=Uuid4IdGenerator() if ids is None else ids,
        )
        self.services = Services(
            promote_artifact=PromoteArtifact(
                self.adapters.promotables,
                self.adapters.served,
                self.adapters.store,
                self.adapters.ids,
                self.adapters.clock,
            ),
            withdraw_served_model=WithdrawServedModel(self.adapters.served, self.adapters.clock),
        )

    @staticmethod
    def _registries(
        settings: Settings | None,
        promotables: PromotableArtifactRepository | None,
        served: ServedModelRepository | None,
    ) -> tuple[PromotableArtifactRepository, ServedModelRepository]:
        """Both repositories, on one engine wherever either was left to be built.

        Raises:
            ValueError: If one was left to be built without settings to build it from.
        """
        if promotables is not None and served is not None:
            return promotables, served
        engine = configured_engine(settings_for(settings, "the registries"))
        return (
            SqlAlchemyPromotableArtifactRepository(engine) if promotables is None else promotables,
            SqlAlchemyServedModelRepository(engine) if served is None else served,
        )
