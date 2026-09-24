from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.serving.domain.artifact_origin import ArtifactOrigin
from emblema.serving.domain.exceptions import (
    AmbiguousArtifactError,
    ArtifactNotPromotableError,
    ArtifactUnavailableError,
)
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.domain.promotable_artifact import PromotableArtifact
from emblema.serving.domain.served_model import ServedModel
from emblema.serving.ports.promotable_artifact_repository import PromotableArtifactRepository
from emblema.serving.ports.served_model_repository import ServedModelRepository
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.id_generator import IdGenerator


@dataclass(frozen=True, kw_only=True)
class PromoteArtifactCommand:
    """Request to serve an artifact a finished campaign kept.

    Attributes:
        checksum: What the artifact's content hashes to, which is how it is named.
        campaign: The campaign it is promoted out of, needed only when several kept it.
        candidate: The competitor it is promoted as, needed only when one campaign kept the
            same artifact as several of them.
    """

    checksum: Checksum
    campaign: CampaignId | None = None
    candidate: CandidateRef | None = None

    def names(self, origin: ArtifactOrigin) -> bool:
        """Whether the origin is among those the command leaves possible."""
        return (self.campaign is None or self.campaign == origin.campaign) and (
            self.candidate is None or self.candidate == origin.candidate
        )


class PromoteArtifact:
    """Puts an artifact into service, provided a finished campaign measured it.

    The artifact is named by its checksum and looked up in the projection of what campaigns
    kept, never trusted from the request, so the reference served is the one that was measured.
    Its bytes are confirmed present before anything is recorded: the store is content-addressed
    and checks every read against the checksum, so a model whose artifact is there will load the
    bytes it was measured with, and one whose artifact is gone would never load at all.
    """

    def __init__(
        self,
        promotables: PromotableArtifactRepository,
        served: ServedModelRepository,
        store: ArtifactStore,
        ids: IdGenerator,
        clock: Clock,
    ) -> None:
        self._promotables = promotables
        self._served = served
        self._store = store
        self._ids = ids
        self._clock = clock

    def __call__(self, command: PromoteArtifactCommand) -> ServedModelId:
        """Serve the artifact and return the identity of the model now serving it.

        Raises:
            ArtifactNotPromotableError: If no finished campaign kept an artifact with that
                checksum, or none did under the campaign or competitor named.
            AmbiguousArtifactError: If it was kept more than once and the request does not say
                which origin to promote it out of.
            ArtifactUnavailableError: If the artifact is no longer in the store.
            ArtifactAlreadyServedError: If a model not withdrawn already serves it.
            InvalidServedModelError: If the clock reads earlier than the campaign finished.
        """
        kept = self._promotable(command)
        if not self._store.exists(kept.artifact):
            raise ArtifactUnavailableError(
                f"artifact {command.checksum} is not in the store: nothing to serve"
            )
        model = ServedModel.promoted(
            kept, served_model_id=self._ids.generate(ServedModelId), at=self._clock.now()
        )
        self._served.save(model)
        return model.served_model_id

    def _promotable(self, command: PromoteArtifactCommand) -> PromotableArtifact:
        """The one artifact the command names.

        Raises:
            ArtifactNotPromotableError: If it names none.
            AmbiguousArtifactError: If it names several.
        """
        found = tuple(
            kept
            for kept in self._promotables.find_by_checksum(command.checksum)
            if command.names(kept.origin)
        )
        if not found:
            raise ArtifactNotPromotableError(
                f"no finished campaign kept an artifact with checksum {command.checksum}"
                + self._qualifier(command)
            )
        if len(found) > 1:
            origins = ", ".join(
                f"{kept.origin.candidate} in campaign {kept.origin.campaign}" for kept in found
            )
            raise AmbiguousArtifactError(
                f"artifact {command.checksum} was kept more than once ({origins}); name the "
                "campaign to promote it out of, and the competitor where one campaign kept it "
                "twice"
            )
        return found[0]

    @staticmethod
    def _qualifier(command: PromoteArtifactCommand) -> str:
        """What the command narrowed the search by, as the refusal repeats it."""
        parts = []
        if command.candidate is not None:
            parts.append(f" as {command.candidate}")
        if command.campaign is not None:
            parts.append(f" in campaign {command.campaign}")
        return "".join(parts)
