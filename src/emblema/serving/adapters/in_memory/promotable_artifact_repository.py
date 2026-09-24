from emblema.serving.domain.artifact_origin import ArtifactOrigin
from emblema.serving.domain.promotable_artifact import PromotableArtifact
from emblema.shared.kernel.checksums import Checksum


class InMemoryPromotableArtifactRepository:
    """Keeps the projection in a dictionary keyed by origin, for a process without a database."""

    def __init__(self) -> None:
        self._artifacts: dict[ArtifactOrigin, PromotableArtifact] = {}

    def save(self, artifact: PromotableArtifact) -> None:
        self._artifacts[artifact.origin] = artifact

    def find_by_checksum(self, checksum: Checksum) -> tuple[PromotableArtifact, ...]:
        return tuple(
            sorted(
                (
                    artifact
                    for artifact in self._artifacts.values()
                    if artifact.artifact.checksum == checksum
                ),
                key=lambda artifact: (
                    artifact.completed_at,
                    str(artifact.origin.campaign),
                    str(artifact.origin.candidate),
                ),
            )
        )
