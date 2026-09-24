from dataclasses import dataclass

from emblema.serving.domain.promotable_artifact import PromotableArtifact
from emblema.serving.ports.promotable_artifact_repository import PromotableArtifactRepository


@dataclass(frozen=True, kw_only=True)
class RecordPromotableArtifactsCommand:
    """What one finished campaign made promotable.

    Attributes:
        artifacts: Every artifact the campaign kept, as this context knows it.
    """

    artifacts: tuple[PromotableArtifact, ...]


class RecordPromotableArtifacts:
    """Adds what a finished campaign kept to the projection promotion is checked against.

    Recording the same campaign again replaces what the first delivery left, which is what lets
    an announcement be repeated after one that failed.
    """

    def __init__(self, promotables: PromotableArtifactRepository) -> None:
        self._promotables = promotables

    def __call__(self, command: RecordPromotableArtifactsCommand) -> None:
        for artifact in command.artifacts:
            self._promotables.save(artifact)
