from pathlib import Path

from emblema.shared.adapters.storage.files import chunks_of, write_verified
from emblema.shared.adapters.storage.layout import content_address
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.retention import Retention
from emblema.shared.ports.exceptions import ArtifactIntegrityError, ArtifactNotFoundError


class InMemoryArtifactStore:
    """Artifact store in a dictionary, so that an application test touches no disk."""

    def __init__(self) -> None:
        self._content: dict[str, bytes] = {}

    def put(self, content: bytes, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        ref = content_address(content, retention)
        self._content.setdefault(ref.key, content)
        return ref

    def put_file(self, source: Path, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        return self.put(b"".join(chunks_of(source)), retention)

    def get(self, ref: ArtifactRef) -> bytes:
        content = self._content.get(ref.key)
        if content is None:
            raise ArtifactNotFoundError(ref.key)
        if not ref.checksum.matches(content):
            raise ArtifactIntegrityError(f"content under {ref.key!r} does not match {ref.checksum}")
        return content

    def get_file(self, ref: ArtifactRef, destination: Path) -> None:
        if ref.key not in self._content:
            raise ArtifactNotFoundError(ref.key)
        write_verified(ref, (self._content[ref.key],), destination)

    def exists(self, ref: ArtifactRef) -> bool:
        return ref.key in self._content
