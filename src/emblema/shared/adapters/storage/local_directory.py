import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from emblema.shared.adapters.storage.files import chunks_of, write_verified
from emblema.shared.adapters.storage.layout import content_address, file_address
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import Retention
from emblema.shared.ports.exceptions import ArtifactIntegrityError, ArtifactNotFoundError


class LocalDirectoryArtifactStore:
    """Artifact store on the local filesystem, one file per artifact under ``root``.

    The fake of the port for tests and for development without Docker. A write goes to a
    temporary file in the target directory and is moved into place in one step, so a concurrent
    reader never sees a partial artifact.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def put(self, content: bytes, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        return self._store(content_address(content, retention), (content,))

    def put_file(self, source: Path, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        return self._store(file_address(source, retention), chunks_of(source))

    def get(self, ref: ArtifactRef) -> bytes:
        path = self._path(ref)
        if not self._is_inside_root(path):
            raise ArtifactNotFoundError(ref.key)
        try:
            content = path.read_bytes()
        except FileNotFoundError as error:
            raise ArtifactNotFoundError(ref.key) from error
        if not ref.checksum.matches(content):
            raise ArtifactIntegrityError(f"content under {ref.key!r} does not match {ref.checksum}")
        return content

    def get_file(self, ref: ArtifactRef, destination: Path) -> None:
        if not self.exists(ref):
            raise ArtifactNotFoundError(ref.key)
        write_verified(ref, chunks_of(self._path(ref)), destination)

    def exists(self, ref: ArtifactRef) -> bool:
        path = self._path(ref)
        return self._is_inside_root(path) and path.is_file()

    def _store(self, ref: ArtifactRef, chunks: Iterable[bytes]) -> ArtifactRef:
        path = self._path(ref)
        if not path.exists():
            self._write_atomically(path, chunks)
        return ref

    def _path(self, ref: ArtifactRef) -> Path:
        return self._root.joinpath(*ref.key.split("/"))

    def _is_inside_root(self, path: Path) -> bool:
        # A reference read back from the database is data, not a promise: a key that climbs out of
        # the root names no artifact of this store, whatever sits at the end of it.
        return path.resolve().is_relative_to(self._root.resolve())

    @staticmethod
    def _write_atomically(path: Path, chunks: Iterable[bytes]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                for chunk in chunks:
                    handle.write(chunk)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
