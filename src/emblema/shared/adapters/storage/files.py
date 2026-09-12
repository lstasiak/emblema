"""Moving an artifact between a store and the local filesystem, a chunk at a time.

Shared by every store adapter, so each is left with the part that is about its own technology.
"""

import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO

from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.exceptions import ArtifactIntegrityError

# Large enough that moving a multi-gigabyte artifact is not a syscall benchmark, small enough that
# a process moving several at once is not the reason it runs out of memory.
CHUNK_SIZE = 1 << 20


def chunks_of(source: Path, size: int = CHUNK_SIZE) -> Iterator[bytes]:
    with source.open("rb") as handle:
        while chunk := handle.read(size):
            yield chunk


def write_verified(ref: ArtifactRef, chunks: Iterable[bytes], destination: Path) -> None:
    """Write ``chunks`` at ``destination``, atomically and only once they hash to the reference.

    Nothing observes a partial or unverified artifact at the destination, so a reader may map the
    file without hashing it again.

    Raises:
        ArtifactIntegrityError: If the chunks do not hash to the reference's checksum.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            written = Checksum.of_chunks(_passing(chunks, handle), ref.checksum.algorithm)
        if written != ref.checksum:
            raise ArtifactIntegrityError(f"content under {ref.key!r} does not match {ref.checksum}")
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _passing(chunks: Iterable[bytes], handle: IO[bytes]) -> Iterator[bytes]:
    """The chunks, each written as it passes, so one pass over them both stores and hashes."""
    for chunk in chunks:
        handle.write(chunk)
        yield chunk
