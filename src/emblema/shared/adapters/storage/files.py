"""Moving an artifact between a store and the local filesystem, a chunk at a time.

Shared by every store adapter, so each is left with the part that is about its own technology.
"""

import fcntl
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from itertools import chain
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


def write_resuming(
    ref: ArtifactRef, chunks_after: Callable[[int], Iterable[bytes]], destination: Path
) -> None:
    """Write the artifact at ``destination``, continuing whatever an interrupted attempt left.

    The bytes received so far stay in ``<destination>.part``, so a transfer cut at seven of ten
    gigabytes asks only for the rest. ``chunks_after`` gives the content from an offset. As with
    ``write_verified``, the destination receives the whole artifact or nothing. A process that
    finds another writing the same partial file downloads on its own instead of waiting.

    Raises:
        ArtifactIntegrityError: If the whole does not hash to the reference; the partial file goes.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    with partial.open("ab") as handle:
        if not _holds(handle, partial):
            write_verified(ref, chunks_after(0), destination)
            return
        received = os.fstat(handle.fileno()).st_size
        whole = chain(chunks_of(partial), _passing(chunks_after(received), handle))
        try:
            written = Checksum.of_chunks(whole, ref.checksum.algorithm)
        except BaseException:
            handle.flush()
            if os.fstat(handle.fileno()).st_size == 0:
                partial.unlink()
            raise
        handle.flush()
        if written != ref.checksum:
            partial.unlink()
            raise ArtifactIntegrityError(f"content under {ref.key!r} does not match {ref.checksum}")
        os.replace(partial, destination)


def _holds(handle: IO[bytes], partial: Path) -> bool:
    """Whether this process alone writes the partial file, which may have moved since it opened."""
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return os.fstat(handle.fileno()).st_ino == partial.stat().st_ino
    except (BlockingIOError, FileNotFoundError):
        return False


def _passing(chunks: Iterable[bytes], handle: IO[bytes]) -> Iterator[bytes]:
    """The chunks, each written as it passes, so one pass over them both stores and hashes."""
    for chunk in chunks:
        handle.write(chunk)
        yield chunk
