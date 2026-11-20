"""Reading a file in pieces, shared by the adapters whose checksum covers whole archives."""

from collections.abc import Iterator
from pathlib import Path

# Small enough that an archive of gigabytes is never held whole, large enough to hash fast.
_CHUNK_BYTES = 1 << 20


def chunks_of(path: Path) -> Iterator[bytes]:
    """The file's bytes in order, a megabyte at a time."""
    with path.open("rb") as file:
        while chunk := file.read(_CHUNK_BYTES):
            yield chunk
