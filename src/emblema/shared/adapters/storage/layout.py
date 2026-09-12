"""Key layout shared by every artifact store adapter.

Keys are ``<retention>/<algorithm>/<digest>``. The retention class comes first so that one
lifecycle rule per environment prefix expires every transient object; the digest comes last so
that identical content maps to one object whichever process stored it. Every adapter uses this
layout, so a reference produced by one resolves in another once the bytes are copied.
"""

from pathlib import Path

from emblema.shared.adapters.storage.files import chunks_of
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.artifact_store import Retention


def address_of(checksum: Checksum, retention: Retention) -> ArtifactRef:
    return ArtifactRef(f"{retention}/{checksum.algorithm}/{checksum.digest}", checksum)


def content_address(content: bytes, retention: Retention) -> ArtifactRef:
    return address_of(Checksum.of_bytes(content), retention)


def file_address(source: Path, retention: Retention) -> ArtifactRef:
    """Address of a file's contents, read a chunk at a time so its size never bounds the caller."""
    return address_of(Checksum.of_chunks(chunks_of(source)), retention)
