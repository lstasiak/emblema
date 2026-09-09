"""Key layout shared by every artifact store adapter.

Keys are ``<retention>/<algorithm>/<digest>``. The retention class comes first so that one
lifecycle rule per environment prefix expires every transient object; the digest comes last so
that identical content maps to one object whichever process stored it. Every adapter uses this
layout, so a reference produced by one resolves in another once the bytes are copied.
"""

from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.artifact_store import Retention


def content_address(content: bytes, retention: Retention) -> ArtifactRef:
    checksum = Checksum.of_bytes(content)
    return ArtifactRef(f"{retention}/{checksum.algorithm}/{checksum.digest}", checksum)
