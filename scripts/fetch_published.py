"""Fetch what a run reads from the artifact store before any process needs it.

Run from a notebook cell on a paid card or from the Mac: a block fetched once here is one a dozen
workers do not each fetch, and a transfer cut short continues from what it received.

    python scripts/fetch_published.py blocks WORKSPACE DIGEST [DIGEST ...]
    python scripts/fetch_published.py labels DIGEST
"""

import argparse
import io
import tarfile
from collections.abc import Sequence
from pathlib import Path

from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.config.settings import Settings
from emblema.entrypoints.configured import configured_store
from emblema.shared.adapters.windows.block_workspace import BlockWorkspace
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.artifact_store import ArtifactStore

RAW = Path("data/raw")


def fetch_blocks(store: ArtifactStore, workspace: Path, digests: Sequence[str]) -> list[str]:
    """Each publication's block in the workspace; a line naming each."""
    blocks = BlockWorkspace(store, workspace)
    said = []
    for digest in digests:
        ref = ArtifactRef(f"durable/sha256/{digest}", Checksum.parse(f"sha256:{digest}"))
        manifest = PublishedCorpusManifestJson().decode(store.get(ref))
        path = blocks.fetched(manifest.block)
        said.append(f"{manifest.corpus}: {path.stat().st_size / 2**30:.1f} GB")
    return said


def fetch_labels(store: ArtifactStore, digest: str, destination: Path = RAW) -> int:
    """The archive of task labels unpacked at ``destination``; how many bytes it held."""
    ref = ArtifactRef(f"transient/sha256/{digest}", Checksum.parse(f"sha256:{digest}"))
    data = store.get(ref)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(destination, filter="data")
    return len(data)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - environment
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    blocks = commands.add_parser("blocks")
    blocks.add_argument("workspace", type=Path)
    blocks.add_argument("digests", nargs="+")
    labels = commands.add_parser("labels")
    labels.add_argument("digest")
    arguments = parser.parse_args(argv)
    store = configured_store(Settings())
    if arguments.command == "blocks":
        for line in fetch_blocks(store, arguments.workspace, arguments.digests):
            print(line, flush=True)
    else:
        print(f"labels: {fetch_labels(store, arguments.digest)} bytes", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
