"""What the block archive does beyond its port: keep a block on this machine once it has one."""

from pathlib import Path

import pytest

from emblema.catalog.adapters.archive.block_corpus_archive import BlockCorpusArchive
from emblema.catalog.application.assemblers.published_corpus_manifest_assembler import (
    PublishedCorpusManifestAssembler,
)
from emblema.catalog.domain.corpus_unit import TimeExtent
from emblema.catalog.domain.exceptions import WindowNotArchivableError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.placed_window import PlacedWindow
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from tests.shared.adapters.windows.support import COLLAPSING, TIMED

FIRST, SECOND = UnitKey("u1"), UnitKey("u2")
PLACED = (
    PlacedWindow(FIRST, TimeExtent(0.0, 10.0), TIMED),
    PlacedWindow(SECOND, TimeExtent(0.0, 10.0), TIMED),
)


def archive_over(store: InMemoryArtifactStore, workspace: Path) -> BlockCorpusArchive:
    return BlockCorpusArchive(store, workspace, PublishedCorpusManifestAssembler())


def test_a_block_just_archived_waits_in_the_workspace_under_its_digest(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"

    written = archive_over(InMemoryArtifactStore(), workspace).write_windows(PLACED)

    assert [path.name for path in workspace.iterdir()] == [written.block.checksum.digest]


def test_reading_a_block_this_machine_archived_asks_the_store_for_nothing(
    tmp_path: Path,
) -> None:
    # A publisher that trains on what it just published would otherwise download the block it
    # has on disk; an empty store proves the read came from the workspace.
    workspace = tmp_path / "workspace"
    written = archive_over(InMemoryArtifactStore(), workspace).write_windows(PLACED)
    elsewhere = archive_over(InMemoryArtifactStore(), workspace)

    windows = elsewhere.read_windows(written, [FIRST, SECOND])

    assert list(windows) == [TIMED, TIMED]


def test_a_window_the_archive_cannot_store_leaves_the_workspace_clean(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    archive = archive_over(InMemoryArtifactStore(), workspace)

    with pytest.raises(WindowNotArchivableError, match="unit u1"):
        archive.write_windows([PlacedWindow(FIRST, TimeExtent(0.0, 10.0), COLLAPSING)])

    assert list(workspace.iterdir()) == []
