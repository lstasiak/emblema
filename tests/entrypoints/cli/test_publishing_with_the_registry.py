"""Two processes, one registry: the second publication of a corpus finds the first one's.

Marked ``integration``: it needs the metadata database of the local stack, migrated.
"""

from pathlib import Path

import pytest
from sqlalchemy import Engine

from emblema.catalog.adapters.in_memory.corpus_reader import InMemoryCorpusReader
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.config.settings import Settings
from emblema.entrypoints.cli.composition_root import CompositionRoot
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from tests.catalog.domain.support import SCHEMA, description, measured_units
from tests.support.corpora import CORPUS, publish_command
from tests.support.database import clear_catalog, migrated_engine

pytestmark = pytest.mark.integration

UNITS = measured_units(4)


@pytest.fixture
def database() -> Engine:
    engine = migrated_engine()
    clear_catalog(engine)
    return engine


def process(store: InMemoryArtifactStore, workspace: Path) -> CompositionRoot:
    """A process over the configured database, with the corpus and the artifacts in memory."""
    told = description(b"records", SCHEMA, units=len(UNITS), observations=64)
    return CompositionRoot(
        Settings(),
        corpus_root=workspace / "raw",
        workspace=workspace,
        reader=InMemoryCorpusReader(told, UNITS),
        store=store,
    )


def test_a_corpus_published_by_one_process_is_the_corpus_the_next_one_publishes(
    database: Engine, tmp_path: Path
) -> None:
    store = InMemoryArtifactStore()
    command = publish_command(window=WindowSpec(4.0, 2.0), validation_fraction=0.25)

    first = process(store, tmp_path / "first").services.publish_corpus(command)
    second = process(store, tmp_path / "second").services.publish_corpus(command)

    assert first == second
    corpus = process(store, tmp_path / "third").adapters.corpora.find_by_name(CORPUS)
    assert corpus is not None
    assert len(corpus.frozen_versions) == 1
