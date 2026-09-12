"""The publishing process assembled and run end to end, on adapters that reach nothing outside.

What is checked here is the wiring: that the process this composition root builds does publish a
corpus, and that every use case in it received an adapter rather than a default nobody meant. The
behaviour of each use case is settled in its own test; without this one, a process could be
correct in every part and still be wired to the wrong store.
"""

import argparse
from pathlib import Path
from typing import NamedTuple

import pytest

from emblema.catalog.adapters.archive.block_corpus_archive import BlockCorpusArchive
from emblema.catalog.adapters.in_memory.corpus_reader import InMemoryCorpusReader
from emblema.catalog.domain.corpus_unit import CorpusUnit, TimeExtent
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.observation import Observation
from emblema.config.artifact_store_settings import ArtifactStoreSettings
from emblema.config.settings import Settings
from emblema.entrypoints.cli.composition import Services, build_services
from emblema.entrypoints.cli.publish_corpus import parse, publish
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.storage.s3 import S3ArtifactStore
from tests.catalog.domain.support import SCHEMA, description

DATA = b"records"
UNITS = [
    (
        CorpusUnit(UnitKey(f"u{number}"), TimeExtent(0.0, 8.0)),
        [
            Observation(channel, float(time), float(time) + offset)
            for time in range(8)
            for offset, channel in enumerate(("pressure", "temperature"))
        ],
    )
    for number in range(1, 5)
]


@pytest.fixture
def settings() -> Settings:
    """Settings a composition may read without any of them reaching a network."""
    return Settings(
        artifact_store=ArtifactStoreSettings(
            endpoint_url="http://127.0.0.1:3900",
            region="garage",
            bucket="emblema",
            key_prefix="test",
        )
    )


class Process(NamedTuple):
    """The assembled process and the store its artifacts were meant to land in."""

    services: Services
    store: InMemoryArtifactStore


@pytest.fixture
def process(settings: Settings, tmp_path: Path) -> Process:
    told = description(DATA, SCHEMA, units=len(UNITS), observations=sum(len(o) for _, o in UNITS))
    store = InMemoryArtifactStore()
    services = build_services(
        settings,
        corpus_root=tmp_path / "raw",
        workspace=tmp_path / "workspace",
        reader=InMemoryCorpusReader(told, UNITS),
        archive=BlockCorpusArchive(store, tmp_path / "workspace"),
    )
    return Process(services, store)


def arguments(**overrides: object) -> argparse.Namespace:
    parsed = parse(
        ["--corpus", "cmapss", "--root", "data/raw/cmapss", "--window", "4", "--stride", "2"]
    )
    for name, value in overrides.items():
        setattr(parsed, name, value)
    return parsed


def test_the_process_publishes_a_corpus_that_reads_back(process: Process) -> None:
    ref = publish(process.services, arguments(validation_fraction=0.25))

    manifest = process.services.archive.read_manifest(ref)
    windows = process.services.archive.read_windows(manifest, manifest.split.training)
    assert manifest.corpus == "cmapss"
    assert len(windows) > 0


def test_the_artifacts_land_in_the_store_the_process_was_given(process: Process) -> None:
    # Every use case holding its own store would leave each of them correct and the process
    # broken; the corpus would be published where nobody looks for it.
    ref = publish(process.services, arguments(validation_fraction=0.25))

    manifest = process.services.archive.read_manifest(ref)
    assert process.store.exists(ref)
    assert process.store.exists(manifest.block)


def test_what_the_command_line_says_is_what_the_corpus_was_cut_with(process: Process) -> None:
    ref = publish(process.services, arguments(validation_fraction=0.25, seed=7, window=2.0))

    manifest = process.services.archive.read_manifest(ref)
    assert (manifest.window.length, manifest.window.stride) == (2.0, 2.0)
    assert manifest.split_seed == 7
    assert len(manifest.split.validation) == 1


def test_without_overrides_the_process_stores_artifacts_where_the_settings_point(
    settings: Settings, tmp_path: Path
) -> None:
    # The overrides above are what every other test here uses, so nothing would otherwise exercise
    # the wiring a real run gets: the store named by the environment, behind a block archive.
    services = build_services(
        settings, corpus_root=tmp_path / "raw", workspace=tmp_path / "workspace"
    )

    assert isinstance(services.archive, BlockCorpusArchive)
    assert isinstance(services.archive._store, S3ArtifactStore)


def test_the_command_line_states_the_window_it_was_given() -> None:
    parsed = arguments()

    assert parsed.window == 4.0
    assert parsed.stride == 2.0
    assert parsed.seed == 1
