"""Contract of the CorpusReader port, run against every adapter.

The C-MAPSS adapter reads a copy of the sample in the repository, so that the data behind it can
be changed the way a corpus changes on disk; the in-memory adapter has its description replaced.
"""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest

from emblema.catalog.adapters.in_memory.corpus_reader import InMemoryCorpusReader
from emblema.catalog.adapters.readers.cmapss import CmapssCorpusReader
from emblema.catalog.ports.corpus_reader import CorpusReader
from tests.catalog.domain.support import description

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "cmapss"


class Harness(NamedTuple):
    """A reader under test plus a way to change the data behind it, bypassing the port."""

    reader: CorpusReader
    change_data: Callable[[], None]


def in_memory(tmp_path: Path) -> Harness:
    reader = InMemoryCorpusReader(description())
    return Harness(reader, lambda: reader.replace_data(description(b"changed")))


def cmapss(tmp_path: Path) -> Harness:
    root = tmp_path / "cmapss"
    shutil.copytree(SAMPLE, root)
    file = root / "train_FD001.txt"

    def change_one_value() -> None:
        file.write_bytes(file.read_bytes().replace(b"518.67", b"518.68", 1))

    return Harness(CmapssCorpusReader(root, subsets=("FD001",)), change_one_value)


ADAPTERS: dict[str, Callable[[Path], Harness]] = {"in_memory": in_memory, "cmapss": cmapss}


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> Harness:
    factory: Callable[[Path], Harness] = request.param
    return factory(tmp_path)


def test_describing_the_same_data_twice_gives_equal_descriptions(harness: Harness) -> None:
    assert harness.reader.describe() == harness.reader.describe()


def test_changed_values_change_the_checksum_and_nothing_else(harness: Harness) -> None:
    before = harness.reader.describe()

    harness.change_data()
    after = harness.reader.describe()

    assert after.content.checksum != before.content.checksum
    assert after.channel_schema == before.channel_schema
    assert after.sampling_regime is before.sampling_regime
