"""Contract of the CorpusReader port, run against every adapter.

Every adapter is asked for its data, then asked again once that data has changed. What changing
it means differs: the adapters reading files read a copy of the sample in the repository and a
byte of the copy is edited, the in-memory adapter has its data replaced, and the generated corpus
has no data to edit — its specification is its data, so a changed specification is a changed
corpus and a second reader. The harness therefore hands back the reader to ask again, rather than
assuming the first one still speaks for the corpus.

The data each adapter is given is what its format allows rather than what its corpus happens to
contain: blank lines the released files do not have, line endings of the other kind, a unit of a
single row, a unit that reports nothing, a channel that stays silent in the generated one, and in
the pickled one a channel without rows, a month without telemetry and instants at another
resolution.
"""

import math
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from emblema.catalog.adapters.in_memory.corpus_reader import InMemoryCorpusReader
from emblema.catalog.adapters.readers.cmapss import CmapssCorpusReader
from emblema.catalog.adapters.readers.esa_ad import EsaAdCorpusReader
from emblema.catalog.adapters.readers.skab import SkabCorpusReader
from emblema.catalog.adapters.readers.smd import SmdCorpusReader
from emblema.catalog.adapters.synthetic.synthetic_corpus_reader import SyntheticCorpusReader
from emblema.catalog.domain.exceptions import UnknownUnitError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.ports.corpus_reader import CorpusReader
from tests.catalog.domain.support import STATIC_SCHEMA, description, grid
from tests.catalog.domain.support import unit as make_unit
from tests.support.corpora import SAMPLE, sample
from tests.support.synthetic import HOSTILE, PROCESS


class Harness(NamedTuple):
    """A reader under test plus a way to change the data behind it, bypassing the port.

    ``change_data`` hands back the reader to ask afterwards: usually the same one, and a new one
    where the data is the specification it was built from.
    """

    reader: CorpusReader
    change_data: Callable[[], CorpusReader]


def in_memory(tmp_path: Path) -> Harness:
    channels = ["temperature", "pressure"]
    units = [
        (
            make_unit("a", 0.0, 4.0, StaticFeature("age", 61.0)),
            grid(channels, [0.0, 1.0, 2.5, 3.0]),
        ),
        (make_unit("b", 0.0, 2.0), grid(channels, [0.0, 1.0])),
    ]
    observations = sum(len(values) for _, values in units)
    reader = InMemoryCorpusReader(
        description(schema=STATIC_SCHEMA, units=2, observations=observations), units
    )

    def change_data() -> CorpusReader:
        reader.replace_data(
            description(b"changed", schema=STATIC_SCHEMA, units=2, observations=observations), units
        )
        return reader

    return Harness(reader, change_data)


def cmapss(tmp_path: Path) -> Harness:
    root = tmp_path / "cmapss"
    shutil.copytree(SAMPLE, root)
    file = root / "train_FD001.txt"
    # The copy is padded with blank lines the released files do not have: a line that carries
    # nothing must not end a unit early, and only counting what comes out shows that it does not.
    file.write_bytes(file.read_bytes().replace(b"\n", b"\n\n"))

    def change_one_value() -> CorpusReader:
        file.write_bytes(file.read_bytes().replace(b"518.67", b"518.68", 1))
        return CmapssCorpusReader(root, subsets=("FD001",))

    return Harness(CmapssCorpusReader(root, subsets=("FD001",)), change_one_value)


def skab(tmp_path: Path) -> Harness:
    root = tmp_path / "skab"
    shutil.copytree(sample("skab"), root)
    subsets = ("anomaly-free", "other", "valve1")
    rows = (root / "valve1" / "0.csv").read_bytes().split(b"\n")
    # An experiment of a single row: the format allows one and the sample holds none.
    (root / "valve1" / "1.csv").write_bytes(b"\n".join(rows[:2]) + b"\n")
    for path in root.rglob("*.csv"):
        # Blank lines the released files do not have: a line that carries nothing must not end an
        # experiment early, and only counting what comes out shows that it does not.
        path.write_bytes(path.read_bytes().replace(b"\n", b"\n\n"))

    def change_one_value() -> CorpusReader:
        path = root / "valve1" / "0.csv"
        path.write_bytes(path.read_bytes().replace(b"0.0265878", b"0.0265879", 1))
        return SkabCorpusReader(root, subsets=subsets)

    return Harness(SkabCorpusReader(root, subsets=subsets), change_one_value)


def smd(tmp_path: Path) -> Harness:
    root = tmp_path / "smd"
    shutil.copytree(sample("smd"), root)
    subsets = ("1", "2")
    minutes = (root / "machine-1-1.txt").read_bytes().split(b"\n")
    # A machine of a single minute: the format allows one and the sample holds none.
    (root / "machine-1-2.txt").write_bytes(minutes[0] + b"\n")
    first = root / "machine-1-1.txt"
    first.write_bytes(first.read_bytes().replace(b"\n", b"\n\n"))
    second = root / "machine-2-1.txt"
    # The released files end their lines with a line feed alone; a file written on Windows would
    # not, and the carriage return belongs to the ending rather than to the last metric.
    second.write_bytes(second.read_bytes().replace(b"\n", b"\r\n"))

    def change_one_value() -> CorpusReader:
        first.write_bytes(first.read_bytes().replace(b"0.032258", b"0.032259", 1))
        return SmdCorpusReader(root, subsets=subsets)

    return Harness(SmdCorpusReader(root, subsets=subsets), change_one_value)


def esa_ad(tmp_path: Path) -> Harness:
    pd = pytest.importorskip("pandas")
    root = tmp_path / "esa_ad"
    shutil.copytree(sample("esa_ad"), root)
    subsets = ("ESA-Mission1", "ESA-Mission2")

    # pandas ships no type information, so a frame is whatever it hands back.
    def rewrite(mission: str, number: int, edit: Callable[[Any], Any]) -> None:
        path = root / mission / "channels" / f"channel_{number}.zip"
        edit(pd.read_pickle(path)).to_pickle(
            path, compression={"method": "zip", "archive_name": f"channel_{number}"}
        )

    # What the format allows and the sample holds none of: a channel without a row, a channel whose
    # rows all fall in the test half, a second row in a bin already occupied, instants kept at
    # second rather than nanosecond resolution, and a month in which nothing reached the ground.
    rewrite("ESA-Mission1", 42, lambda frame: frame.iloc[:0])
    rewrite("ESA-Mission1", 43, lambda frame: frame.iloc[-30:])
    rewrite(
        "ESA-Mission1", 44, lambda frame: pd.concat([frame, frame.shift(freq="3ms")]).sort_index()
    )
    rewrite("ESA-Mission2", 18, lambda frame: frame.set_axis(frame.index.as_unit("s")))
    rewrite(
        "ESA-Mission2",
        19,
        lambda frame: frame[(frame.index < "2000-05-01") | (frame.index >= "2000-06-01")],
    )
    for number in range(20, 29):
        rewrite(
            "ESA-Mission2",
            number,
            lambda frame: frame[(frame.index < "2000-05-01") | (frame.index >= "2000-06-01")],
        )

    def change_one_value() -> CorpusReader:
        def nudge(frame: Any) -> Any:
            frame.iloc[0, 0] += 1
            return frame

        rewrite("ESA-Mission1", 41, nudge)
        return EsaAdCorpusReader(root, subsets=subsets)

    return Harness(EsaAdCorpusReader(root, subsets=subsets), change_one_value)


def synthetic(tmp_path: Path) -> Harness:
    def change_the_specification() -> CorpusReader:
        noisier = HOSTILE.with_dials(noise=HOSTILE.noise * 2)
        return SyntheticCorpusReader(PROCESS, noisier)

    return Harness(SyntheticCorpusReader(PROCESS, HOSTILE), change_the_specification)


ADAPTERS: dict[str, Callable[[Path], Harness]] = {
    "in_memory": in_memory,
    "cmapss": cmapss,
    "skab": skab,
    "smd": smd,
    "esa_ad": esa_ad,
    "synthetic": synthetic,
}


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> Harness:
    factory: Callable[[Path], Harness] = request.param
    return factory(tmp_path)


def test_describing_the_same_data_twice_gives_equal_descriptions(harness: Harness) -> None:
    assert harness.reader.describe() == harness.reader.describe()


def test_changed_values_change_the_checksum_and_nothing_else(harness: Harness) -> None:
    before = harness.reader.describe()

    after = harness.change_data().describe()

    assert after.content.checksum != before.content.checksum
    assert after.channel_schema == before.channel_schema
    assert after.sampling_regime is before.sampling_regime


def test_units_are_as_many_as_described_and_uniquely_keyed(harness: Harness) -> None:
    units = list(harness.reader.read_units())

    assert len(units) == harness.reader.describe().content.unit_count
    assert len({unit.key for unit in units}) == len(units)


def test_observations_summed_over_units_are_as_many_as_described(harness: Harness) -> None:
    total = sum(
        1
        for unit in harness.reader.read_units()
        for _ in harness.reader.read_observations(unit.key)
    )

    assert total == harness.reader.describe().content.observation_count


def test_observations_of_a_unit_are_time_ordered_inside_its_extent_on_timed_channels(
    harness: Harness,
) -> None:
    schema = harness.reader.describe().channel_schema
    timed = {channel.name for channel in schema if not channel.timeless}
    for unit in harness.reader.read_units():
        last = -math.inf
        for observation in harness.reader.read_observations(unit.key):
            assert observation.time >= last
            assert unit.extent.contains(observation.time)
            assert observation.channel in timed
            last = observation.time


def test_static_features_sit_on_timeless_channels(harness: Harness) -> None:
    schema = harness.reader.describe().channel_schema
    timeless = {channel.name for channel in schema if channel.timeless}
    for unit in harness.reader.read_units():
        assert {feature.channel for feature in unit.static_features} <= timeless


def test_a_key_the_reader_cannot_resolve_is_reported_before_the_stream(harness: Harness) -> None:
    with pytest.raises(UnknownUnitError):
        harness.reader.read_observations(UnitKey("no-such-unit"))
