from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from emblema.catalog.adapters.archive.block_corpus_archive import BlockCorpusArchive
from emblema.catalog.adapters.in_memory.corpus_reader import InMemoryCorpusReader
from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.application.tokenise_corpus_version import (
    TokeniseCorpusVersion,
    TokeniseCorpusVersionCommand,
)
from emblema.catalog.domain.corpus import Corpus
from emblema.catalog.domain.corpus_unit import CorpusUnit, TimeExtent
from emblema.catalog.domain.exceptions import (
    CorpusDataChangedError,
    CorpusVersionNotFrozenError,
    InvalidUnitSplitError,
)
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.observation import Observation
from emblema.catalog.domain.window_spec import WindowSpec
from emblema.catalog.ports.corpus_archive import CorpusArchive
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from tests.catalog.domain.support import (
    AT,
    LICENCE,
    SCHEMA,
    SOURCE,
    corpus_id,
    description,
    version_id,
)

DATA = b"records"
WINDOW = WindowSpec(length=4.0, stride=2.0)


def readings(key: str) -> tuple[CorpusUnit, list[Observation]]:
    """A unit measured on both channels at every whole instant of its extent."""
    unit = CorpusUnit(UnitKey(key), TimeExtent(0.0, 8.0))
    observations = [
        Observation(channel, float(time), float(time) + offset)
        for time in range(8)
        for offset, channel in enumerate(("pressure", "temperature"))
    ]
    return unit, observations


UNITS = [readings(f"u{number}") for number in range(1, 5)]


def registered(units: Sequence[tuple[CorpusUnit, list[Observation]]] = ()) -> Corpus:
    """A corpus holding one frozen version of the data the reader below sees."""
    corpus = Corpus(corpus_id(), "corpus", SOURCE)
    told = description(DATA, SCHEMA, units=len(units), observations=sum(len(o) for _, o in units))
    return (
        corpus.add_version(version_id(), SCHEMA, told.sampling_regime, LICENCE)
        .record_content(version_id(), told.content)
        .freeze_version(version_id(), AT)
    )


class Fixture:
    """The use case wired to adapters that touch nothing outside this process."""

    def __init__(
        self,
        corpus: Corpus,
        units: Sequence[tuple[CorpusUnit, list[Observation]]],
        workspace: Path,
    ) -> None:
        told = description(
            DATA, SCHEMA, units=len(units), observations=sum(len(o) for _, o in units)
        )
        self.corpora = InMemoryCorpusRepository()
        self.corpora.save(corpus)
        self.reader = InMemoryCorpusReader(told, units)
        self.store = InMemoryArtifactStore()
        self.archive: CorpusArchive = BlockCorpusArchive(self.store, workspace)
        self.tokenise = TokeniseCorpusVersion(
            self.corpora, self.reader, SlidingWindowTokeniser(), self.archive
        )

    def run(self, fraction: float = 0.25, seed: int = 1) -> ArtifactRef:
        return self.tokenise(
            TokeniseCorpusVersionCommand(
                corpus_id=corpus_id(),
                version_id=version_id(),
                window=WINDOW,
                validation_fraction=fraction,
                seed=seed,
            )
        )


@pytest.fixture
def wired(tmp_path: Path) -> Callable[..., Fixture]:
    """Builds the use case over in-memory adapters.

    The archive writes a block before handing it to the store, so that intermediate file needs a
    directory; the store itself never touches the disk.
    """

    def build(
        corpus: Corpus, units: Sequence[tuple[CorpusUnit, list[Observation]]] = UNITS
    ) -> Fixture:
        return Fixture(corpus, units, tmp_path / "workspace")

    return build


def test_the_published_corpus_holds_a_window_of_every_unit(wired: Callable[..., Fixture]) -> None:
    fixture = wired(registered(UNITS))

    ref = fixture.run()

    manifest = fixture.archive.read_manifest(ref)
    assert set(manifest.units) == {unit.key for unit, _ in UNITS}
    assert manifest.window_count == len(UNITS) * 3
    assert manifest.empty_units == ()


def test_the_manifest_names_the_version_and_the_data_it_was_cut_from(
    wired: Callable[..., Fixture],
) -> None:
    fixture = wired(registered(UNITS))

    ref = fixture.run()

    manifest = fixture.archive.read_manifest(ref)
    assert manifest.corpus == "corpus"
    assert manifest.corpus_version == version_id()
    assert manifest.corpus_checksum == fixture.reader.describe().content.checksum
    assert manifest.window == WINDOW
    assert manifest.split_seed == 1


def test_the_scheme_is_fitted_on_the_training_units_alone(wired: Callable[..., Fixture]) -> None:
    # Statistics fitted on every unit would carry what a held-out unit holds into the numbers a
    # model sees, which is the leak the split exists to prevent.
    fixture = wired(registered(UNITS))

    ref = fixture.run()

    manifest = fixture.archive.read_manifest(ref)
    training = [unit for unit, _ in UNITS if unit.key in manifest.split.training]
    observations = sum(len(obs) for unit, obs in UNITS if unit.key in manifest.split.training)
    assert len(training) == 3
    channel = manifest.scheme.vocabulary.id_of("corpus", "pressure")
    assert manifest.scheme.statistics_of(channel).count == observations // 2


def test_the_windows_of_the_training_units_come_back_from_the_archive(
    wired: Callable[..., Fixture],
) -> None:
    fixture = wired(registered(UNITS))

    ref = fixture.run()

    manifest = fixture.archive.read_manifest(ref)
    windows = fixture.archive.read_windows(manifest, manifest.split.training)
    assert len(windows) == 3 * len(manifest.split.training)


def test_the_same_corpus_and_command_publish_the_same_artifact(
    wired: Callable[..., Fixture],
) -> None:
    first = wired(registered(UNITS)).run()
    second = wired(registered(UNITS)).run()

    assert first == second


def test_a_unit_that_yields_no_window_is_named_rather_than_dropped_silently(
    wired: Callable[..., Fixture],
) -> None:
    silent: tuple[CorpusUnit, list[Observation]] = (
        CorpusUnit(UnitKey("u5"), TimeExtent(0.0, 8.0)),
        [],
    )
    units = [*UNITS, silent]
    fixture = wired(registered(units), units)

    ref = fixture.run()

    manifest = fixture.archive.read_manifest(ref)
    assert manifest.empty_units == (UnitKey("u5"),)
    assert UnitKey("u5") not in manifest.units


def test_data_that_no_longer_matches_the_frozen_version_is_refused(
    wired: Callable[..., Fixture],
) -> None:
    fixture = wired(registered(UNITS))
    fixture.reader.replace_data(description(b"other data", SCHEMA, units=4, observations=64), UNITS)

    with pytest.raises(CorpusDataChangedError):
        fixture.run()


def test_a_draft_version_cannot_be_published(wired: Callable[..., Fixture]) -> None:
    corpus = Corpus(corpus_id(), "corpus", SOURCE).add_version(
        version_id(), SCHEMA, description().sampling_regime, LICENCE
    )
    fixture = wired(corpus)

    with pytest.raises(CorpusVersionNotFrozenError):
        fixture.run()


def test_too_few_units_for_the_fraction_asked_is_refused(wired: Callable[..., Fixture]) -> None:
    fixture = wired(registered(UNITS))

    with pytest.raises(InvalidUnitSplitError):
        fixture.run(fraction=0.1)
