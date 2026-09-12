from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from emblema.catalog.adapters.archive.block_corpus_archive import BlockCorpusArchive
from emblema.catalog.adapters.in_memory.corpus_reader import InMemoryCorpusReader
from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.application.assemblers.published_corpus_manifest_assembler import (
    PublishedCorpusManifestAssembler,
)
from emblema.catalog.application.use_cases.tokenise_corpus_version import (
    TokeniseCorpusVersion,
    TokeniseCorpusVersionCommand,
)
from emblema.catalog.domain.channels.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.exceptions import (
    ChannelRedeclaredError,
    CorpusDataChangedError,
)
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.registry.corpus import Corpus
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.catalog.ports.corpus_archive import CorpusArchive
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from tests.catalog.domain.support import (
    AT,
    LICENCE,
    OTHER_SCHEMA,
    SCHEMA,
    SOURCE,
    corpus_id,
    description,
    measured_units,
    version_id,
)

DATA = b"records"
WINDOW = WindowSpec(length=4.0, stride=2.0)
UNITS = measured_units(4)


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
        data: bytes = DATA,
    ) -> None:
        told = description(
            data, SCHEMA, units=len(units), observations=sum(len(o) for _, o in units)
        )
        self.corpora = InMemoryCorpusRepository()
        self.corpora.save(corpus)
        self.reader = InMemoryCorpusReader(told, units)
        self.store = InMemoryArtifactStore()
        self.archive: CorpusArchive = BlockCorpusArchive(
            self.store, workspace, PublishedCorpusManifestAssembler()
        )
        self.tokenise = TokeniseCorpusVersion(
            self.corpora, self.reader, SlidingWindowTokeniser(), self.archive
        )

    def run(
        self,
        fraction: float = 0.25,
        seed: int = 1,
        vocabulary: ChannelVocabulary | None = None,
    ) -> ArtifactRef:
        return self.tokenise(
            TokeniseCorpusVersionCommand(
                corpus_id=corpus_id(),
                version_id=version_id(),
                window=WINDOW,
                validation_fraction=fraction,
                seed=seed,
                vocabulary=ChannelVocabulary() if vocabulary is None else vocabulary,
            )
        )


@pytest.fixture
def wired(tmp_path: Path) -> Callable[..., Fixture]:
    """Builds the use case over in-memory adapters.

    The archive writes a block before handing it to the store, so that intermediate file needs a
    directory; the store itself never touches the disk.
    """

    def build(
        corpus: Corpus,
        units: Sequence[tuple[CorpusUnit, list[Observation]]] = UNITS,
        data: bytes = DATA,
    ) -> Fixture:
        return Fixture(corpus, units, tmp_path / "workspace", data)

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
    windows = fixture.archive.read_windows(manifest.archived, manifest.split.training)
    assert len(windows) == 3 * len(manifest.split.training)


def test_the_same_corpus_and_command_publish_the_same_artifact(
    wired: Callable[..., Fixture],
) -> None:
    first = wired(registered(UNITS)).run()
    second = wired(registered(UNITS)).run()

    assert first == second


def test_a_vocabulary_given_keeps_its_identifiers_and_grows_by_this_corpus(
    wired: Callable[..., Fixture],
) -> None:
    # Identifiers index a model's embedding table, so a corpus published to be trained on beside
    # an earlier one takes the identifiers after that one's, and leaves those untouched.
    prior = ChannelVocabulary().extended_with("other", OTHER_SCHEMA)
    fixture = wired(registered(UNITS))

    ref = fixture.run(vocabulary=prior)

    scheme = fixture.archive.read_manifest(ref).scheme
    assert scheme.vocabulary.entry(1) == prior.entry(1)
    assert scheme.vocabulary.id_of("corpus", "pressure") == 2
    assert scheme.vocabulary.id_of("corpus", "temperature") == 3
    assert scheme.statistics[0] is None


def test_a_reader_no_longer_showing_the_frozen_data_stops_the_run(
    wired: Callable[..., Fixture],
) -> None:
    # A frozen version stands for particular bytes. If the files under the reader have been
    # replaced since, the artifact would carry that version's identity over data it never froze,
    # and every run pinned to the identifier would silently train on something else.
    fixture = wired(registered(UNITS), data=b"the files were replaced")

    with pytest.raises(CorpusDataChangedError, match="checksum"):
        fixture.run()


def test_a_vocabulary_declaring_a_channel_of_this_corpus_differently_is_refused(
    wired: Callable[..., Fixture],
) -> None:
    # An identifier cannot change meaning: pressure in bar under an identifier that a published
    # artifact already reads as pressure in pascal would make one embedding stand for two things.
    fixture = wired(registered(UNITS))
    redeclared = ChannelVocabulary().extended_with(
        "corpus", ChannelSchema(frozenset({Channel("pressure", "bar")}))
    )

    with pytest.raises(ChannelRedeclaredError):
        fixture.run(vocabulary=redeclared)
