import math
from collections.abc import Callable, Sequence
from dataclasses import replace
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
from emblema.catalog.domain.channels.channel_statistics import QUANTILE_KNOTS, ROBUST_BOUND
from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.channels.normalisation import Normalisation
from emblema.catalog.domain.cleaning.cleaning_policy import CleaningPolicy
from emblema.catalog.domain.cleaning.cleaning_record import ChannelCleaning
from emblema.catalog.domain.cleaning.value_bounds import ValueBounds
from emblema.catalog.domain.exceptions import (
    ChannelRedeclaredError,
    CorpusDataChangedError,
    InvalidTokenisationSchemeError,
    UnknownChannelError,
)
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.domain.registry.corpus import Corpus
from emblema.catalog.domain.tokenisation.split_policy import (
    NamedSplit,
    SeededSplit,
    SplitPolicy,
)
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
    STATIC_SCHEMA,
    corpus_id,
    description,
    measured_units,
    version_id,
)

DATA = b"records"
WINDOW = WindowSpec(length=4.0, stride=2.0)
UNITS = measured_units(4)


def registered(
    units: Sequence[tuple[CorpusUnit, list[Observation]]] = (), schema: ChannelSchema = SCHEMA
) -> Corpus:
    """A corpus holding one frozen version of the data the reader below sees."""
    corpus = Corpus(corpus_id(), "corpus", SOURCE)
    told = description(DATA, schema, units=len(units), observations=sum(len(o) for _, o in units))
    return (
        corpus.add_version(version_id(), schema, told.sampling_regime, LICENCE)
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
        schema: ChannelSchema = SCHEMA,
    ) -> None:
        told = description(
            data, schema, units=len(units), observations=sum(len(o) for _, o in units)
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
        split: SplitPolicy | None = None,
        vocabulary: ChannelVocabulary | None = None,
        statistics_from: ArtifactRef | None = None,
        cleaning: CleaningPolicy | None = None,
        normalisation: Normalisation = Normalisation.STANDARD,
        robust_bound: float | None = None,
    ) -> ArtifactRef:
        return self.tokenise(
            TokeniseCorpusVersionCommand(
                corpus_id=corpus_id(),
                version_id=version_id(),
                window=WINDOW,
                split=split if split is not None else SeededSplit(fraction, seed),
                vocabulary=ChannelVocabulary() if vocabulary is None else vocabulary,
                statistics_from=statistics_from,
                cleaning=cleaning,
                normalisation=normalisation,
                robust_bound=robust_bound,
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
        schema: ChannelSchema = SCHEMA,
    ) -> Fixture:
        return Fixture(corpus, units, tmp_path / "workspace", data, schema)

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


def test_statistics_adopted_from_an_earlier_manifest_are_its_and_are_recorded(
    wired: Callable[..., Fixture],
) -> None:
    # A corpus cut from the data of an earlier one, to be read beside it, normalises as it did:
    # another split here fits nothing, and the manifest says where the statistics came from.
    fixture = wired(registered(UNITS))
    earlier = fixture.run(fraction=0.25, seed=1)

    ref = fixture.run(fraction=0.5, seed=2, statistics_from=earlier)

    adopted, fitted = fixture.archive.read_manifest(ref), fixture.archive.read_manifest(earlier)
    assert adopted.split != fitted.split
    assert adopted.scheme.statistics == fitted.scheme.statistics
    assert adopted.statistics_from == earlier
    assert fitted.statistics_from is None
    training = adopted.split.training
    assert list(fixture.archive.read_windows(adopted.archived, training)) == list(
        fixture.archive.read_windows(fitted.archived, training)
    )


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


def test_a_publication_that_names_its_held_out_units_records_them_and_no_seed(
    wired: Callable[..., Fixture],
) -> None:
    """A corpus whose units differ in kind is split by naming, and the names are what it kept."""
    fixture = wired(registered(UNITS))
    named = UNITS[0][0].key

    manifest = fixture.archive.read_manifest(fixture.run(split=NamedSplit.of((named,))))

    assert manifest.split.validation == frozenset({named})
    assert len(manifest.split.training) == len(UNITS) - 1
    # Nothing was drawn, so the publication records the names it was given and no seed.
    assert manifest.split_seed is None


# Every unit reads pressure 0 to 7, one value an instant: 6 and 7 lie past the outer bound and 5
# between the bounds, so each unit loses two values and has one clipped.
PRESSURE_HELD = CleaningPolicy(
    (
        ValueBounds(
            channel="pressure",
            removed_below=0.0,
            clipped_below=0.0,
            clipped_above=4.5,
            removed_above=5.5,
        ),
    )
)


def test_cleaned_values_are_the_ones_fitted_and_the_ones_tokenised(
    wired: Callable[..., Fixture],
) -> None:
    fixture = wired(registered(UNITS))

    manifest = fixture.archive.read_manifest(fixture.run(cleaning=PRESSURE_HELD))

    channel = manifest.scheme.vocabulary.id_of("corpus", "pressure")
    statistics = manifest.scheme.statistics_of(channel)
    assert statistics.count == 3 * 6
    windows = fixture.archive.read_windows(manifest.archived, manifest.split.training)
    raw = [
        statistics.denormalise(value)
        for window in windows
        for channel_id, value in zip(window.channel_ids, window.values, strict=True)
        if channel_id == channel
    ]
    assert max(raw) == pytest.approx(4.5)


def test_the_manifest_counts_what_cleaning_removed_and_clipped_over_every_unit(
    wired: Callable[..., Fixture],
) -> None:
    fixture = wired(registered(UNITS))

    manifest = fixture.archive.read_manifest(fixture.run(cleaning=PRESSURE_HELD))

    assert manifest.cleaning is not None
    assert manifest.cleaning.policy == PRESSURE_HELD
    assert manifest.cleaning.counts == (
        ChannelCleaning(channel="pressure", removed=2 * len(UNITS), clipped=len(UNITS)),
    )


def test_a_publication_without_cleaning_records_none(wired: Callable[..., Fixture]) -> None:
    fixture = wired(registered(UNITS))

    assert fixture.archive.read_manifest(fixture.run()).cleaning is None


def test_bounds_on_a_channel_the_corpus_does_not_declare_are_refused(
    wired: Callable[..., Fixture],
) -> None:
    # A misspelt channel would otherwise leave its values raw without a word.
    misspelt = CleaningPolicy(
        (
            ValueBounds(
                channel="presure",
                removed_below=0.0,
                clipped_below=0.0,
                clipped_above=1.0,
                removed_above=1.0,
            ),
        )
    )
    fixture = wired(registered(UNITS))

    with pytest.raises(UnknownChannelError, match="presure"):
        fixture.run(cleaning=misspelt)


@pytest.mark.parametrize("normalisation", [Normalisation.ROBUST, Normalisation.QUANTILE])
def test_the_normalisation_named_is_fitted_and_survives_the_archive(
    wired: Callable[..., Fixture], normalisation: Normalisation
) -> None:
    fixture = wired(registered(UNITS))

    manifest = fixture.archive.read_manifest(fixture.run(normalisation=normalisation))

    channel = manifest.scheme.vocabulary.id_of("corpus", "temperature")
    statistics = manifest.scheme.statistics_of(channel)
    assert statistics.normalisation is normalisation
    if normalisation is Normalisation.ROBUST:
        # Every training unit reads temperature 1 to 8: the median is 4.5, its deviation 2.
        assert statistics.median == pytest.approx(4.5)
        assert statistics.mad == pytest.approx(2.0)
        assert statistics.bound == ROBUST_BOUND
    else:
        assert len(statistics.quantiles) == QUANTILE_KNOTS
        assert statistics.quantiles[0] == 1.0
        assert statistics.quantiles[-1] == 8.0


def test_the_bound_named_holds_the_robust_score_and_survives_the_archive(
    wired: Callable[..., Fixture],
) -> None:
    fixture = wired(registered(UNITS))

    manifest = fixture.archive.read_manifest(
        fixture.run(normalisation=Normalisation.ROBUST, robust_bound=100.0)
    )

    channel = manifest.scheme.vocabulary.id_of("corpus", "temperature")
    assert manifest.scheme.statistics_of(channel).bound == 100.0


@pytest.mark.parametrize("normalisation", [Normalisation.STANDARD, Normalisation.QUANTILE])
def test_a_bound_beside_another_normalisation_than_the_robust_score_is_refused(
    wired: Callable[..., Fixture], normalisation: Normalisation
) -> None:
    fixture = wired(registered(UNITS))

    with pytest.raises(InvalidTokenisationSchemeError, match="belongs to the robust score"):
        fixture.run(normalisation=normalisation, robust_bound=100.0)


@pytest.mark.parametrize("bound", [0.0, -1.0, math.inf, math.nan])
def test_a_bound_that_cannot_hold_a_score_is_refused(
    wired: Callable[..., Fixture], bound: float
) -> None:
    fixture = wired(registered(UNITS))

    with pytest.raises(InvalidTokenisationSchemeError, match="finite positive bound"):
        fixture.run(normalisation=Normalisation.ROBUST, robust_bound=bound)


@pytest.mark.parametrize(
    ("normalisation", "cleaning", "robust_bound"),
    [
        (Normalisation.QUANTILE, None, None),
        (Normalisation.STANDARD, PRESSURE_HELD, None),
        (Normalisation.STANDARD, None, 100.0),
    ],
    ids=["normalisation", "cleaning", "bound"],
)
def test_what_adopted_statistics_bring_cannot_be_named_beside_them(
    wired: Callable[..., Fixture],
    normalisation: Normalisation,
    cleaning: CleaningPolicy | None,
    robust_bound: float | None,
) -> None:
    fixture = wired(registered(UNITS))
    earlier = fixture.run()

    with pytest.raises(InvalidTokenisationSchemeError, match="bring their own"):
        fixture.run(
            statistics_from=earlier,
            normalisation=normalisation,
            cleaning=cleaning,
            robust_bound=robust_bound,
        )


def test_adopted_statistics_bring_the_bounds_their_values_were_held_to(
    wired: Callable[..., Fixture],
) -> None:
    # Statistics fitted on cleaned values describe nothing of a raw value they would normalise.
    fixture = wired(registered(UNITS))
    earlier = fixture.run(fraction=0.25, seed=1, cleaning=PRESSURE_HELD)

    ref = fixture.run(fraction=0.5, seed=2, statistics_from=earlier)

    adopted, fitted = fixture.archive.read_manifest(ref), fixture.archive.read_manifest(earlier)
    assert adopted.cleaning == fitted.cleaning
    assert adopted.scheme.statistics == fitted.scheme.statistics


def test_adopting_statistics_of_raw_values_reads_this_corpus_raw(
    wired: Callable[..., Fixture],
) -> None:
    fixture = wired(registered(UNITS))
    earlier = fixture.run(fraction=0.25, seed=1)

    adopted = fixture.archive.read_manifest(
        fixture.run(fraction=0.5, seed=2, statistics_from=earlier)
    )

    assert adopted.cleaning is None


AGED = [
    (replace(unit, static_features=(StaticFeature("age", age),)), observations)
    for (unit, observations), age in zip(UNITS, (40.0, 300.0, 70.0, 95.0), strict=True)
]
AGE_HELD = CleaningPolicy(
    (
        ValueBounds(
            channel="age",
            removed_below=0.0,
            clipped_below=18.0,
            clipped_above=90.0,
            removed_above=120.0,
        ),
    )
)


def test_a_descriptor_is_cleaned_before_it_is_fitted_and_counted_over_every_unit(
    wired: Callable[..., Fixture],
) -> None:
    fixture = wired(registered(AGED, STATIC_SCHEMA), AGED, schema=STATIC_SCHEMA)

    manifest = fixture.archive.read_manifest(fixture.run(cleaning=AGE_HELD))

    assert manifest.cleaning is not None
    assert manifest.cleaning.counts == (ChannelCleaning(channel="age", removed=1, clipped=1),)
    age = manifest.scheme.statistics_of(manifest.scheme.vocabulary.id_of("corpus", "age"))
    training = [
        min(float(a), 90.0)
        for (unit, _), a in zip(UNITS, (40.0, 300.0, 70.0, 95.0), strict=True)
        if unit.key in manifest.split.training and a <= 120.0
    ]
    assert age.count == len(training)
    assert age.mean == pytest.approx(sum(training) / len(training))


def test_a_channel_whose_every_value_is_removed_is_published_without_statistics_or_tokens(
    wired: Callable[..., Fixture],
) -> None:
    # Bounds that remove a whole channel leave nothing to fit and nothing to tokenise, as a
    # channel the corpus never measured; the counts say where its values went.
    every_pressure = CleaningPolicy(
        (
            ValueBounds(
                channel="pressure",
                removed_below=100.0,
                clipped_below=100.0,
                clipped_above=200.0,
                removed_above=200.0,
            ),
        )
    )
    fixture = wired(registered(UNITS))

    manifest = fixture.archive.read_manifest(fixture.run(cleaning=every_pressure))

    pressure = manifest.scheme.vocabulary.id_of("corpus", "pressure")
    assert manifest.scheme.statistics[pressure - 1] is None
    windows = fixture.archive.read_windows(manifest.archived, manifest.split.training)
    assert all(pressure not in window.channel_ids for window in windows)
    assert manifest.cleaning is not None
    assert manifest.cleaning.counts[0].removed == 8 * len(UNITS)
