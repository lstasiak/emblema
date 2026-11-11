from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from itertools import chain

from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.channels.normalisation import Normalisation
from emblema.catalog.domain.cleaning.cleaning_policy import CleaningPolicy
from emblema.catalog.domain.cleaning.cleaning_record import ChannelCleaning, CleaningRecord
from emblema.catalog.domain.exceptions import (
    CorpusDataChangedError,
    InvalidTokenisationSchemeError,
    UnknownChannelError,
)
from emblema.catalog.domain.identifiers import CorpusId, UnitKey
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.tokenisation.placed_window import PlacedWindow
from emblema.catalog.domain.tokenisation.split_policy import SplitPolicy
from emblema.catalog.domain.tokenisation.tokenisation_manifest import TokenisationManifest
from emblema.catalog.domain.tokenisation.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.tokenisation.unit_split import UnitSplit
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.catalog.ports.corpus_archive import CorpusArchive
from emblema.catalog.ports.corpus_reader import CorpusReader
from emblema.catalog.ports.corpus_repository import CorpusRepository
from emblema.catalog.ports.tokeniser import Tokeniser
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class TokeniseCorpusVersionCommand:
    """Request to turn a frozen version of a corpus into a published, tokenised artifact.

    Attributes:
        corpus_id: Corpus the version belongs to.
        version_id: Version to tokenise; must be frozen.
        window: How windows are laid over each unit's time axis.
        split: How the units held out from fitting the scheme are chosen; recorded with the
            publication, so the same split can be made again.
        vocabulary: Channels already registered, whose identifiers this corpus's channels are
            appended after. Corpora tokenised under one growing vocabulary can be trained on
            together, because no two of their channels share an identifier; a fresh vocabulary
            is right for the first corpus only.
        statistics_from: Manifest of a corpus whose statistics this one's channels adopt rather
            than fit on its training side, so that the two hand a model the same values;
            ``None`` fits them.
        cleaning: Bounds the corpus's values are held to before fitting and tokenising;
            ``None`` reads them raw. Statistics adopted from another manifest bring the bounds
            that manifest's values were held to, so none is named beside them.
        normalisation: How the statistics fitted here normalise a value; statistics adopted
            from another manifest bring their own.
    """

    corpus_id: CorpusId
    version_id: CorpusVersionId
    window: WindowSpec
    split: SplitPolicy
    vocabulary: ChannelVocabulary = field(default_factory=ChannelVocabulary)
    statistics_from: ArtifactRef | None = None
    cleaning: CleaningPolicy | None = None
    normalisation: Normalisation = Normalisation.STANDARD


class TokeniseCorpusVersion:
    """Turns a frozen corpus version into the artifact a training run reads instead of source files.

    Units are split before a window is cut, the scheme is fitted on the training side only and
    every unit is tokenised under that one scheme, so nothing a held-out unit holds reaches the
    statistics that normalise what a model sees. A cleaning policy is applied to both passes, so
    the statistics describe the values the tokens carry.
    """

    def __init__(
        self,
        corpora: CorpusRepository,
        reader: CorpusReader,
        tokeniser: Tokeniser,
        archive: CorpusArchive,
    ) -> None:
        self._corpora = corpora
        self._reader = reader
        self._tokeniser = tokeniser
        self._archive = archive

    def __call__(self, command: TokeniseCorpusVersionCommand) -> ArtifactRef:
        """Tokenise, archive and describe; return the reference to the manifest.

        Raises:
            CorpusNotFoundError: If the corpus is unknown.
            ChannelRedeclaredError: If the vocabulary already registers a channel of this corpus
                with another unit or kind.
            CorpusVersionNotFoundError: If the corpus has no such version.
            CorpusVersionNotFrozenError: If the version is still a draft.
            CorpusDataChangedError: If the reader no longer sees the data that version froze.
            CorpusReadError: If the reader cannot read or validate the data.
            InvalidUnitSplitError: If the corpus holds too few units for the fraction asked.
            ArtifactNotFoundError: If the manifest named for the statistics is not archived.
            MalformedManifestError: If it is archived but is not a manifest.
            MissingChannelStatisticsError: If it fitted no statistics for a channel of this
                corpus.
            WindowNotArchivableError: If a window cannot be stored at the archive's precision.
            UnknownChannelError: If the cleaning policy bounds a channel the corpus does not
                declare, most likely a misspelt one.
            InvalidTokenisationSchemeError: If a normalisation or a cleaning policy is named for
                statistics adopted from another manifest, which bring their own.
        """
        if command.statistics_from is not None and (
            command.normalisation is not Normalisation.STANDARD or command.cleaning is not None
        ):
            raise InvalidTokenisationSchemeError(
                "statistics adopted from another manifest bring their own normalisation and the "
                "bounds their values were held to; neither can be named beside them"
            )
        source = (
            None
            if command.statistics_from is None
            else self._archive.read_manifest(command.statistics_from)
        )
        policy = command.cleaning if source is None else _adopted_policy(source)
        corpus = self._corpora.get(command.corpus_id)
        version = corpus.get_version(command.version_id)
        if policy is not None:
            undeclared = sorted(policy.channels - set(version.channel_schema.names))
            if undeclared:
                raise UnknownChannelError(
                    f"the cleaning policy bounds channels {corpus.name!r} does not declare: "
                    f"{undeclared}"
                )
        frozen = version.frozen_content()
        description = self._reader.describe()
        if description.content.checksum != frozen.checksum:
            raise CorpusDataChangedError(
                f"version {command.version_id} froze data with checksum {frozen.checksum}, "
                f"but the reader sees {description.content.checksum}"
            )
        units = list(self._reader.read_units())
        split = command.split.applied_to([unit.key for unit in units])
        unfitted = TokenisationScheme.for_vocabulary(command.vocabulary).extended_with(
            corpus.name, version.channel_schema
        )
        fitting, publishing = _Cleaning(policy), _Cleaning(policy, counts=True)
        scheme = (
            self._fitted(corpus.name, units, split, unfitted, fitting, command.normalisation)
            if source is None
            else unfitted.adopting(corpus.name, source.scheme)
        )
        archived = self._archive.write_windows(
            self._placed(corpus.name, units, scheme, command.window, publishing)
        )
        indexed = set(archived.units)
        manifest = TokenisationManifest(
            corpus=corpus.name,
            corpus_version=command.version_id,
            corpus_checksum=frozen.checksum,
            archived=archived,
            window=command.window,
            scheme=scheme,
            split=split,
            split_seed=command.split.recorded_seed,
            empty_units=tuple(unit.key for unit in units if unit.key not in indexed),
            statistics_from=command.statistics_from,
            cleaning=publishing.record(),
        )
        return self._archive.write_manifest(manifest)

    def _fitted(
        self,
        corpus: str,
        units: Sequence[CorpusUnit],
        split: UnitSplit,
        unfitted: TokenisationScheme,
        cleaning: "_Cleaning",
        normalisation: Normalisation,
    ) -> TokenisationScheme:
        """The scheme this corpus is tokenised under, fitted on the training units alone."""
        training = [cleaning.unit(unit) for unit in units if unit.key in split.training]
        return self._tokeniser.fit(
            corpus,
            cleaning.observations(self._observations(unit.key for unit in training)),
            chain.from_iterable(unit.static_features for unit in training),
            unfitted,
            normalisation,
        )

    def _observations(self, keys: Iterable[UnitKey]) -> Iterator[Observation]:
        return chain.from_iterable(self._reader.read_observations(key) for key in keys)

    def _placed(
        self,
        corpus: str,
        units: Sequence[CorpusUnit],
        scheme: TokenisationScheme,
        window: WindowSpec,
        cleaning: "_Cleaning",
    ) -> Iterator[PlacedWindow]:
        """Every window of every unit, one unit at a time, so no corpus is ever held whole."""
        for unit in units:
            yield from self._tokeniser.tokenise(
                corpus,
                cleaning.unit(unit),
                cleaning.observations(self._reader.read_observations(unit.key)),
                scheme,
                window,
            )


def _adopted_policy(source: TokenisationManifest) -> CleaningPolicy | None:
    """The bounds the values of a manifest whose statistics are adopted were held to.

    Its statistics describe values held to those bounds; tokens held to others, or to none, would
    be normalised by statistics of values they are not.
    """
    return None if source.cleaning is None else source.cleaning.policy


class _Cleaning:
    """A cleaning policy applied to a stream of values, counting what it changed when asked to.

    The fitting pass reads the training units only, the tokenising pass every unit; only the
    second counts, so the record states what the published values lost, held-out units included.
    """

    def __init__(self, policy: CleaningPolicy | None, *, counts: bool = False) -> None:
        self._policy = policy
        self._counts = counts
        self._removed: dict[str, int] = {}
        self._clipped: dict[str, int] = {}

    def unit(self, unit: CorpusUnit) -> CorpusUnit:
        if self._policy is None:
            return unit
        kept = []
        for feature in unit.static_features:
            cleaned = self._policy.cleaned_feature(feature)
            self._count(feature.channel, feature.value, None if cleaned is None else cleaned.value)
            if cleaned is not None:
                kept.append(cleaned)
        return replace(unit, static_features=tuple(kept))

    def observations(self, observations: Iterable[Observation]) -> Iterator[Observation]:
        if self._policy is None:
            yield from observations
            return
        for observation in observations:
            cleaned = self._policy.cleaned_observation(observation)
            self._count(
                observation.channel, observation.value, None if cleaned is None else cleaned.value
            )
            if cleaned is not None:
                yield cleaned

    def record(self) -> CleaningRecord | None:
        if self._policy is None:
            return None
        return CleaningRecord(
            policy=self._policy,
            counts=tuple(
                ChannelCleaning(
                    channel=each.channel,
                    removed=self._removed.get(each.channel, 0),
                    clipped=self._clipped.get(each.channel, 0),
                )
                for each in self._policy.bounds
            ),
        )

    def _count(self, channel: str, value: float, cleaned: float | None) -> None:
        if not self._counts:
            return
        if cleaned is None:
            self._removed[channel] = self._removed.get(channel, 0) + 1
        elif cleaned != value:
            self._clipped[channel] = self._clipped.get(channel, 0) + 1
