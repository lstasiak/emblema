from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import chain

from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.domain.channel_schema import ChannelSchema
from emblema.catalog.domain.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.corpus_unit import CorpusUnit
from emblema.catalog.domain.exceptions import CorpusDataChangedError
from emblema.catalog.domain.identifiers import CorpusId, UnitKey
from emblema.catalog.domain.observation import Observation
from emblema.catalog.domain.placed_window import PlacedWindow
from emblema.catalog.domain.tokenisation_manifest import TokenisationManifest
from emblema.catalog.domain.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.unit_split import UnitSplit
from emblema.catalog.domain.window_spec import WindowSpec
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
        validation_fraction: Share of units held out from fitting the scheme.
        seed: Seed the split is drawn with; recorded, so the same split can be drawn again.
        vocabulary: Channels already registered, whose identifiers this corpus's channels are
            appended after. Corpora tokenised under one growing vocabulary can be trained on
            together, because no two of their channels share an identifier; a fresh vocabulary
            is right for the first corpus only.
    """

    corpus_id: CorpusId
    version_id: CorpusVersionId
    window: WindowSpec
    validation_fraction: float
    seed: int
    vocabulary: ChannelVocabulary = field(default_factory=ChannelVocabulary)


class TokeniseCorpusVersion:
    """Turns a frozen corpus version into the artifact a training run reads instead of source files.

    Units are split before a window is cut, the scheme is fitted on the training side only and
    every unit is tokenised under that one scheme, so nothing a held-out unit holds reaches the
    statistics that normalise what a model sees.
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
            WindowNotArchivableError: If a window cannot be stored at the archive's precision.
        """
        corpus = self._corpora.get(command.corpus_id)
        version = corpus.get_version(command.version_id)
        frozen = version.frozen_content()
        description = self._reader.describe()
        if description.content.checksum != frozen.checksum:
            raise CorpusDataChangedError(
                f"version {command.version_id} froze data with checksum {frozen.checksum}, "
                f"but the reader sees {description.content.checksum}"
            )
        units = list(self._reader.read_units())
        split = UnitSplit.by_seed(
            (unit.key for unit in units), command.validation_fraction, command.seed
        )
        scheme = self._fitted(corpus.name, units, split, version.channel_schema, command.vocabulary)
        archived = self._archive.write_windows(
            self._placed(corpus.name, units, scheme, command.window)
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
            split_seed=command.seed,
            empty_units=tuple(unit.key for unit in units if unit.key not in indexed),
        )
        return self._archive.write_manifest(manifest)

    def _fitted(
        self,
        corpus: str,
        units: Sequence[CorpusUnit],
        split: UnitSplit,
        schema: ChannelSchema,
        vocabulary: ChannelVocabulary,
    ) -> TokenisationScheme:
        """The scheme this corpus is tokenised under, fitted on the training units alone."""
        training = [unit for unit in units if unit.key in split.training]
        unfitted = TokenisationScheme.for_vocabulary(vocabulary).extended_with(corpus, schema)
        return self._tokeniser.fit(
            corpus,
            self._observations(unit.key for unit in training),
            chain.from_iterable(unit.static_features for unit in training),
            unfitted,
        )

    def _observations(self, keys: Iterable[UnitKey]) -> Iterator[Observation]:
        return chain.from_iterable(self._reader.read_observations(key) for key in keys)

    def _placed(
        self,
        corpus: str,
        units: Sequence[CorpusUnit],
        scheme: TokenisationScheme,
        window: WindowSpec,
    ) -> Iterator[PlacedWindow]:
        """Every window of every unit, one unit at a time, so no corpus is ever held whole."""
        for unit in units:
            yield from self._tokeniser.tokenise(
                corpus, unit, self._reader.read_observations(unit.key), scheme, window
            )
