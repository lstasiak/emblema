from dataclasses import dataclass

from emblema.catalog.application.use_cases.register_corpus import (
    RegisterCorpus,
    RegisterCorpusCommand,
)
from emblema.catalog.application.use_cases.register_corpus_version import (
    RegisterCorpusVersion,
    RegisterCorpusVersionCommand,
)
from emblema.catalog.application.use_cases.tokenise_corpus_version import (
    TokeniseCorpusVersion,
    TokeniseCorpusVersionCommand,
)
from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.identifiers import CorpusId
from emblema.catalog.domain.registry.corpus_source import CorpusSource
from emblema.catalog.domain.registry.licence import Licence
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.catalog.ports.corpus_archive import CorpusArchive
from emblema.catalog.ports.corpus_reader import CorpusReader
from emblema.catalog.ports.corpus_repository import CorpusRepository
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class PublishCorpusCommand:
    """Request to take a raw corpus from registration to a published, tokenised artifact.

    Attributes:
        name: Name the corpus is registered under, and the vocabulary knows it by.
        source: Where the data comes from; recorded when the corpus is first registered.
        licence: Terms the data was obtained under; recorded when a version is first frozen.
        window: How windows are laid over each unit's time axis.
        validation_fraction: Share of units held out from fitting the scheme.
        seed: Seed the split is drawn with.
        vocabulary_from: Manifest of a corpus published earlier whose vocabulary this one
            continues, so that the two can be trained on together; ``None`` starts a vocabulary.
    """

    name: str
    source: CorpusSource
    licence: Licence
    window: WindowSpec
    validation_fraction: float
    seed: int
    vocabulary_from: ArtifactRef | None = None


class PublishCorpus:
    """Takes a corpus from raw files to a published artifact, registering only what is missing.

    Each step stays a use case of its own; this one sequences them and decides at each whether
    there is anything to do. A corpus registered under the name is reused, a version frozen over
    exactly the data the reader sees is reused, so publishing the same data twice yields the same
    manifest.
    """

    def __init__(
        self,
        register_corpus: RegisterCorpus,
        register_corpus_version: RegisterCorpusVersion,
        tokenise_corpus_version: TokeniseCorpusVersion,
        corpora: CorpusRepository,
        reader: CorpusReader,
        archive: CorpusArchive,
    ) -> None:
        self._register_corpus = register_corpus
        self._register_corpus_version = register_corpus_version
        self._tokenise_corpus_version = tokenise_corpus_version
        self._corpora = corpora
        self._reader = reader
        self._archive = archive

    def __call__(self, command: PublishCorpusCommand) -> ArtifactRef:
        """Publish; return the reference to the manifest.

        Raises:
            ArtifactNotFoundError: If the manifest named for the vocabulary is not archived.
            MalformedManifestError: If it is archived but is not a manifest.
            CorpusReadError: If the reader cannot read or validate the data.
            InvalidUnitSplitError: If the corpus holds too few units for the fraction asked.
            WindowNotArchivableError: If a window cannot be stored at the archive's precision.
        """
        vocabulary = (
            ChannelVocabulary()
            if command.vocabulary_from is None
            else self._archive.read_manifest(command.vocabulary_from).scheme.vocabulary
        )
        corpus_id, version_id = self._registered(command)
        return self._tokenise_corpus_version(
            TokeniseCorpusVersionCommand(
                corpus_id=corpus_id,
                version_id=version_id,
                window=command.window,
                validation_fraction=command.validation_fraction,
                seed=command.seed,
                vocabulary=vocabulary,
            )
        )

    def _registered(self, command: PublishCorpusCommand) -> tuple[CorpusId, CorpusVersionId]:
        """The corpus and the frozen version of the data the reader sees, registered if need be.

        The data is described here only when the corpus already exists, to look for a version of
        it; a new corpus goes straight to registration, which describes the data itself.
        """
        corpus = self._corpora.find_by_name(command.name)
        if corpus is None:
            corpus_id = self._register_corpus(
                RegisterCorpusCommand(name=command.name, source=command.source)
            )
        else:
            corpus_id = corpus.id
            frozen = corpus.frozen_version_describing(self._reader.describe())
            if frozen is not None:
                return corpus_id, frozen.id
        version = self._register_corpus_version(
            RegisterCorpusVersionCommand(corpus_id=corpus_id, licence=command.licence)
        )
        return corpus_id, version.version_id
