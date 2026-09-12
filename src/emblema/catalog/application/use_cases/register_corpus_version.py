from dataclasses import dataclass

from emblema.catalog.application.assemblers.corpus_version_ref_assembler import (
    CorpusVersionRefAssembler,
)
from emblema.catalog.contracts.corpus_version_ref import CorpusVersionRef
from emblema.catalog.contracts.events import CorpusVersionFrozen
from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.domain.identifiers import CorpusId
from emblema.catalog.domain.licence import Licence
from emblema.catalog.ports.corpus_reader import CorpusReader
from emblema.catalog.ports.corpus_repository import CorpusRepository
from emblema.shared.events.domain_event import EventId
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.event_publisher import EventPublisher
from emblema.shared.ports.id_generator import IdGenerator


@dataclass(frozen=True, kw_only=True)
class RegisterCorpusVersionCommand:
    """Request to register the data a reader currently sees as a new version of a corpus.

    Attributes:
        corpus_id: Corpus the version belongs to.
        licence: Terms the data was obtained under; a fact about the source, not the bytes, so
            the caller states it rather than the reader.
    """

    corpus_id: CorpusId
    licence: Licence


class RegisterCorpusVersion:
    """Registers what a reader sees as a new, immediately frozen version of a corpus.

    One description, one aggregate change, one save: the version is opened, filled and frozen in
    memory and the corpus is stored only once all of that succeeded, so a failed reading or data
    already frozen under another version leaves the corpus exactly as it was. The event is
    published after the save, so no subscriber learns of a version that was not stored.
    """

    def __init__(
        self,
        corpora: CorpusRepository,
        reader: CorpusReader,
        ids: IdGenerator,
        clock: Clock,
        events: EventPublisher,
        refs: CorpusVersionRefAssembler,
    ) -> None:
        self._corpora = corpora
        self._reader = reader
        self._ids = ids
        self._clock = clock
        self._events = events
        self._refs = refs

    def __call__(self, command: RegisterCorpusVersionCommand) -> CorpusVersionRef:
        """Describe, freeze and publish; return the reference other contexts may pin.

        Raises:
            CorpusNotFoundError: If the corpus is unknown; the data is not read in that case.
            CorpusReadError: If the reader cannot read or validate the data.
            SameDataAlreadyFrozenError: If a frozen version already describes the same data.
        """
        corpus = self._corpora.get(command.corpus_id)
        description = self._reader.describe()
        version_id = self._ids.generate(CorpusVersionId)
        at = self._clock.now()
        corpus = (
            corpus.add_version(
                version_id,
                description.channel_schema,
                description.sampling_regime,
                command.licence,
            )
            .record_content(version_id, description.content)
            .freeze_version(version_id, at)
        )
        self._corpora.save(corpus)
        ref = self._refs.assemble(corpus.get_version(version_id))
        self._events.publish(
            CorpusVersionFrozen(event_id=self._ids.generate(EventId), occurred_at=at, version=ref)
        )
        return ref
