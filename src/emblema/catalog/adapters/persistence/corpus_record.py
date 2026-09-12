from typing import Self
from uuid import UUID

from sqlalchemy import Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from emblema.catalog.adapters.persistence.corpus_version_record import CorpusVersionRecord
from emblema.catalog.adapters.persistence.orm import Base
from emblema.catalog.domain.corpus import Corpus
from emblema.catalog.domain.corpus_source import CorpusSource
from emblema.catalog.domain.identifiers import CorpusId


class CorpusRecord(Base):
    """Row of ``catalog.corpus`` with its versions: the persistence model of the aggregate.

    The aggregate stays a value; this class is how it is laid in tables and rebuilt from them,
    and the two conversions are the whole mapping. Versions are owned: a version no longer in the
    collection is deleted with the next flush, which is what lets a whole state replace another.
    """

    __tablename__ = "corpus"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True)
    source_name: Mapped[str] = mapped_column(Text)
    source_uri: Mapped[str] = mapped_column(Text)
    versions: Mapped[list[CorpusVersionRecord]] = relationship(
        cascade="all, delete-orphan", order_by=CorpusVersionRecord.number, lazy="selectin"
    )

    @classmethod
    def from_corpus(cls, corpus: Corpus) -> Self:
        return cls(
            id=corpus.id.value,
            name=corpus.name,
            source_name=corpus.source.name,
            source_uri=corpus.source.uri,
            versions=[
                CorpusVersionRecord.from_version(corpus.id, version) for version in corpus.versions
            ],
        )

    def to_corpus(self) -> Corpus:
        return Corpus(
            CorpusId(self.id),
            self.name,
            CorpusSource(self.source_name, self.source_uri),
            tuple(record.to_version() for record in self.versions),
        )
