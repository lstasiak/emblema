from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from emblema.catalog.adapters.persistence.corpus_record import CorpusRecord
from emblema.catalog.domain.corpus import Corpus
from emblema.catalog.domain.exceptions import CorpusNameTakenError, CorpusNotFoundError
from emblema.catalog.domain.identifiers import CorpusId


class SqlAlchemyCorpusRepository:
    """Repository over the Catalog schema of the metadata database.

    A corpus is read whole and written whole: ``save`` merges the record of the state the caller
    holds, so the row is inserted or updated by identity and versions the aggregate no longer
    holds are deleted, in one transaction. The uniqueness of names is a constraint of the schema,
    surfaced as the domain's error; any other integrity failure is a bug and propagates.
    """

    NAME_CONSTRAINT = "uq_corpus_name"

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get(self, corpus_id: CorpusId) -> Corpus:
        with Session(self._engine) as session:
            record = session.get(CorpusRecord, corpus_id.value)
            if record is None:
                raise CorpusNotFoundError(f"no corpus {corpus_id}")
            return record.to_corpus()

    def find_by_name(self, name: str) -> Corpus | None:
        with Session(self._engine) as session:
            record = session.scalars(
                select(CorpusRecord).where(CorpusRecord.name == name)
            ).one_or_none()
            return None if record is None else record.to_corpus()

    def save(self, corpus: Corpus) -> None:
        try:
            with Session(self._engine) as session, session.begin():
                session.merge(CorpusRecord.from_corpus(corpus))
        except IntegrityError as error:
            if self._violates_name_uniqueness(error):
                raise CorpusNameTakenError(f"corpus name {corpus.name!r} is taken") from error
            raise

    @classmethod
    def _violates_name_uniqueness(cls, error: IntegrityError) -> bool:
        diagnostics = getattr(error.orig, "diag", None)
        return getattr(diagnostics, "constraint_name", None) == cls.NAME_CONSTRAINT
