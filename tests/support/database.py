"""The metadata database of the local stack, for the integration tests that write to it.

Migrated the way a deployment is — by the migration tree, not by the model — so that what the
tests run against is what a process would find.
"""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, MetaData, create_engine, text

from emblema.catalog.adapters.persistence.corpus_record import CorpusRecord
from emblema.config.settings import Settings
from emblema.pretraining.adapters.persistence.backbone_record import BackboneRecord

REPO_ROOT = Path(__file__).resolve().parents[2]


def migrated_engine() -> Engine:
    """An engine on the configured database, brought to the head of the migration tree."""
    command.upgrade(Config(str(REPO_ROOT / "alembic.ini")), "head")
    return create_engine(Settings().require_database().sqlalchemy_url())


def clear_catalog(engine: Engine) -> None:
    """Empty the Catalog's tables, so a test starts from a registry that holds nothing."""
    _truncate(engine, CorpusRecord.metadata)


def clear_pretraining(engine: Engine) -> None:
    """Empty Pretraining's tables, so a test starts from a registry that holds nothing."""
    _truncate(engine, BackboneRecord.metadata)


def _truncate(engine: Engine, metadata: MetaData) -> None:
    with engine.begin() as connection:
        tables = ", ".join(table.fullname for table in metadata.sorted_tables)
        connection.execute(text(f"TRUNCATE {tables}"))
