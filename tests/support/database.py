"""The metadata database of the local stack, for the integration tests that write to it.

Migrated the way a deployment is — by the migration tree, not by the model — so that what the
tests run against is what a process would find.
"""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text

from emblema.catalog.adapters.persistence.corpus_record import CorpusRecord
from emblema.config.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]


def migrated_engine() -> Engine:
    """An engine on the configured database, brought to the head of the migration tree."""
    command.upgrade(Config(str(REPO_ROOT / "alembic.ini")), "head")
    return create_engine(Settings().database.sqlalchemy_url())


def clear_catalog(engine: Engine) -> None:
    """Empty the Catalog's tables, so a test starts from a registry that holds nothing."""
    with engine.begin() as connection:
        tables = ", ".join(table.fullname for table in CorpusRecord.metadata.sorted_tables)
        connection.execute(text(f"TRUNCATE {tables}"))
