"""Alembic environment: one tree for every context's schema, connected as the application is."""

from alembic import context
from sqlalchemy import create_engine

from emblema.catalog.adapters.persistence.corpus_record import CorpusRecord
from emblema.config.settings import Settings

# Every context's tables, so that a comparison covers the whole database.
target_metadata = [CorpusRecord.metadata]


def run_migrations_offline() -> None:
    context.configure(
        url=Settings().database.sqlalchemy_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        include_schemas=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(Settings().database.sqlalchemy_url())
    with engine.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, include_schemas=True
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
