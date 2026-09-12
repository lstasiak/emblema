"""The migration tree and the model describe the same database.

A table the model has and the migrations do not would be found by the first query against it, in
a process, on another machine. Here it is found by comparing the two on the migrated database of
the local stack, which is what a deployment runs against.
"""

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from emblema.catalog.adapters.persistence.corpus_record import CorpusRecord
from tests.support.database import migrated_engine

pytestmark = pytest.mark.integration


def test_the_migrations_produce_exactly_the_tables_the_model_describes() -> None:
    engine = migrated_engine()

    with engine.connect() as connection:
        context = MigrationContext.configure(
            connection,
            opts={"include_schemas": True, "compare_type": True, "compare_server_default": True},
        )
        differences = compare_metadata(context, CorpusRecord.metadata)

    assert differences == []
