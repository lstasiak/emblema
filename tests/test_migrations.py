"""The migration tree and the models describe the same database.

A table a model has and the migrations do not would be found by the first query against it, in
a process, on another machine. Here it is found by comparing the two on the migrated database of
the local stack, which is what a deployment runs against: every context's schema against its own
metadata, because the tree is one and each schema is one context's.
"""

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import MetaData

from emblema.catalog.adapters.persistence.corpus_record import CorpusRecord
from emblema.pretraining.adapters.persistence.backbone_record import BackboneRecord
from tests.support.database import migrated_engine

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "metadata",
    [CorpusRecord.metadata, BackboneRecord.metadata],
    ids=lambda metadata: str(metadata.schema),
)
def test_the_migrations_produce_exactly_the_tables_the_model_describes(metadata: MetaData) -> None:
    engine = migrated_engine()

    def within_the_schema(name: str | None, kind: str, parents: object) -> bool:
        return kind != "schema" or name == metadata.schema

    with engine.connect() as connection:
        context = MigrationContext.configure(
            connection,
            opts={
                "include_schemas": True,
                "include_name": within_the_schema,
                "compare_type": True,
                "compare_server_default": True,
            },
        )
        differences = compare_metadata(context, metadata)

    assert differences == []
