"""The database the integration tests own, migrated the way a deployment is.

The tests write to a database of their own, named after the configured one with ``_test``
appended, never to the configured one itself: they empty tables between tests, and the
configured database is the development registry, where a backbone ordered on this machine waits
for the result a notebook is training. Whoever calls ``migrated_engine`` gets the test database
created if it is missing, migrated by the migration tree rather than by the model — so that what
the tests run against is what a process would find — and named to every ``Settings()`` built
afterwards in this process, so that Alembic's environment and the composition roots under test
reach it without knowing it exists.
"""

import os
import re
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, MetaData, create_engine, text

from emblema.catalog.adapters.persistence.corpus_record import CorpusRecord
from emblema.config.database_settings import DatabaseSettings
from emblema.config.settings import Settings
from emblema.evaluation.adapters.persistence.evaluation_campaign_record import (
    EvaluationCampaignRecord,
)
from emblema.pretraining.adapters.persistence.backbone_record import BackboneRecord
from emblema.serving.adapters.persistence.promotable_artifact_record import (
    PromotableArtifactRecord,
)

# Imported so that emptying Serving empties the served models too, whichever records the test
# that asks for it happened to import.
from emblema.serving.adapters.persistence.served_model_record import (  # noqa: F401
    ServedModelRecord,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_SUFFIX = "_test"
DATABASE_NAME_VARIABLE = "EMBLEMA_DATABASE__NAME"
# The database is named in SQL by the tests, so its name is kept to what needs no quoting.
_PLAIN_NAME = re.compile(r"[a-z_][a-z0-9_]*")
# Where a connection goes to create a database: the server's own, which every server has.
_MAINTENANCE_DATABASE = "postgres"


class NotATestDatabaseError(RuntimeError):
    """The tests were about to empty a database that is not theirs."""


def name_test_database() -> DatabaseSettings:
    """The test database's settings, named to this process's environment.

    The name is the configured one with the suffix, unless it carries it already. It is put into
    the environment so that every ``Settings()`` built afterwards — by Alembic's environment, by a
    composition root a test assembles — names the test database rather than the configured one.

    Raises:
        NotATestDatabaseError: If the configured name is not one the tests can name in SQL.
    """
    configured = Settings().require_database()
    name = configured.name
    if not name.endswith(TEST_SUFFIX):
        name = f"{name}{TEST_SUFFIX}"
    if _PLAIN_NAME.fullmatch(name) is None:
        raise NotATestDatabaseError(f"{name!r} is not a database name the tests can create")
    os.environ[DATABASE_NAME_VARIABLE] = name
    return configured.model_copy(update={"name": name})


def create_test_database() -> DatabaseSettings:
    """The test database, named, created if it is missing and at the head of the migration tree.

    Raises:
        NotATestDatabaseError: If the configured name is not one the tests can name in SQL.
    """
    settings = name_test_database()
    _create_if_missing(settings)
    command.upgrade(Config(str(REPO_ROOT / "alembic.ini")), "head")
    return settings


def migrated_engine() -> Engine:
    """An engine on the test database, set up by ``create_test_database``."""
    return create_engine(create_test_database().sqlalchemy_url())


def clear_catalog(engine: Engine) -> None:
    """Empty the Catalog's tables, so a test starts from a registry that holds nothing."""
    _truncate(engine, CorpusRecord.metadata)


def clear_pretraining(engine: Engine) -> None:
    """Empty Pretraining's tables, so a test starts from a registry that holds nothing."""
    _truncate(engine, BackboneRecord.metadata)


def clear_evaluation(engine: Engine) -> None:
    """Empty Evaluation's tables, so a test starts from a registry that holds nothing."""
    _truncate(engine, EvaluationCampaignRecord.metadata)


def clear_serving(engine: Engine) -> None:
    """Empty Serving's tables, so a test starts from a registry that holds nothing."""
    _truncate(engine, PromotableArtifactRecord.metadata)


def _create_if_missing(settings: DatabaseSettings) -> None:
    # A database is created outside any transaction, so the service connection autocommits; the
    # initialisation scripts of the local stack run on an empty volume only, and a stack that
    # was started before the tests had a database of their own has to get it from here.
    maintenance = create_engine(
        settings.model_copy(update={"name": _MAINTENANCE_DATABASE}).sqlalchemy_url(),
        isolation_level="AUTOCOMMIT",
    )
    try:
        with maintenance.connect() as connection:
            found = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": settings.name}
            ).scalar()
            if found is None:
                connection.execute(text(f"CREATE DATABASE {settings.name}"))
    finally:
        maintenance.dispose()


def _truncate(engine: Engine, metadata: MetaData) -> None:
    name = engine.url.database
    if name is None or not name.endswith(TEST_SUFFIX):
        raise NotATestDatabaseError(
            f"refusing to empty {name!r}: the tests empty only a database named *{TEST_SUFFIX}"
        )
    with engine.begin() as connection:
        tables = ", ".join(table.fullname for table in metadata.sorted_tables)
        connection.execute(text(f"TRUNCATE {tables}"))
