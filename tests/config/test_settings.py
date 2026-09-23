import pytest

from emblema.config.settings import Settings
from tests.support.settings import ARTIFACT_STORE, BROKER, DATABASE

# Stated outright rather than read from the environment: what this machine's `.env` names must
# not decide whether a process without one of these is refused. Every optional group is pinned,
# not only the one a case is about — an unpinned one is read from the file and the case passes
# for the wrong reason.
TRAINING_ONLY = Settings(artifact_store=ARTIFACT_STORE, database=None, broker=None, worker=None)
WITH_REGISTRY = Settings(artifact_store=ARTIFACT_STORE, database=DATABASE)


def test_a_process_that_names_no_database_may_still_be_configured() -> None:
    assert TRAINING_ONLY.database is None


def test_a_process_that_needs_the_database_is_refused_without_one() -> None:
    with pytest.raises(ValueError, match="EMBLEMA_DATABASE__"):
        TRAINING_ONLY.require_database()


def test_the_database_named_is_the_one_required() -> None:
    assert WITH_REGISTRY.require_database() == DATABASE


def test_a_process_that_needs_the_queue_is_refused_without_one() -> None:
    with pytest.raises(ValueError, match="EMBLEMA_BROKER__"):
        TRAINING_ONLY.require_broker()


def test_the_broker_named_is_the_one_required() -> None:
    named = Settings(artifact_store=ARTIFACT_STORE, broker=BROKER)

    assert named.require_broker() == BROKER


def test_a_process_that_runs_campaign_cells_is_refused_without_a_worker() -> None:
    with pytest.raises(ValueError, match="EMBLEMA_WORKER__"):
        TRAINING_ONLY.require_worker()
