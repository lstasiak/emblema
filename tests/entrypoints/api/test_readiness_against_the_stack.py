"""The readiness probe reaches the database and the bucket the settings name.

Against the local stack, so marked ``integration``: what it holds is that the two real checks
the root composes make a round trip each, and report it, rather than that a check can fail.
"""

import pytest

from emblema.config.settings import Settings
from emblema.entrypoints.api.composition_root import CompositionRoot
from tests.support.database import create_test_database

pytestmark = pytest.mark.integration


def test_the_process_is_ready_once_the_stack_answers() -> None:
    create_test_database()

    report = CompositionRoot(Settings()).readiness.report()

    assert report.ready, report.checks
    assert report.checks == {"artifact_store": "ok", "database": "ok"}
