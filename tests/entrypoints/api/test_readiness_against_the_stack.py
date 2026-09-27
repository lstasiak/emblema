"""The readiness probe reaches the database and the bucket the settings name.

Against the local stack, so marked ``integration``: what it holds is that the two real checks
the root composes make a round trip each, and report it, rather than that a check can fail. The
API group is given, not read: the probe is about the stack, and a process that reads only the
stack's groups is what the suite runs in.
"""

import pytest

from emblema.config.settings import Settings
from emblema.entrypoints.api.composition_root import CompositionRoot
from tests.support.database import create_test_database
from tests.support.settings import API

pytestmark = pytest.mark.integration


def test_the_process_is_ready_once_the_stack_answers() -> None:
    create_test_database()

    report = CompositionRoot(Settings(api=API)).readiness.report()

    assert report.ready, report.checks
    assert report.checks == {"artifact_store": "ok", "database": "ok"}
