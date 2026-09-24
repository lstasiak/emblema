"""The modules Celery is pointed at: what each registers, and what a delivered job reaches.

Imported inside a fixture rather than at the top, with every setting it reads named in the
environment, because a module builds its application as it is imported — which is what
``celery -A`` needs, and what makes a process without a broker fail where it is started rather
than at the first job. Both processes are held to the same facts, because a worker that registered
its job under another name would sit idle while its cells waited.
"""

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType

import pytest

from emblema.evaluation.application.use_cases.advance_campaign import RUN_CAMPAIGN_CELL
from emblema.shared.adapters.queues.celery_job_queue import CeleryJobQueue
from emblema.shared.jobs.job_argument import JobArgument
from emblema.shared.ports.job_queue import JobQueue

PROCESSES = {
    "ml": "emblema.entrypoints.workers.ml.celery_app",
    "general": "emblema.entrypoints.workers.general.celery_app",
}
JOB = {"campaign": "c", "candidate": "lora", "budget": "200", "seed": 1}
ENVIRONMENT = {
    "EMBLEMA_ARTIFACT_STORE__ENDPOINT_URL": "http://127.0.0.1:3900",
    "EMBLEMA_ARTIFACT_STORE__REGION": "garage",
    "EMBLEMA_ARTIFACT_STORE__BUCKET": "emblema",
    "EMBLEMA_ARTIFACT_STORE__KEY_PREFIX": "test",
    "EMBLEMA_BROKER__HOST": "127.0.0.1",
    "EMBLEMA_BROKER__PORT": "5672",
    "EMBLEMA_BROKER__USER": "nobody",
    "EMBLEMA_BROKER__PASSWORD": "unused",
    "EMBLEMA_BROKER__VHOST": "/",
}


class RecordingWorker:
    """A worker that files what it was handed instead of running a grid's cell."""

    def __init__(self) -> None:
        self.received: list[Mapping[str, JobArgument]] = []

    def run_campaign_cell(self, arguments: Mapping[str, JobArgument]) -> None:
        self.received.append(dict(arguments))


@dataclass(frozen=True)
class StubRoot:
    """A composition that supplies the one use case a worker is built on."""

    services: object


@pytest.fixture(params=list(PROCESSES), ids=list(PROCESSES))
def process(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """A module as a freshly started process imports it, on a broker nothing is sent to."""
    name = PROCESSES[request.param]
    for variable, value in ENVIRONMENT.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.delitem(sys.modules, name, raising=False)
    imported = __import__(name, fromlist=["app"])
    monkeypatch.delitem(sys.modules, name, raising=False)
    return imported


def test_the_module_registers_the_job_a_campaign_submits_under_the_name_it_submits_it_by(
    process: ModuleType,
) -> None:
    assert RUN_CAMPAIGN_CELL in process.app.tasks


def test_a_delivered_job_reaches_the_worker_this_process_holds(
    process: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = RecordingWorker()
    monkeypatch.setattr(process, "_worker", recording)

    process.run_campaign_cell(**JOB)

    assert recording.received == [JOB]


def test_the_worker_submits_through_the_application_this_process_listens_on(
    process: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A second application built beside this one would become Celery's current one and carry
    # none of the configuration this one was given.
    handed: list[JobQueue | None] = []

    def from_environment(jobs: JobQueue | None = None) -> StubRoot:
        handed.append(jobs)
        return StubRoot(services=RecordingWorker())

    monkeypatch.setattr(process, "_worker", None)
    monkeypatch.setattr(process.CompositionRoot, "from_environment", from_environment)

    process.worker()

    assert isinstance(handed[0], CeleryJobQueue)
