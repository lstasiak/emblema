"""Contract of the JobQueue port, run against every adapter.

The Celery adapter is an integration test of its own below: what it promises that the immediate
one cannot is that an unreachable broker is reported rather than swallowed, and that needs a
broker to be unreachable against.
"""

from collections.abc import Mapping

import pytest
from celery import Celery

from emblema.shared.adapters.queues.celery_application import celery_application
from emblema.shared.adapters.queues.celery_job_queue import CeleryJobQueue
from emblema.shared.adapters.queues.immediate_job_queue import ImmediateJobQueue
from emblema.shared.jobs.job_argument import JobArgument
from emblema.shared.jobs.queued_job import QueuedJob
from emblema.shared.jobs.worker_pool import WorkerPool
from emblema.shared.ports.exceptions import JobQueueError

WORK = "tests.work"
JOB = QueuedJob(name=WORK, pool=WorkerPool.CPU, arguments={"what": "a thing", "how_many": 2})


def test_a_submitted_job_reaches_the_handler_with_its_arguments() -> None:
    received: list[Mapping[str, JobArgument]] = []

    ImmediateJobQueue({WORK: received.append}).submit(JOB)

    assert received == [JOB.arguments]


def test_a_job_this_process_runs_nothing_for_is_refused() -> None:
    with pytest.raises(JobQueueError, match="runs no job"):
        ImmediateJobQueue({}).submit(JOB)


def test_a_handler_that_fails_fails_the_submission() -> None:
    def raise_it(_: Mapping[str, JobArgument]) -> None:
        raise RuntimeError("the work failed")

    with pytest.raises(RuntimeError, match="the work failed"):
        ImmediateJobQueue({WORK: raise_it}).submit(JOB)


def test_the_application_holds_a_delivered_job_until_the_work_is_answered_for() -> None:
    # Acknowledged after the work, not on delivery, so a cell whose worker died is offered
    # again; on AMQP the broker holds it for as long as the worker does, with no timeout to set.
    assert celery_application("memory://").conf.task_acks_late is True


def test_a_submission_that_returns_is_one_the_broker_took_responsibility_for() -> None:
    options = celery_application("memory://").conf.broker_transport_options

    assert options["confirm_publish"] is True


def test_the_application_sends_to_the_pool_beside_the_accelerator_unless_a_job_says_otherwise() -> (
    None
):
    assert celery_application("memory://").conf.task_default_queue == str(WorkerPool.ML)


def test_an_unreachable_broker_is_reported_rather_than_swallowed() -> None:
    # A port nothing listens on, so the connection is refused rather than left hanging.
    unreachable = Celery(broker="amqp://guest:guest@127.0.0.1:1//")
    unreachable.conf.broker_transport_options = {"max_retries": 0}
    unreachable.conf.broker_connection_retry_on_startup = False

    with pytest.raises(JobQueueError, match="did not accept"):
        CeleryJobQueue(unreachable).submit(JOB)
