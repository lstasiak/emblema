from celery import Celery
from kombu.exceptions import OperationalError

from emblema.shared.jobs.queued_job import QueuedJob
from emblema.shared.ports.exceptions import JobQueueError


class CeleryJobQueue:
    """Hands jobs to Celery, routed to the queue the job names.

    Tasks are sent by name rather than by calling an imported function, so a process that only
    enqueues never imports what runs: the API and the command line stay free of the training
    stack, and the worker is the one place that knows how a job is carried out. The broker is
    reached on submission, so an unreachable one is reported here instead of silently dropping
    work.
    """

    def __init__(self, app: Celery) -> None:
        self._app = app

    def submit(self, job: QueuedJob) -> None:
        """Hand the job to its queue.

        Raises:
            JobQueueError: If the broker cannot be reached; nothing was accepted.
        """
        try:
            self._app.send_task(job.name, kwargs=dict(job.arguments), queue=str(job.pool))
        except OperationalError as error:
            raise JobQueueError(f"the queue did not accept {job.name!r}: {error}") from error
