from typing import Protocol

from emblema.shared.jobs.queued_job import QueuedJob


class JobQueue(Protocol):
    """Hands work to whatever runs it after this call returns.

    Submitting says only that the job was accepted, never that it ran or succeeded: a caller
    that needed the outcome would not be scheduling. What a worker learns about the job is what
    the job carries and nothing else, so a job names what to work on and the worker finds the
    rest for itself.
    """

    def submit(self, job: QueuedJob) -> None:
        """Hand the job over.

        Raises:
            JobQueueError: If the queue cannot be reached.
        """
        ...
