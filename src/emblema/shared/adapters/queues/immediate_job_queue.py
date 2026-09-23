from collections.abc import Callable, Mapping

from emblema.shared.jobs.job_argument import JobArgument
from emblema.shared.jobs.queued_job import QueuedJob
from emblema.shared.ports.exceptions import JobQueueError

type JobHandler = Callable[[Mapping[str, JobArgument]], None]
"""What a process does with one job's arguments."""


class ImmediateJobQueue:
    """Runs each job before the call returns, in the process that submitted it.

    A scheduler for a process with no broker: a report, a test, a machine running a campaign
    from end to end on its own. It keeps the promise the port makes while removing everything
    that makes a queue hard to reason about, so a campaign's whole cycle can be exercised as one
    call.

    What it does not remove is the queue's semantics, and that is deliberate: a job that fails
    here fails the submission, where a broker would have retried it. A caller that depends on
    the difference is a caller that depends on the adapter.
    """

    def __init__(self, handlers: Mapping[str, JobHandler]) -> None:
        """Run jobs with ``handlers``, one per job name this process knows."""
        self._handlers = dict(handlers)

    def submit(self, job: QueuedJob) -> None:
        """Run the job now.

        Raises:
            JobQueueError: If this process knows no handler of that name.
        """
        handler = self._handlers.get(job.name)
        if handler is None:
            raise JobQueueError(f"this process runs no job called {job.name!r}")
        handler(job.arguments)
