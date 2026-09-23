from collections.abc import Mapping
from dataclasses import dataclass

from emblema.shared.jobs.job_argument import JobArgument
from emblema.shared.jobs.worker_pool import WorkerPool


@dataclass(frozen=True, kw_only=True)
class QueuedJob:
    """One piece of work handed to a queue, described so a worker can reconstruct it.

    The arguments are scalars, not objects: what crosses this port is serialised by a broker and
    read back in another process, possibly on another machine, so anything that cannot survive
    that trip has no business here. A job therefore names what to work on — identifiers,
    coordinates — and the worker builds the rest from its own composition root.

    Attributes:
        name: What the worker knows this kind of work as.
        pool: Which pool of workers is meant to pick it up.
        arguments: What to work on, by name.
    """

    name: str
    pool: WorkerPool
    arguments: Mapping[str, JobArgument]
