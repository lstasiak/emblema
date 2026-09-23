from enum import StrEnum


class WorkerPool(StrEnum):
    """Which pool of workers a job is meant for.

    The split is not about priority but about where the work can physically run: one pool lives
    in a container and one runs on the host beside the accelerator, because the accelerator this
    project trains on is not visible from inside a container.

    Attributes:
        CPU: Work that needs no accelerator — classical candidates, statistics, export. Run by
            containerised workers.
        ML: Work that trains or scores a network. Run by a worker started on the host, one job
            at a time, because a process that has imported the ML stack cannot safely fork.
    """

    CPU = "cpu"
    ML = "ml"
