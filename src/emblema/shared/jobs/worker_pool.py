from enum import StrEnum


class WorkerPool(StrEnum):
    """Which pool of workers a job is meant for.

    Not about priority and not about hardware: every pool runs wherever it is deployed, and on
    the target platform that is a container with no accelerator in it. What separates them is
    what their image carries. An image with the machine-learning stack in it is large, takes one
    job at a time because a process that has imported that stack cannot safely fork, and holds
    each one for as long as a network takes to learn. An image without it is small, and its jobs
    are counted in seconds.

    Attributes:
        GENERAL: Work that needs nothing of the machine-learning stack, which is most of it.
        ML: Work that trains or scores a network.
    """

    GENERAL = "general"
    ML = "ml"
