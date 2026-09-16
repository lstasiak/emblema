from dataclasses import dataclass

from emblema.shared.kernel.identifiers import EntityId


@dataclass(frozen=True)
class TaskId(EntityId):
    """Identity of a downstream task, the Evaluation entity other contexts refer to.

    It lives in the published language because a campaign's outcome names the task it was run
    for, and whoever reads that outcome must be able to say which task it was without reaching
    into this context.
    """
