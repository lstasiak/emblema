from dataclasses import dataclass

from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.task.downstream_task import DownstreamTask


@dataclass(frozen=True, kw_only=True)
class FittingSource:
    """Another task's labels, offered to a fit so a candidate learns from more than the target.

    The task travels beside its sample because the windows are read out of the corpus the task
    was defined against, and a sample names only the task it came from. Pairing them here means
    a fit cannot be handed labels of one corpus and told to read them out of another.

    Invariants: the sample was drawn from this task.

    Attributes:
        task: Task the labels belong to, and whose corpus their windows are read from.
        sample: The labelled windows this task contributes.
    """

    task: DownstreamTask
    sample: LabelSample

    def __post_init__(self) -> None:
        self.task.accept_sample(self.sample)
