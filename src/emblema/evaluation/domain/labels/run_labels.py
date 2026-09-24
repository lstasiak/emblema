from dataclasses import dataclass

from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.task.downstream_task import DownstreamTask


@dataclass(frozen=True, kw_only=True)
class RunLabels:
    """What one run is measured against: the task, the labels it learns, the windows it answers.

    What a candidate is made of decides nothing here. A backbone adapted under a transfer mode
    and a fit of trees that began empty are scored on the same windows, from the same budget,
    drawn under the same seed, or the comparison between them says as much about how each was
    given its labels as about the methods. Keeping the three together is what makes that a
    property of the value rather than a coincidence of two call sites.

    Attributes:
        task: Task as stored, carrying the manifest its windows are read from.
        sample: Labels the candidate may learn from, drawn from the tuning side alone.
        scored: Windows the candidate answers, labelled by the same scheme as the sample.
    """

    task: DownstreamTask
    sample: LabelSample
    scored: tuple[LabelledWindow, ...]
