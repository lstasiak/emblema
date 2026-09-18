from collections.abc import Sequence
from typing import Protocol

from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan


class AdaptationRuntime(Protocol):
    """Makes a candidate out of the backbone under a plan, teaches it a sample, scores it.

    Where the arithmetic happens and what the backbone is made of are the adapter's business:
    a torch module on whatever device the machine has, or nothing learnt at all. What the port
    fixes is that a run is described by a plan, a task and a sample of that task's labels, that
    it is scored on the validation windows it is handed and on nothing else, and that it reports
    an answer per window rather than a summary. Weights never cross this port: a plan names the
    artifact the pretrained ones are in, and the outcome carries none.
    """

    def adapt(
        self,
        plan: AdaptationPlan,
        task: DownstreamTask,
        sample: LabelSample,
        validation: Sequence[LabelledWindow],
    ) -> AdaptationOutcome:
        """Learn the task from ``sample`` under ``plan`` and answer every window of ``validation``.

        The outcome predicts the validation windows in the order given, one each.

        Raises:
            ForeignLabelSampleError: If the sample was drawn from another task.
            UnknownBackboneError: If the plan names pretrained weights the runtime cannot supply.
            LoraTargetNotFoundError: If the plan's low-rank updates name a layer the backbone
                does not have.
            InvalidAdaptationOutcomeError: If there is no validation window to answer.
        """
        ...
