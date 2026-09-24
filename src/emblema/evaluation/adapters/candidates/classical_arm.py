from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CandidateRef, TaskId
from emblema.evaluation.domain.classical.classical_method import ClassicalMethod


@dataclass(frozen=True, kw_only=True)
class ClassicalArm:
    """One classical method a campaign competes, under a name of its own.

    What separates one of these from another is what a window is read as and what is fitted on
    that reading, and which other tasks it is allowed to learn from. Arms of one kind share how
    hard they fit, the way the neural arms share one schedule: two baselines that differed in
    their trees as well as in their features would leave a reader unable to say which of the two
    differences a result came from.

    Attributes:
        ref: What the campaign calls this baseline.
        method: What a window is read as, and what is fitted on that reading.
        sources: Other tasks whose labels are fitted alongside the target's; empty for a
            baseline that learns the target alone.
    """

    ref: CandidateRef
    method: ClassicalMethod
    sources: tuple[TaskId, ...]
