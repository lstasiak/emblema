from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CandidateRef, TaskId
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme


@dataclass(frozen=True, kw_only=True)
class ClassicalArm:
    """One classical method a campaign competes, under a name of its own.

    What separates one of these from another is what a window is turned into before any tree is
    grown, and which other tasks it is allowed to learn from. How hard it then fits is shared,
    the way the neural arms share one schedule: two baselines that differed in their trees as
    well as in their features would leave a reader unable to say which of the two differences a
    result came from.

    Attributes:
        ref: What the campaign calls this baseline.
        features: What a window is turned into before anything is fitted.
        sources: Other tasks whose labels are fitted alongside the target's; empty for a
            baseline that learns the target alone.
    """

    ref: CandidateRef
    features: FeatureScheme
    sources: tuple[TaskId, ...]
