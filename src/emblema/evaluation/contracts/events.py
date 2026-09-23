"""What the Evaluation context publishes about its own work, for the contexts that act on it."""

from dataclasses import dataclass

from emblema.evaluation.contracts.evaluated_candidate import EvaluatedCandidate
from emblema.evaluation.contracts.identifiers import CampaignId, TaskId
from emblema.shared.events.domain_event import DomainEvent


@dataclass(frozen=True, kw_only=True)
class FrozenTestSplitOpened(DomainEvent):
    """The frozen side of a task was handed to a run.

    This is published every time it happens, not only the first, so that the record shows whether
    the promise of a single final run was kept. A project that cannot say how often it read its
    own test set is not in a position to claim it read it once.

    Attributes:
        task: Task whose frozen side was opened.
        unit_count: How many units were handed over.
        source: Where those units come from, as the set is known outside this system.
    """

    task: TaskId
    unit_count: int
    source: str


@dataclass(frozen=True, kw_only=True)
class CampaignCompleted(DomainEvent):
    """A campaign ran every cell of its grid and reached a verdict.

    This is the whole of what leaves this context about a comparison, and it is deliberately
    small: who competed, what each is made of, what each scored, which artifact was kept, and
    the verdict as a sentence a person reads. It carries no cell of the grid, no interval and no
    p-value — a context that promoted on those would be re-deciding a question this one has
    already answered — and no reference into this context's own tables.

    Attributes:
        campaign: Identity of the campaign that finished.
        task: Task every candidate answered.
        verdict: What the campaign concluded, in one sentence.
        candidates: Every competitor, the control arm included.
    """

    campaign: CampaignId
    task: TaskId
    verdict: str
    candidates: tuple[EvaluatedCandidate, ...]
