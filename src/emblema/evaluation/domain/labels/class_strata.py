from collections.abc import Sequence
from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import InvalidOutcomeError, SingleClassSampleError
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.outcome_scheme import OutcomeScheme
from emblema.shared.kernel.ordering import seeded_rank


@dataclass(frozen=True)
class ClassStrata:
    """The strata a budget of a binary task is drawn across: its two outcomes, in proportion.

    A budget drawn without regard to the outcome varies in how many positives it holds, and a
    candidate is then measured on its draw as much as on its method. An equal share of each
    outcome would change the question instead: a candidate taught at even odds answers the
    probabilities of a population that does not exist. So every prefix of the draw holds each
    outcome within one window of its share of the pool.

    The draw takes one window at a time from whichever outcome lies furthest below its share so
    far, in integers so no rounding decides a step; members of an outcome are taken in the order
    their keys rank under the seed, and a tie between the outcomes is broken by the seed as well.
    A smaller budget is a prefix of a larger one under the same seed, so the budgets of a curve
    are nested.
    """

    def draw(self, pool: Sequence[LabelledWindow], wanted: int, seed: int) -> tuple[int, ...]:
        """Positions of ``wanted`` windows in ``pool``, each outcome in proportion to its share.

        Raises:
            InvalidOutcomeError: If a window's target is neither zero nor one.
            SingleClassSampleError: If the drawn windows hold one outcome only, which a budget too
                small for the rarer outcome's share or a pool of one outcome leaves: a candidate
                taught from it cannot learn what tells the outcomes apart.
        """
        outcomes = OutcomeScheme.OUTCOMES
        members: dict[float, list[int]] = {outcome: [] for outcome in outcomes}
        for index, labelled in enumerate(pool):
            if labelled.target not in members:
                raise InvalidOutcomeError(
                    f"a window of {labelled.window.unit} is labelled {labelled.target}, "
                    "not an outcome"
                )
            members[labelled.target].append(index)
        for outcome in outcomes:
            members[outcome].sort(
                key=lambda index: seeded_rank(
                    seed, pool[index].window.unit, pool[index].window.position
                )
            )
        tie_order = {outcome: seeded_rank(seed, "outcome", outcome) for outcome in outcomes}
        taken: list[int] = []
        drawn = dict.fromkeys(outcomes, 0)
        for step in range(1, wanted + 1):
            outcome = max(
                (outcome for outcome in outcomes if drawn[outcome] < len(members[outcome])),
                key=lambda outcome: (
                    len(members[outcome]) * step - drawn[outcome] * len(pool),
                    tie_order[outcome],
                ),
            )
            taken.append(members[outcome][drawn[outcome]])
            drawn[outcome] += 1
        if min(drawn.values()) == 0:
            raise SingleClassSampleError(
                f"{wanted} windows drawn from {len(members[1.0])} positive and "
                f"{len(members[0.0])} negative hold one outcome only"
            )
        return tuple(taken)
