from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from emblema.pretraining.domain.assessment.check import Area, Check, Rule, Status


class Outcome(Enum):
    ACCEPT = "accept"
    RECALIBRATE = "recalibrate"
    CHANGE_MASKING = "change-masking"
    CHANGE_LOGIC = "change-logic"


@dataclass(frozen=True)
class Decision:
    """What the checks add up to: one outcome for the run and what to do about it.

    Attributes:
        outcome: The verdict on the run.
        headline: The verdict in a sentence.
        actions: The readings of every check that did not pass, failures first, then by area.
    """

    outcome: Outcome
    headline: str
    actions: tuple[str, ...]


def decide(checks: Sequence[Check]) -> Decision:
    """The outcome the checks support, by the precedence of what a failure invalidates.

    A fault in the implementation invalidates every other number, so it comes first. A kind of
    mask whose baseline sits at the noise floor is trivial whatever the training, so it comes next.
    A kind the model has not beaten, in a run not shown to have stopped learning, is a question for
    a longer run, not for the strategy. Only a run shown to have stopped learning that still loses
    to a baseline condemns the kind of mask.
    """

    def failing(rule: Rule) -> list[Check]:
        return [check for check in checks if check.rule is rule and check.status is Status.FAIL]

    def names(found: Iterable[Check]) -> str:
        return ", ".join(check.name for check in found)

    severity = {Status.FAIL: 0, Status.WARN: 1}
    areas = list(Area)
    pending = sorted(
        (check for check in checks if check.status in severity),
        key=lambda check: (severity[check.status], areas.index(check.area)),
    )
    actions = tuple(f"[{check.area.value}] {check.name}: {check.reading}" for check in pending)
    converged = any(
        check.rule is Rule.CONVERGED and check.status is Status.PASS for check in checks
    )
    broken = [
        check
        for check in checks
        if check.area is Area.IMPLEMENTATION and check.status is Status.FAIL
    ]
    if broken:
        return Decision(
            Outcome.CHANGE_LOGIC,
            f"Change the logic: {names(broken)} failed, which a correct pipeline cannot do; no "
            "other number of this run can be trusted until it passes.",
            actions,
        )
    if cramped := failing(Rule.ROOM):
        return Decision(
            Outcome.CHANGE_MASKING,
            f"Change the masking strategy: {names(cramped)} — the baseline already sits at the "
            "noise floor, so this kind of mask has nothing to teach on this corpus however long "
            "the model trains.",
            actions,
        )
    if trivial := failing(Rule.TRIVIAL) + failing(Rule.BEYOND_LINEAR):
        if not converged:
            return Decision(
                Outcome.RECALIBRATE,
                f"Recalibrate the training before reading {names(trivial)}: the run is not shown "
                "to have stopped learning, so a longer one may still beat the baseline and the "
                "verdict is provisional.",
                actions,
            )
        return Decision(
            Outcome.CHANGE_MASKING,
            f"Change the masking strategy: {names(trivial)} — a model that doubling the budget "
            "no longer improves does not beat the trivial or the linear baseline, so this kind of "
            "mask teaches nothing transferable.",
            actions,
        )
    headline = "Accept: every kind of mask beats its baseline and the implementation checks pass."
    if not converged:
        headline += " Whether a longer run would widen the margins is not shown."
    return Decision(Outcome.ACCEPT, headline, actions)
