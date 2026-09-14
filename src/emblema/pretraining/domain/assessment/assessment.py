from dataclasses import dataclass

from emblema.pretraining.domain.assessment.check import Check, Rule, Status
from emblema.pretraining.domain.assessment.decision import Decision, decide
from emblema.pretraining.domain.assessment.kind_summary import BEYOND_LINEAR, KindSummary
from emblema.pretraining.domain.assessment.results import Results
from emblema.pretraining.domain.assessment.rules import (
    above_floor,
    beats_mean,
    beyond_linear,
    collapsed,
    converged,
    evidence,
    generalisation_gap,
    hidden_share,
    kinds_present,
    loss_falls,
    room,
    spectrum_fitted,
    spectrum_informative,
    triviality,
    validation_stable,
)
from emblema.pretraining.domain.assessment.summarised_run import SummarisedRun
from emblema.pretraining.domain.exceptions import IncomparableRunsError


@dataclass(frozen=True)
class Assessment:
    """A run, its summaries and checks, and the decision they support — computed once.

    Attributes:
        results: The run assessed.
        summaries: Its kinds of mask with their intervals.
        shorter: The run of the same configuration with at most half the epochs, if one exists.
        checks: Every rule applied.
        decision: The outcome.
    """

    results: Results
    summaries: tuple[KindSummary, ...]
    shorter: Results | None
    checks: tuple[Check, ...]
    decision: Decision

    @property
    def triviality_negative(self) -> bool:
        """Whether every kind the strategy draws was seen and beat the baselines that decide."""
        return not any(
            check.status is Status.FAIL
            and check.rule in (Rule.TRIVIAL, Rule.BEYOND_LINEAR, Rule.KINDS_PRESENT)
            for check in self.checks
        )


def assess(run: SummarisedRun, shorter: SummarisedRun | None = None) -> Assessment:
    """Every rule applied to ``run``, the convergence judged against ``shorter``.

    Raises:
        IncomparableRunsError: If ``shorter`` is not a run of the same configuration with at most
            half the epochs — a comparison with anything else would pass or fail for the wrong
            reason.
    """
    results = run.results
    if shorter is not None and not results.configuration_of(shorter.results):
        raise IncomparableRunsError("the shorter run must be a run of the same configuration")
    if shorter is not None and shorter.results.epochs > results.epochs // 2:
        raise IncomparableRunsError(
            f"the shorter run must have at most half the epochs, got {shorter.results.epochs} "
            f"against {results.epochs}"
        )
    summaries = run.summaries
    judged = [summary for summary in summaries if not summary.apart]
    checks = [
        loss_falls(results.curve),
        hidden_share(results),
        kinds_present(results.strategy, judged),
        collapsed(judged),
    ]
    checks += [above_floor(summary) for summary in judged]
    checks += [
        converged(results, judged, shorter),
        validation_stable(results.curve),
        generalisation_gap(results.curve),
    ]
    for summary in judged:
        checks += [triviality(summary), room(summary), beats_mean(summary)]
        if summary.kind in BEYOND_LINEAR:
            checks.append(beyond_linear(summary))
        checks.append(evidence(summary))
    checks += [spectrum_informative(results.spectrum), spectrum_fitted(results.spectrum)]
    return Assessment(
        results,
        summaries,
        None if shorter is None else shorter.results,
        tuple(checks),
        decide(checks),
    )
