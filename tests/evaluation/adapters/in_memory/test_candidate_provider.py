from emblema.evaluation.adapters.in_memory.candidate_provider import (
    InMemoryCandidateProvider,
    StatedAnswers,
    StatedErrors,
)
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from tests.evaluation.support import CONTROL, TASK, candidate, cell, units

STAYS = tuple(sorted(units("a", "b", "c"), key=str))
CELL = cell(CONTROL, LabelBudget.of(50), 1)


def evaluated(provider: InMemoryCandidateProvider) -> CandidateEvaluation:
    return CandidateEvaluation(
        task=TASK,
        cell=CELL,
        purpose=RunPurpose.TUNING,
        retain=False,
        declared=candidate(CONTROL),
    )


def test_stated_errors_score_each_unit_over_one_window_and_keep_no_answer() -> None:
    provider = InMemoryCandidateProvider(
        (candidate(CONTROL),), STAYS, StatedErrors(lambda _: (3.0, 4.0, 5.0))
    )

    result = provider.evaluate(evaluated(provider))

    assert [(str(e.unit), e.squared_error, e.windows) for e in result.errors] == [
        ("a", 9.0, 1),
        ("b", 16.0, 1),
        ("c", 25.0, 1),
    ]
    assert result.predictions == ()


def test_stated_answers_are_kept_and_their_squared_errors_are_the_units_errors() -> None:
    provider = InMemoryCandidateProvider(
        (candidate(CONTROL),), STAYS, StatedAnswers(lambda _: ((1.0, 0.9), (0.0, 0.2), (1.0, 0.5)))
    )

    result = provider.evaluate(evaluated(provider))

    assert [(p.target, p.predicted) for p in result.predictions] == [
        (1.0, 0.9),
        (0.0, 0.2),
        (1.0, 0.5),
    ]
    assert [round(e.squared_error, 6) for e in result.errors] == [0.01, 0.04, 0.25]
