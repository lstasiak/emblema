from uuid import UUID

import pytest

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import (
    ForeignLabelSampleError,
    FrozenTestSplitClosedError,
    UnknownGroundTruthError,
    UnlabelledWindowError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from tests.evaluation.support import FORECAST, TASK, TEST_SIDE, labelled, task, units, window


def test_a_task_labels_windows_by_its_scheme_from_the_failure_of_their_unit() -> None:
    windows = [window("a", 0, 100.0), window("a", 1, 250.0), window("b", 2, 40.0)]

    failed = {UnitKey("a"): 300.0, UnitKey("b"): 50.0}

    read = task().labelled(windows, {w: failed[w.unit] for w in windows})

    assert [(w.window.position, w.target) for w in read] == [(0, 125.0), (1, 50.0), (2, 10.0)]


def test_a_window_past_the_failure_of_its_unit_cannot_be_labelled() -> None:
    with pytest.raises(UnlabelledWindowError):
        task().labelled([window("a", 0, 301.0)], {window("a", 0, 301.0): 300.0})


def test_a_window_the_ground_truth_says_nothing_about_cannot_be_labelled() -> None:
    with pytest.raises(UnknownGroundTruthError, match="unit b at 1"):
        task().labelled(
            [window("a", 0, 100.0), window("b", 1, 40.0)], {window("a", 0, 100.0): 300.0}
        )


def test_a_forecasting_task_labels_a_window_with_the_exact_reading_it_is_given() -> None:
    windows = [window("a", 0, 100.0), window("b", 1, 40.0)]

    read = task(labels=FORECAST).labelled(windows, {windows[0]: -0.25, windows[1]: 1.5})

    assert [w.target for w in read] == [-0.25, 1.5]


def test_a_sample_drawn_from_the_task_is_accepted() -> None:
    sample = LabelSample(
        task=TASK, windows=(labelled("a", 0, 10.0, 5.0),), budget=LabelBudget.of(1), seed=1
    )

    task().accept_sample(sample)


def test_a_sample_drawn_from_another_task_is_refused() -> None:
    sample = LabelSample(
        task=TaskId(UUID(int=9)),
        windows=(labelled("a", 0, 10.0, 5.0),),
        budget=LabelBudget.of(1),
        seed=1,
    )

    with pytest.raises(ForeignLabelSampleError, match="drawn from task"):
        task().accept_sample(sample)


def test_a_task_names_the_units_a_budget_may_be_drawn_from() -> None:
    assert task().tuning_units == units("a", "b")
    assert task().validation_units == units("c")


def test_a_tuning_run_is_refused_the_frozen_side() -> None:
    with pytest.raises(FrozenTestSplitClosedError, match="frozen for a tuning run"):
        task().open_test_split(RunPurpose.TUNING)


def test_the_final_run_is_handed_the_frozen_side() -> None:
    assert task().open_test_split(RunPurpose.FINAL) == TEST_SIDE
