from dataclasses import replace
from uuid import UUID

import pytest

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.classical.fitting_source import FittingSource
from emblema.evaluation.domain.exceptions import ForeignLabelSampleError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from tests.evaluation.support import TASK, labelled, task

SAMPLE = LabelSample(
    task=TASK,
    windows=(labelled("a", 0, 10.0, 5.0), labelled("b", 3, 10.0, 9.0)),
    budget=LabelBudget.of(2),
    seed=7,
)


def test_a_source_pairs_a_task_with_the_labels_drawn_from_it() -> None:
    source = FittingSource(task=task(), sample=SAMPLE)

    assert source.sample.windows == SAMPLE.windows


def test_labels_of_one_task_may_not_be_offered_as_another_task_s() -> None:
    elsewhere = replace(SAMPLE, task=TaskId(UUID(int=9)))

    with pytest.raises(ForeignLabelSampleError):
        FittingSource(task=task(), sample=elsewhere)
