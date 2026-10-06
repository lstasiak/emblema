"""A cell's result written for another machine and read back, answers included."""

from emblema.evaluation.adapters.documents.cell_result_document import CellResultDocument
from emblema.evaluation.adapters.documents.downstream_task_document import DownstreamTaskDocument
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.task.task_windows import TaskWindows
from tests.evaluation.support import (
    CONTENDER,
    OUTCOME,
    OUTCOMES,
    answered,
    result,
    task,
)

DOCUMENTS = CellResultDocument()


def test_a_result_comes_back_with_every_answer_it_gave() -> None:
    stated = answered(CONTENDER, LabelBudget.of(50), 1, (0.9, 0.2, 0.7, 0.4, 0.1, 0.3))

    assert DOCUMENTS.decode(DOCUMENTS.encode(stated)) == stated


def test_a_result_written_before_answers_were_kept_reads_back_without_them() -> None:
    written = DOCUMENTS.encode(result(CONTENDER, LabelBudget.of(50), 1, (3.0, 4.0)))
    del written["predictions"]

    read = DOCUMENTS.decode(written)

    assert read.predictions == ()
    assert read.errors == result(CONTENDER, LabelBudget.of(50), 1, (3.0, 4.0)).errors


def test_a_task_over_outcomes_travels_with_its_outcome_and_comes_back_spread_over_both() -> None:
    documents = DownstreamTaskDocument()
    stated = task(labels=OUTCOME, strata=OUTCOMES)

    written = documents.encode(stated)

    assert written["labels"] == {"scheme": "outcome", "outcome": "In-hospital_death"}
    assert documents.decode(written) == stated


def test_a_task_reading_every_window_is_written_as_it_was_before_the_choice_existed() -> None:
    written = DownstreamTaskDocument().encode(task(labels=OUTCOME, strata=OUTCOMES))

    assert "windows" not in written


def test_a_task_reading_the_first_window_of_each_unit_travels_with_the_choice() -> None:
    documents = DownstreamTaskDocument()
    stated = task(labels=OUTCOME, strata=OUTCOMES, windows=TaskWindows.FIRST)

    written = documents.encode(stated)

    assert written["windows"] == "first"
    assert documents.decode(written) == stated
