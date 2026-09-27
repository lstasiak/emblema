"""Every adapter answers the same way about windows it knows and refuses the ones it does not.

The turbofan sample in the repository holds engines 39 and 91 of the first subset, recorded for
128 and 135 cycles. The corpus times a unit from cycle one to one past its last, so those engines
fail at 129 and 136 — the numbers a window's end is compared against. The generated corpus is
asked about the first sensor of its first two units. The intensive-care sample records one death
in hospital, stay 132551 of set A, and survivals for the others.
"""

from collections.abc import Mapping
from pathlib import Path

import pytest

from emblema.evaluation.adapters.in_memory.ground_truth import InMemoryGroundTruth
from emblema.evaluation.adapters.readers.cmapss_ground_truth import CmapssGroundTruth
from emblema.evaluation.adapters.readers.corpus_ground_truths import CorpusGroundTruths
from emblema.evaluation.adapters.readers.physionet2012_ground_truth import (
    Physionet2012GroundTruth,
)
from emblema.evaluation.adapters.synthetic.synthetic_ground_truth import SyntheticGroundTruth
from emblema.evaluation.domain.exceptions import (
    UnknownGroundTruthError,
    UnreadableGroundTruthError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme
from emblema.evaluation.domain.labels.outcome_scheme import OutcomeScheme
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.ports.ground_truth import GroundTruth
from emblema.shared.adapters.synthetic.layouts import CONTROL_B
from emblema.shared.adapters.synthetic.sensor_signal import SensorSignal
from tests.evaluation.support import window
from tests.support.corpora import sample
from tests.support.synthetic import PROCESS, miniature

FIRST, SECOND = UnitKey("FD001/39"), UnitKey("FD001/91")
FAILURES = {FIRST: 129.0, SECOND: 136.0}
WINDOWS = (window("FD001/39", 0, 50.0), window("FD001/39", 1, 55.0), window("FD001/91", 2, 60.0))
LAYOUT = miniature(CONTROL_B)
SYNTHETIC_WINDOWS = (
    window(f"{LAYOUT.name}/0", 0, 40.0),
    window(f"{LAYOUT.name}/0", 1, 52.0),
    window(f"{LAYOUT.name}/1", 2, 40.0),
)
DEATH = OutcomeScheme("In-hospital_death")
STAYS = (
    window("set-a/132551", 0, 48.0),
    window("set-a/132539", 1, 48.0),
    window("set-b/149509", 2, 48.0),
)


class Case:
    """One adapter with the windows it knows and one it does not."""

    def __init__(
        self,
        truth: GroundTruth,
        known: tuple[TaskWindow, ...],
        unknown: TaskWindow,
        corpus: str = "cmapss",
    ) -> None:
        self.truth, self.known, self.unknown, self.corpus = truth, known, unknown, corpus

    def asked(self, windows: tuple[TaskWindow, ...]) -> Mapping[TaskWindow, float]:
        return self.truth.truths_of(self.corpus, windows)


@pytest.fixture(params=["in memory", "C-MAPSS", "synthetic", "by corpus", "PhysioNet"])
def case(request: pytest.FixtureRequest) -> Case:
    if request.param == "in memory":
        return Case(InMemoryGroundTruth(FAILURES), WINDOWS, window("FD001/999", 3, 10.0))
    if request.param == "C-MAPSS":
        return Case(CmapssGroundTruth(sample("cmapss")), WINDOWS, window("FD001/999", 3, 10.0))
    if request.param == "PhysioNet":
        stays = Physionet2012GroundTruth(sample("physionet2012"), DEATH)
        return Case(stays, STAYS, window("set-a/999999", 3, 48.0), "physionet2012")
    if request.param == "by corpus":
        registered = CorpusGroundTruths({"cmapss": CmapssGroundTruth(sample("cmapss"))})
        return Case(registered, WINDOWS, window("FD001/999", 3, 10.0))
    truth = SyntheticGroundTruth(SensorSignal(PROCESS, LAYOUT), ForecastScheme("s01", 12.0))
    return Case(
        truth, SYNTHETIC_WINDOWS, window(f"{LAYOUT.name}/{LAYOUT.units}", 3, 10.0), LAYOUT.name
    )


def test_every_window_asked_about_is_answered(case: Case) -> None:
    truths = case.asked(case.known)

    assert set(truths) == set(case.known)
    assert all(isinstance(value, float) for value in truths.values())


def test_asking_about_one_window_answers_about_that_window(case: Case) -> None:
    whole, one = case.asked(case.known), case.asked(case.known[1:2])

    assert one == {case.known[1]: whole[case.known[1]]}


def test_a_window_the_ground_truth_says_nothing_about_is_refused(case: Case) -> None:
    with pytest.raises(UnknownGroundTruthError):
        case.asked((*case.known, case.unknown))


def test_asking_about_nothing_answers_nothing(case: Case) -> None:
    assert case.asked(()) == {}


@pytest.mark.parametrize(
    "truth", [InMemoryGroundTruth(FAILURES), CmapssGroundTruth(sample("cmapss"))]
)
def test_every_window_of_an_engine_is_answered_with_its_failure(truth: GroundTruth) -> None:
    first, again, other = WINDOWS

    assert truth.truths_of("cmapss", WINDOWS) == {first: 129.0, again: 129.0, other: 136.0}


def test_a_truth_stated_per_window_is_answered_before_its_unit() -> None:
    truth = InMemoryGroundTruth(FAILURES, {WINDOWS[1]: 1.5})

    assert truth.truths_of("cmapss", WINDOWS[:2]) == {WINDOWS[0]: 129.0, WINDOWS[1]: 1.5}


@pytest.mark.parametrize("name", ["39", "FD001/engine-39", "FD001/"])
def test_a_unit_not_named_by_subset_and_engine_is_refused(name: str) -> None:
    with pytest.raises(UnknownGroundTruthError, match="named"):
        CmapssGroundTruth(sample("cmapss")).truths_of("cmapss", [window(name, 0, 1.0)])


def test_a_subset_whose_file_is_not_there_says_so(tmp_path: Path) -> None:
    with pytest.raises(UnreadableGroundTruthError, match="cannot read"):
        CmapssGroundTruth(tmp_path).truths_of("cmapss", [window("FD001/1", 0, 1.0)])


def test_a_ground_truth_file_that_is_not_one_is_refused(tmp_path: Path) -> None:
    (tmp_path / "train_FD001.txt").write_text("engine cycle setting\n", encoding="utf-8")

    with pytest.raises(UnreadableGroundTruthError, match="engine number"):
        CmapssGroundTruth(tmp_path).truths_of("cmapss", [window("FD001/1", 0, 1.0)])


def test_an_empty_ground_truth_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "train_FD001.txt").write_text("\n   \n", encoding="utf-8")

    with pytest.raises(UnreadableGroundTruthError, match="no engine"):
        CmapssGroundTruth(tmp_path).truths_of("cmapss", [window("FD001/1", 0, 1.0)])


def test_blank_lines_between_records_are_not_an_engine(tmp_path: Path) -> None:
    (tmp_path / "train_FD001.txt").write_text("1 1 0.0\n\n1 2 0.0\n2 1 0.0\n", encoding="utf-8")
    first, second = window("FD001/1", 0, 1.0), window("FD001/2", 1, 1.0)

    counted = CmapssGroundTruth(tmp_path).truths_of("cmapss", [first, second])

    assert counted == {first: 3.0, second: 2.0}


def test_a_corpus_nobody_registered_is_refused_rather_than_answered_by_whichever_reader() -> None:
    registered = CorpusGroundTruths({"cmapss": CmapssGroundTruth(sample("cmapss"))})

    with pytest.raises(UnknownGroundTruthError, match="knows no ground truth of corpus 'skab'"):
        registered.truths_of("skab", WINDOWS)


def test_every_window_of_a_stay_is_answered_with_the_outcome_the_stay_recorded() -> None:
    later = window("set-a/132551", 4, 96.0)

    answered = Physionet2012GroundTruth(sample("physionet2012"), DEATH).truths_of(
        "physionet2012", (*STAYS, later)
    )

    assert answered == {STAYS[0]: 1.0, STAYS[1]: 0.0, STAYS[2]: 0.0, later: 1.0}


@pytest.mark.parametrize("name", ["132551", "a/132551", "set-a/"])
def test_a_unit_not_named_by_set_and_record_is_refused(name: str) -> None:
    with pytest.raises(UnknownGroundTruthError, match="named"):
        Physionet2012GroundTruth(sample("physionet2012"), DEATH).truths_of(
            "physionet2012", [window(name, 0, 48.0)]
        )


def test_a_set_whose_outcomes_are_not_there_says_so(tmp_path: Path) -> None:
    with pytest.raises(UnreadableGroundTruthError, match="cannot read"):
        Physionet2012GroundTruth(tmp_path, DEATH).truths_of(
            "physionet2012", [window("set-c/152871", 0, 48.0)]
        )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "is empty"),
        ("RecordID,SOFA\n1,3\n", "no column In-hospital_death"),
        ("RecordID,In-hospital_death\n1,0,7\n", "3 fields"),
        ("RecordID,In-hospital_death\n1,0\n1,1\n", "not a new number"),
        ("RecordID,In-hospital_death\nx,0\n", "not a new number"),
        ("RecordID,In-hospital_death\n1,dead\n", "records 'dead'"),
    ],
)
def test_an_outcome_file_that_is_not_one_is_refused(
    tmp_path: Path, content: str, message: str
) -> None:
    (tmp_path / "Outcomes-a.txt").write_text(content, encoding="utf-8")

    with pytest.raises(UnreadableGroundTruthError, match=message):
        Physionet2012GroundTruth(tmp_path, DEATH).truths_of(
            "physionet2012", [window("set-a/1", 0, 48.0)]
        )
