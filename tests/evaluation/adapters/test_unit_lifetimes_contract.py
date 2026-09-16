"""Both adapters answer the same question about the same two engines.

The sample in the repository holds engines 39 and 91 of the first subset, recorded for 128 and 135
cycles. The corpus times a unit from cycle one to one past its last, so those engines fail at 129
and 136 — the numbers a window's end is compared against.
"""

from pathlib import Path

import pytest

from emblema.evaluation.adapters.in_memory.unit_lifetimes import InMemoryUnitLifetimes
from emblema.evaluation.adapters.readers.cmapss_unit_lifetimes import CmapssUnitLifetimes
from emblema.evaluation.domain.exceptions import (
    UnknownUnitLifetimeError,
    UnreadableGroundTruthError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.ports.unit_lifetimes import UnitLifetimes
from tests.support.corpora import sample

FIRST, SECOND = UnitKey("FD001/39"), UnitKey("FD001/91")
FAILURES = {FIRST: 129.0, SECOND: 136.0}


@pytest.fixture(params=["in memory", "C-MAPSS"])
def lifetimes(request: pytest.FixtureRequest) -> UnitLifetimes:
    if request.param == "in memory":
        return InMemoryUnitLifetimes(FAILURES)
    return CmapssUnitLifetimes(sample("cmapss"))


def test_each_unit_failed_one_past_its_last_cycle(lifetimes: UnitLifetimes) -> None:
    assert lifetimes.failure_times([FIRST, SECOND]) == FAILURES


def test_asking_about_one_unit_answers_about_that_unit(lifetimes: UnitLifetimes) -> None:
    assert lifetimes.failure_times([SECOND]) == {SECOND: 136.0}


def test_a_unit_the_ground_truth_says_nothing_about_is_refused(lifetimes: UnitLifetimes) -> None:
    with pytest.raises(UnknownUnitLifetimeError):
        lifetimes.failure_times([FIRST, UnitKey("FD001/999")])


def test_asking_about_nothing_answers_nothing(lifetimes: UnitLifetimes) -> None:
    assert lifetimes.failure_times([]) == {}


@pytest.mark.parametrize("name", ["39", "FD001/engine-39", "FD001/"])
def test_a_unit_not_named_by_subset_and_engine_is_refused(name: str) -> None:
    with pytest.raises(UnknownUnitLifetimeError, match="named"):
        CmapssUnitLifetimes(sample("cmapss")).failure_times([UnitKey(name)])


def test_a_subset_whose_file_is_not_there_says_so(tmp_path: Path) -> None:
    with pytest.raises(UnreadableGroundTruthError, match="cannot read"):
        CmapssUnitLifetimes(tmp_path).failure_times([UnitKey("FD001/1")])


def test_a_ground_truth_file_that_is_not_one_is_refused(tmp_path: Path) -> None:
    (tmp_path / "train_FD001.txt").write_text("engine cycle setting\n", encoding="utf-8")

    with pytest.raises(UnreadableGroundTruthError, match="engine number"):
        CmapssUnitLifetimes(tmp_path).failure_times([UnitKey("FD001/1")])


def test_an_empty_ground_truth_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "train_FD001.txt").write_text("\n   \n", encoding="utf-8")

    with pytest.raises(UnreadableGroundTruthError, match="no engine"):
        CmapssUnitLifetimes(tmp_path).failure_times([UnitKey("FD001/1")])


def test_blank_lines_between_records_are_not_an_engine(tmp_path: Path) -> None:
    (tmp_path / "train_FD001.txt").write_text("1 1 0.0\n\n1 2 0.0\n2 1 0.0\n", encoding="utf-8")

    failures = CmapssUnitLifetimes(tmp_path).failure_times([UnitKey("FD001/1"), UnitKey("FD001/2")])

    assert failures == {UnitKey("FD001/1"): 3.0, UnitKey("FD001/2"): 2.0}
