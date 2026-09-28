"""What the death-in-hospital task comes to on the challenge's own files, where they are fetched.

The prevalence and the frozen side are facts about the data rather than choices made here, and
both are quoted outside the code: the share of positives sets how many a budget of labels holds,
and the frozen side has to be the challenge's test set exactly, stay for stay.
"""

from pathlib import Path

import pytest

from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
from emblema.entrypoints.known_ground_truths import KnownGroundTruths
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from tests.support.corpora import raw_root

pytestmark = pytest.mark.skipif(
    raw_root("physionet2012") is None, reason="the PhysioNet 2012 download is not on this machine"
)
DEATH = KnownTasks.PHYSIONET_IN_HOSPITAL_DEATH


@pytest.fixture(scope="module")
def root() -> Path:
    found = raw_root("physionet2012")
    assert found is not None  # the module is skipped without it
    return found


def stays_of(root: Path, subset: str) -> list[str]:
    return sorted(f"{subset}/{path.stem}" for path in (root / subset).glob("*.txt"))


def test_every_stay_of_the_learnt_sets_has_an_outcome_and_one_in_seven_died(root: Path) -> None:
    stays = [*stays_of(root, "set-a"), *stays_of(root, "set-b")]
    windows = [TaskWindow(unit=UnitKey(stay), position=0, ends_at=48.0) for stay in stays]

    outcomes = KnownGroundTruths.under(root.parent).truths_of(DEATH.corpus, windows)

    assert len(outcomes) == 8000
    assert set(outcomes.values()) == {0.0, 1.0}
    assert sum(outcomes.values()) / len(outcomes) == pytest.approx(0.1403, abs=5e-4)


def test_the_frozen_side_is_the_challenges_test_set_stay_for_stay(root: Path) -> None:
    listed = {str(unit) for unit in DEATH.frozen_test(frozenset()).units}

    outcomes_c = {
        f"set-c/{line.split(',')[0]}"
        for line in (root / "Outcomes-c.txt").read_text().splitlines()[1:]
        if line.strip()
    }
    assert listed == outcomes_c == set(stays_of(root, "set-c"))
