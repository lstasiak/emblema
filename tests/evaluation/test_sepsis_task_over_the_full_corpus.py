"""The sepsis task's listings are what its rule makes of the challenge's own files, where fetched.

The frozen side and the stays the task cannot read are shipped as listings, since the frozen side
was cut from the corpus before anything was pretrained on it. A listing that drifted from the rule
would freeze stays the rule does not, so it is held to the rule over the whole download. The
counts are facts about the data that the preregistration quotes.
"""

import pytest

from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
from scripts.sepsis_task_units import AssessedStay, Standing, assessed, frozen, ineligible
from tests.support.corpora import raw_root

pytestmark = pytest.mark.skipif(
    raw_root("physionet2019") is None, reason="the PhysioNet 2019 download is not on this machine"
)
SEPSIS = KnownTasks.PHYSIONET2019_SEPSIS


@pytest.fixture(scope="module")
def stays() -> list[AssessedStay]:
    found = raw_root("physionet2019")
    assert found is not None  # the module is skipped without it
    return assessed(found)


def test_the_shipped_frozen_side_is_what_the_rule_freezes(stays: list[AssessedStay]) -> None:
    listed = sorted(str(unit) for unit in SEPSIS.frozen_test(frozenset()).units)

    assert listed == frozen(stays)


def test_the_shipped_ineligible_stays_are_what_the_rule_names(stays: list[AssessedStay]) -> None:
    assert sorted(SEPSIS.ineligible) == ineligible(stays)


def test_the_task_reads_three_in_four_stays_and_one_in_twenty_turns_septic(
    stays: list[AssessedStay],
) -> None:
    read = [stay for stay in stays if stay.standing is Standing.READ]

    assert len(stays) == 40336
    assert len(read) == 30378
    assert sum(stay.outcome or 0.0 for stay in read) == 1563
