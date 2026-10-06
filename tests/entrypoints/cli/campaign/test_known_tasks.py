"""The tasks a campaign can be told to define, as far as no corpus says it."""

import hashlib
from importlib.resources import files

import pytest

from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.class_strata import ClassStrata
from emblema.evaluation.domain.labels.outcome_scheme import OutcomeScheme
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.task.corpus_sides import CorpusSides
from emblema.evaluation.domain.task.task_windows import TaskWindows
from tests.evaluation.support import MANIFEST

DEATH = KnownTasks.PHYSIONET_IN_HOSPITAL_DEATH


def sides(*training: str, validation: tuple[str, ...]) -> CorpusSides:
    return CorpusSides(
        corpus=DEATH.corpus,
        training=frozenset(map(UnitKey, training)),
        validation=frozenset(map(UnitKey, validation)),
    )


def test_the_death_in_hospital_task_is_spread_over_its_outcomes() -> None:
    command = DEATH.defined_over(
        MANIFEST, sides("set-a/1", "set-b/2", validation=("set-a/3", "set-b/4"))
    )

    assert command.strata == ClassStrata()
    assert command.labels == DEATH.labels
    assert command.units == frozenset(map(UnitKey, ("set-a/1", "set-b/2", "set-a/3", "set-b/4")))


def test_the_frozen_side_is_set_c_as_the_challenge_published_it_whatever_the_corpus_holds() -> None:
    command = DEATH.defined_over(MANIFEST, sides("set-a/1", validation=("set-a/3",)))

    assert command.test.source == "physionet2012/set-c"
    assert len(command.test.units) == 4000
    assert all(str(unit).startswith("set-c/") for unit in command.test.units)
    assert UnitKey("set-c/152871") in command.test.units


def test_a_task_over_a_quantity_is_still_spread_over_ranks_of_its_target() -> None:
    fd001 = KnownTasks.TURBOFAN_FD001

    command = fd001.defined_over(
        MANIFEST,
        CorpusSides(
            corpus=fd001.corpus,
            training=frozenset({UnitKey("FD001/1")}),
            validation=frozenset({UnitKey("FD001/2")}),
        ),
    )

    assert command.strata == TargetBins(4)


def test_the_task_is_named_among_the_known_ones() -> None:
    assert KnownTasks.named("physionet2012-in-hospital-death") is DEATH


SEPSIS = KnownTasks.PHYSIONET2019_SEPSIS
HOSPITAL_B = KnownTasks.PHYSIONET2019_SEPSIS_HOSPITAL_B
# One stay of each listing, and two the task reads, one per hospital.
FROZEN_A, FROZEN_B = "training_setA/p000001", "training_setB/p100021"
INELIGIBLE = "training_setA/p000028"
SHORT = "training_setA/p000015"
READ_A, READ_B = "training_setA/p000003", "training_setB/p100001"


def stays(*training: str, validation: tuple[str, ...]) -> CorpusSides:
    return CorpusSides(
        corpus=SEPSIS.corpus,
        training=frozenset(map(UnitKey, training)),
        validation=frozenset(map(UnitKey, validation)),
    )


def test_the_sepsis_task_reads_the_first_window_of_each_stay_spread_over_its_outcomes() -> None:
    command = SEPSIS.defined_over(MANIFEST, stays(READ_A, validation=(READ_B,)))

    assert command.windows is TaskWindows.FIRST
    assert command.strata == ClassStrata()
    assert command.labels == OutcomeScheme("SepsisLabel")
    assert command.units == frozenset(map(UnitKey, (READ_A, READ_B)))


def test_a_stay_that_cannot_answer_the_sepsis_question_is_not_one_of_its_units() -> None:
    command = SEPSIS.defined_over(MANIFEST, stays(READ_A, INELIGIBLE, validation=(READ_B, SHORT)))

    assert command.units == frozenset(map(UnitKey, (READ_A, READ_B)))


def test_the_frozen_sepsis_side_is_a_fifth_of_both_hospitals_and_outside_the_task() -> None:
    command = SEPSIS.defined_over(MANIFEST, stays(READ_A, validation=(READ_B,)))

    assert command.test.source == "physionet2019/sepsis-frozen"
    assert len(command.test.units) == 6077
    assert {UnitKey(FROZEN_A), UnitKey(FROZEN_B)} <= command.test.units
    assert not command.units & command.test.units
    assert not command.test.units & frozenset(map(UnitKey, SEPSIS.ineligible))


def test_the_second_hospitals_task_reads_and_freezes_its_own_stays_alone() -> None:
    command = HOSPITAL_B.defined_over(MANIFEST, stays(READ_A, validation=(READ_B,)))

    assert command.units == frozenset({UnitKey(READ_B)})
    assert len(command.test.units) == 2947
    assert all(str(unit).startswith("training_setB/") for unit in command.test.units)
    assert command.windows is TaskWindows.FIRST


def test_the_sepsis_tasks_are_named_among_the_known_ones() -> None:
    assert KnownTasks.named("physionet2019-sepsis") is SEPSIS
    assert KnownTasks.named("physionet2019-sepsis-hospital-b") is HOSPITAL_B


def test_a_task_reading_every_window_keeps_its_frozen_side_whatever_its_prefix() -> None:
    command = DEATH.defined_over(MANIFEST, sides("set-a/1", validation=("set-a/3",)))

    assert command.windows is TaskWindows.EVERY
    assert len(command.test.units) == 4000


@pytest.mark.parametrize(
    ("listing", "digest"),
    [
        (
            "physionet2019_sepsis_test.txt",
            "2e8f019fa24892ff7a9782c88cfe9ebe99b29cb223036e66f8813e41cbd8d2cf",
        ),
        (
            "physionet2019_sepsis_ineligible.txt",
            "8a2943000c1d0066f4ece3a6d51c6e66344ffe30f788a86ddc183529338132f3",
        ),
    ],
)
def test_the_sepsis_listings_are_the_ones_the_corpus_was_published_without(
    listing: str, digest: str
) -> None:
    # The frozen stays were cut out of the published corpus by this listing; an edit to it would
    # name a frozen side the backbones may have read.
    content = files("emblema.entrypoints.cli.campaign").joinpath(listing).read_bytes()

    assert hashlib.sha256(content).hexdigest() == digest


def test_the_frozen_sepsis_side_holds_a_fifth_of_each_hospital_and_outcome_and_no_ineligible() -> (
    None
):
    frozen = {str(unit) for unit in SEPSIS.frozen_test(frozenset()).units}

    assert len(frozen) == 6077
    assert sum(unit.startswith("training_setB/") for unit in frozen) == 2947
    assert not frozen & SEPSIS.ineligible
