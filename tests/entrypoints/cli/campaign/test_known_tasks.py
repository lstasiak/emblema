"""The tasks a campaign can be told to define, as far as no corpus says it."""

from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.class_strata import ClassStrata
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.task.corpus_sides import CorpusSides
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
