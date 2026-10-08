"""The experiment files of the curve over the scale of pretraining keep the rules registered for
it: a task's corpus is read as often at every point, the points of one publication differ in the
amount of data alone, a point without a corpus differs from the whole point in that corpus
alone, the larger shape differs from today's in the shape, the micro-batch and the rate, and the
longer run differs from the whole point in the length of its training alone."""

from pathlib import Path

import pytest

from emblema.pretraining.adapters.experiments.experiment_file import ExperimentFile
from emblema.pretraining.domain.training.experiment_configuration import (
    ExperimentConfiguration,
)

EXPERIMENTS = Path(__file__).resolve().parents[4] / "experiments"
CORPORA_OF_THE_TASKS = ("cmapss", "physionet2012", "physionet2019")
READS_BY_THE_MIXTURE_OF_FOUR = 8
TENTH_POINT = "backbone-scale-1e8new-m"
SMALLER_POINT = "backbone-scale-3e8-m"
WHOLE_POINT = "backbone-scale-1e9-m"
LARGER_SHAPE = "backbone-scale-1e9-512x8-m"
LONGER_TRAINING = "backbone-scale-1e9-4ep-m"
LEFT_OUT = ("physionet2012", "cmapss", "physionet2019")
POINTS = (
    TENTH_POINT,
    SMALLER_POINT,
    WHOLE_POINT,
    LARGER_SHAPE,
    LONGER_TRAINING,
    *(f"{WHOLE_POINT[:-2]}-without-{corpus}-m" for corpus in LEFT_OUT),
)


def stated(name: str) -> ExperimentFile:
    return ExperimentFile.load(EXPERIMENTS / f"{name}.toml")


def configured(name: str) -> ExperimentConfiguration:
    return stated(name).configuration()


def differing(one: ExperimentConfiguration, other: ExperimentConfiguration) -> set[str]:
    first, second = one.parameters(), other.parameters()
    return {key for key in first.keys() | second.keys() if first.get(key) != second.get(key)}


@pytest.mark.parametrize("name", POINTS)
def test_a_point_reads_the_corpora_of_the_tasks_as_often_as_the_mixture_of_four(name: str) -> None:
    configuration = configured(name)

    for corpus in CORPORA_OF_THE_TASKS:
        if corpus in stated(name).corpora:
            reads = configuration.budget.epochs * configuration.passes_of(corpus)
            assert reads == READS_BY_THE_MIXTURE_OF_FOUR, corpus


def test_the_two_points_of_one_publication_differ_in_the_amount_of_data_alone() -> None:
    smaller, whole = stated(SMALLER_POINT), stated(WHOLE_POINT)
    beyond_the_four = smaller.corpora[4:]

    assert whole.corpora == (*smaller.corpora, "physionet2019")
    assert beyond_the_four[0] == "tep"
    assert all(corpus.startswith("utsd/") for corpus in beyond_the_four[1:])
    assert smaller.fraction == dict.fromkeys(beyond_the_four, 0.25)
    assert whole.fraction == {}
    assert all(
        key in {"name", "epochs"} or key.startswith(("passes.", "fraction."))
        for key in differing(configured(SMALLER_POINT), configured(WHOLE_POINT))
    )


def test_the_tenth_differs_from_the_quarter_in_the_share_of_the_new_corpora_alone() -> None:
    tenth, quarter = stated(TENTH_POINT), stated(SMALLER_POINT)

    assert tenth.corpora == quarter.corpora
    assert tenth.fraction == dict.fromkeys(quarter.fraction, 0.1)
    assert all(
        key == "name" or key.startswith("fraction.")
        for key in differing(configured(TENTH_POINT), configured(SMALLER_POINT))
    )


@pytest.mark.parametrize("left_out", LEFT_OUT)
def test_a_point_without_a_corpus_differs_from_the_whole_point_in_that_corpus_alone(
    left_out: str,
) -> None:
    without = f"{WHOLE_POINT[:-2]}-without-{left_out}-m"

    assert stated(without).corpora == tuple(
        corpus for corpus in stated(WHOLE_POINT).corpora if corpus != left_out
    )
    assert differing(configured(WHOLE_POINT), configured(without)) <= {
        "name",
        f"passes.{left_out}",
    }


def test_the_larger_shape_differs_from_todays_in_shape_micro_batch_and_rate_alone() -> None:
    todays, larger = configured(WHOLE_POINT), configured(LARGER_SHAPE)

    assert stated(LARGER_SHAPE).corpora == stated(WHOLE_POINT).corpora
    assert differing(todays, larger) <= {
        "name",
        "width",
        "heads",
        "layers",
        "feedforward_width",
        "batch_size",
        "accumulation_steps",
        "learning_rate",
    }
    assert (
        larger.parameters()["effective_batch_size"] == todays.parameters()["effective_batch_size"]
    )


def test_the_longer_run_differs_from_the_whole_point_in_the_length_of_its_training_alone() -> None:
    whole, longer = configured(WHOLE_POINT), configured(LONGER_TRAINING)

    assert stated(LONGER_TRAINING).corpora == stated(WHOLE_POINT).corpora
    assert longer.budget.epochs == 2 * whole.budget.epochs
    assert differing(whole, longer) == {
        "name",
        "epochs",
        *(f"passes.{corpus}" for corpus in CORPORA_OF_THE_TASKS),
    }
