"""The experiment files of the curve over the scale of pretraining keep the rules registered for
it: a task's corpus is read as often at every point, the points of one publication differ in the
amount of data alone, a point without a corpus differs from the whole point in that corpus
alone, the larger shape differs from today's in the shape, the micro-batch and the rate, the
longer run differs from the whole point in the length of its training alone, the rows of a
corpus left out are fitted on it alone, continuing the point without it, with a control that
differs in the start alone, and the whole point continued over the clinical corpora keeps the
clinical recipe's effective batch and the shape of its schedule, with a control that differs in
the parameters moved alone."""

from pathlib import Path

import pytest

from emblema.pretraining.adapters.experiments.experiment_file import ExperimentFile
from emblema.pretraining.domain.training.experiment_configuration import (
    ExperimentConfiguration,
)
from emblema.pretraining.domain.training.trained_parameters import TrainedParameters

EXPERIMENTS = Path(__file__).resolve().parents[4] / "experiments"
CORPORA_OF_THE_TASKS = ("cmapss", "physionet2012", "physionet2019")
READS_BY_THE_MIXTURE_OF_FOUR = 8
TENTH_POINT = "backbone-scale-1e8new-m"
SMALLER_POINT = "backbone-scale-3e8-m"
WHOLE_POINT = "backbone-scale-1e9-m"
LARGER_SHAPE = "backbone-scale-1e9-512x8-m"
LONGER_TRAINING = "backbone-scale-1e9-4ep-m"
MIXTURE_OF_FOUR = "backbone-mixed4-m"
LATENT_OBJECTIVE = "backbone-mixed4-latent-m"
THE_TARGET = {"target", "target_layers", "momentum", "final_momentum"}
LEFT_OUT = ("physionet2012", "cmapss", "physionet2019")
ROWS_FITTED = ("physionet2012", "cmapss", "physionet2019")
# The fits whose control, the same fit on an encoder drawn from the seed, is registered too.
ROWS_CONTROLLED = ("physionet2012", "physionet2019")
CLINICAL_RECIPE = "backbone-clinical-m"
CONTINUED_ON_CLINICAL = "backbone-scale-1e9-continued-clinical-m"
# What a fit of the rows states beside the point it continues: how it starts, what it moves,
# the budget of the fit and the precision it was read at.
THE_FIT = {
    "start",
    "trained",
    "corpus_fraction",
    "precision",
    "epochs",
    "batch_size",
    "accumulation_steps",
    "effective_batch_size",
    "learning_rate",
    "warmup_epochs",
    "final_lr_fraction",
}
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


def test_the_second_objective_differs_from_the_mixture_of_four_in_the_target_alone() -> None:
    assert differing(configured(MIXTURE_OF_FOUR), configured(LATENT_OBJECTIVE)) == {
        "name",
        *THE_TARGET,
    }
    assert differing(
        configured(f"{MIXTURE_OF_FOUR}-seed2"), configured(f"{LATENT_OBJECTIVE}-seed2")
    ) == {"name", *THE_TARGET}
    assert differing(configured(LATENT_OBJECTIVE), configured(f"{LATENT_OBJECTIVE}-seed2")) == {
        "name",
        "seed",
    }
    latent = configured(LATENT_OBJECTIVE)
    assert latent.target.is_latent
    assert latent.target.layers == 4
    assert latent.architecture.layers == 6


@pytest.mark.parametrize("left_out", ROWS_FITTED)
def test_the_rows_of_a_corpus_left_out_are_fitted_on_it_alone_continuing_the_point_without_it(
    left_out: str,
) -> None:
    without = f"{WHOLE_POINT[:-2]}-without-{left_out}-m"
    rows = f"{without[:-2]}-rows-m"
    point, fit = configured(without), configured(rows)

    assert stated(rows).corpora == (left_out,)
    assert fit.start is not None
    assert fit.trained is TrainedParameters.CHANNEL_ROWS
    assert (fit.budget.warmup_epochs, fit.budget.final_lr_fraction) == (0.0, 1.0)
    assert fit.budget.effective_batch_size == 64
    passes = {f"passes.{corpus}" for corpus in CORPORA_OF_THE_TASKS}
    assert differing(point, fit) <= {"name", *THE_FIT, *passes}


@pytest.mark.parametrize("left_out", ROWS_CONTROLLED)
def test_the_control_of_a_fit_of_the_rows_differs_from_it_in_the_start_alone(left_out: str) -> None:
    rows = f"{WHOLE_POINT[:-2]}-without-{left_out}-rows-m"
    untrained = f"{rows[:-2]}-untrained-m"

    assert stated(untrained).corpora == stated(rows).corpora
    assert configured(untrained).start is None
    assert differing(configured(rows), configured(untrained)) == {"name", "start"}


def test_the_continuation_on_the_clinical_corpora_keeps_the_recipes_batch_and_schedule() -> None:
    recipe, continued = configured(CLINICAL_RECIPE), configured(CONTINUED_ON_CLINICAL)

    assert stated(CONTINUED_ON_CLINICAL).corpora == stated(CLINICAL_RECIPE).corpora
    assert continued.start is not None
    assert continued.trained is TrainedParameters.EVERYTHING
    assert continued.budget.effective_batch_size == recipe.budget.effective_batch_size
    assert continued.budget.final_lr_fraction == recipe.budget.final_lr_fraction
    assert 0 < continued.budget.warmup_epochs < continued.budget.epochs < recipe.budget.epochs
    assert continued.budget.learning_rate == recipe.budget.learning_rate / 10
    assert differing(recipe, continued) == {
        "name",
        "start",
        "precision",
        "epochs",
        "batch_size",
        "accumulation_steps",
        "learning_rate",
        "warmup_epochs",
    }
    passes = {f"passes.{corpus}" for corpus in CORPORA_OF_THE_TASKS}
    assert differing(configured(WHOLE_POINT), continued) <= {"name", *THE_FIT, *passes} - {
        "trained",
        "corpus_fraction",
    }


def test_the_second_seed_of_the_continuation_differs_from_it_in_the_seed_alone() -> None:
    assert differing(
        configured(CONTINUED_ON_CLINICAL), configured(f"{CONTINUED_ON_CLINICAL}-seed2")
    ) == {"name", "seed"}


def test_the_control_of_the_continuation_differs_from_it_in_the_parameters_moved_alone() -> None:
    rows = f"{CONTINUED_ON_CLINICAL[:-2]}-rows-m"

    assert stated(rows).corpora == stated(CONTINUED_ON_CLINICAL).corpora
    assert configured(rows).trained is TrainedParameters.CHANNEL_ROWS
    assert differing(configured(CONTINUED_ON_CLINICAL), configured(rows)) == {"name", "trained"}
