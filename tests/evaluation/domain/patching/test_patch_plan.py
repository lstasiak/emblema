from tests.evaluation.support import adaptation_schedule, patch_plan, patch_spec


def test_the_plan_records_its_shape_its_seed_and_the_schedule_it_learns_under() -> None:
    stated = patch_plan(seed=3).parameters()

    assert stated["run_seed"] == 3
    assert {key: stated[key] for key in patch_spec().parameters()} == patch_spec().parameters()
    schedule = adaptation_schedule()
    assert (stated["epochs"], stated["min_steps"], stated["batch_size"]) == (
        schedule.epochs,
        schedule.min_steps,
        schedule.batch_size,
    )
    assert stated["learning_rate"] == schedule.learning_rate


def test_two_plans_that_differ_only_in_the_seed_render_the_same_columns() -> None:
    assert list(patch_plan(seed=1).parameters()) == list(patch_plan(seed=2).parameters())
