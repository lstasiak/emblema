from emblema.shared.adapters.synthetic.layouts import (
    CONTROL_A,
    CONTROL_B,
    CONTROL_B_SHARED,
    CONTROL_B_WIDE,
    LAYOUTS,
    NOISE_A,
    NULL_A,
    NULL_B,
    NULL_B_WIDE,
)
from emblema.shared.kernel.sampling import SamplingRegime


def test_every_layout_is_registered_under_its_own_name() -> None:
    assert set(LAYOUTS) == {
        "control-a",
        "control-b",
        "null-a",
        "null-b",
        "control-b-wide",
        "null-b-wide",
        "control-b-shared",
        "noise-a",
    }
    assert all(name == layout.name for name, layout in LAYOUTS.items())


def test_the_control_pair_shares_the_factors_and_no_trajectory() -> None:
    assert CONTROL_A.trajectory_seed != CONTROL_B.trajectory_seed
    assert CONTROL_A.seed != CONTROL_B.seed


def test_the_control_pair_differs_on_both_axes_of_heterogeneity() -> None:
    assert CONTROL_A.channels != CONTROL_B.channels
    assert CONTROL_A.sampling_regime is SamplingRegime.REGULAR
    assert CONTROL_B.sampling_regime is SamplingRegime.IRREGULAR


def test_a_null_layout_is_its_control_with_one_dial_turned() -> None:
    for control, null in ((CONTROL_A, NULL_A), (CONTROL_B, NULL_B), (CONTROL_B_WIDE, NULL_B_WIDE)):
        differences = {
            field
            for field, value in control.model_dump().items()
            if null.model_dump()[field] != value
        }

        assert differences == {"name", "coupling"}
        assert control.coupling == 1.0
        assert null.coupling == 0.0


def test_a_wide_layout_is_its_narrow_one_with_more_units_and_nothing_else() -> None:
    for narrow, wide in ((CONTROL_B, CONTROL_B_WIDE), (NULL_B, NULL_B_WIDE)):
        differences = {
            field
            for field, value in narrow.model_dump().items()
            if wide.model_dump()[field] != value
        }

        assert differences == {"name", "units"}
        assert wide.units > narrow.units


def test_the_shared_layout_watches_the_first_layouts_trajectories_and_nothing_else_of_it() -> None:
    differences = {
        field
        for field, value in CONTROL_B.model_dump().items()
        if CONTROL_B_SHARED.model_dump()[field] != value
    }

    assert differences == {"name", "units", "trajectory_seed"}
    assert CONTROL_B_SHARED.trajectory_seed == CONTROL_A.trajectory_seed
    assert CONTROL_B_SHARED.units == CONTROL_A.units
    assert CONTROL_B_SHARED.seed == CONTROL_B.seed


def test_the_noise_layout_is_the_null_ones_first_layout_with_its_signal_drowned() -> None:
    differences = {
        field
        for field, value in NULL_A.model_dump().items()
        if NOISE_A.model_dump()[field] != value
    }

    assert differences == {"name", "noise"}
    assert NOISE_A.noise >= 100.0 * NULL_A.noise
