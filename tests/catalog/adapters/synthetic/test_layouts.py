from emblema.catalog.adapters.synthetic.layouts import (
    CONTROL_A,
    CONTROL_B,
    LAYOUTS,
    NULL_A,
    NULL_B,
)
from emblema.shared.kernel.sampling import SamplingRegime


def test_every_layout_is_registered_under_its_own_name() -> None:
    assert set(LAYOUTS) == {"control-a", "control-b", "null-a", "null-b"}
    assert all(name == layout.name for name, layout in LAYOUTS.items())


def test_the_control_pair_shares_the_factors_and_no_trajectory() -> None:
    assert CONTROL_A.trajectory_seed != CONTROL_B.trajectory_seed
    assert CONTROL_A.seed != CONTROL_B.seed


def test_the_control_pair_differs_on_both_axes_of_heterogeneity() -> None:
    assert CONTROL_A.channels != CONTROL_B.channels
    assert CONTROL_A.sampling_regime is SamplingRegime.REGULAR
    assert CONTROL_B.sampling_regime is SamplingRegime.IRREGULAR


def test_a_null_layout_is_its_control_with_one_dial_turned() -> None:
    for control, null in ((CONTROL_A, NULL_A), (CONTROL_B, NULL_B)):
        differences = {
            field
            for field, value in control.model_dump().items()
            if null.model_dump()[field] != value
        }

        assert differences == {"name", "coupling"}
        assert control.coupling == 1.0
        assert null.coupling == 0.0
