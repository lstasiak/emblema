import pytest
from pydantic import ValidationError

from emblema.catalog.adapters.synthetic.sensor_layout import SensorLayout
from emblema.shared.kernel.sampling import SamplingRegime
from tests.support.synthetic import HOSTILE


def layout(**dials: object) -> SensorLayout:
    return HOSTILE.with_dials(**dials)


def test_channels_are_named_by_position() -> None:
    assert layout(channels=3).channel_names == ("s01", "s02", "s03")


def test_a_layout_reporting_every_channel_on_every_step_is_regular() -> None:
    assert layout(cadence=1, synchronous=True).sampling_regime is SamplingRegime.REGULAR


def test_a_layout_that_skips_steps_is_irregular() -> None:
    assert layout(cadence=2, synchronous=True).sampling_regime is SamplingRegime.IRREGULAR


def test_a_layout_whose_channels_report_apart_is_irregular() -> None:
    assert layout(cadence=1, synchronous=False).sampling_regime is SamplingRegime.IRREGULAR


def test_the_shortest_unit_may_not_outrun_the_longest() -> None:
    with pytest.raises(ValidationError, match="shortest unit must not exceed"):
        layout(shortest_unit=9, longest_unit=8)


@pytest.mark.parametrize("coupling", [-0.1, 1.1])
def test_coupling_is_a_fraction(coupling: float) -> None:
    with pytest.raises(ValidationError):
        layout(coupling=coupling)


def test_a_layout_that_reports_nothing_at_all_is_not_a_layout() -> None:
    with pytest.raises(ValidationError):
        layout(missing=1.0)
