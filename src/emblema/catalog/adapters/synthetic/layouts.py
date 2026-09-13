"""The synthetic corpora of the positive control: two coupled layouts and their null pair.

Constants rather than a registry class, for the same reason the subset names of a file-backed
reader are constants: they are the adapter's own specification, not facts about the data that
only the process assembling it could know. A control whose dials can be turned from a command
line is not a control, so the pairs are stated here, named, and changed by editing them.

``control-a`` and ``control-b`` watch the same factors through different sensors, at different
rates and with different losses, and share no trajectory. ``null-a`` and ``null-b`` have the same
shape and the same signal strength but no shared structure at all; what transfer finds between
them is what the pipeline invents.
"""

from collections.abc import Mapping

from emblema.catalog.adapters.synthetic.latent_factor_process import LatentFactorProcess
from emblema.catalog.adapters.synthetic.sensor_layout import SensorLayout

# Four factors over periods from a couple of dozen steps to a few hundred. The band is chosen
# against the layouts that watch it: the sparsest of them reports every three steps, so the
# fastest harmonic is still sampled several times a cycle, and the slowest fits inside a unit. A
# positive control is meant to be easy — a corpus whose signal is aliased away tests the sampling
# rate, not the pipeline.
CONTROL_PROCESS = LatentFactorProcess(
    factors=4, harmonics=3, shortest_period=24.0, longest_period=300.0, seed=101
)

# The dense, regular layout: many sensors reporting on every step, little noise, little lost.
CONTROL_A = SensorLayout(
    name="control-a",
    channels=8,
    factors_per_channel=2,
    coupling=1.0,
    noise=0.05,
    missing=0.02,
    cadence=1,
    synchronous=True,
    units=160,
    shortest_unit=320,
    longest_unit=640,
    time_step=1.0,
    gain_spread=0.2,
    seed=201,
    trajectory_seed=301,
)

# The sparse, irregular layout: fewer sensors, each on its own schedule, noisier and losing more.
# Transfer between the two is a transfer across both axes of heterogeneity at once.
CONTROL_B = SensorLayout(
    name="control-b",
    channels=5,
    factors_per_channel=2,
    coupling=1.0,
    noise=0.08,
    missing=0.10,
    cadence=3,
    synchronous=False,
    units=120,
    shortest_unit=320,
    longest_unit=640,
    time_step=1.0,
    gain_spread=0.2,
    seed=202,
    trajectory_seed=302,
)


def _uncoupled(layout: SensorLayout, name: str) -> SensorLayout:
    """The same layout with the shared factors switched off, and nothing else changed.

    One dial apart from its coupled twin: the same sensors responding to the same factors at the
    same instants, losing the same observations and off by the same amounts. What differs is only
    where the signal comes from, so a difference in what transfer achieves has one explanation.
    """
    return SensorLayout.model_validate({**layout.model_dump(), "name": name, "coupling": 0.0})


NULL_A = _uncoupled(CONTROL_A, "null-a")
NULL_B = _uncoupled(CONTROL_B, "null-b")

LAYOUTS: Mapping[str, SensorLayout] = {
    layout.name: layout for layout in (CONTROL_A, CONTROL_B, NULL_A, NULL_B)
}
