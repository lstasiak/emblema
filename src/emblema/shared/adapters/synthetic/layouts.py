"""The synthetic corpora of the positive control: two coupled layouts and their null pair.

The control tells a broken pipeline from a disproved thesis: transfer failing where shared structure
was put on purpose is a fault of ours, failing where none was put is the pipeline behaving.
``control-a`` and ``control-b`` watch the same factors through different sensors, rates and losses,
and share no trajectory. ``null-a`` and ``null-b`` have the same shape and signal strength but no
shared structure, so whatever transfer finds between them the pipeline invented.

Constants rather than a registry: they are the generator's own specification, and a control whose
dials can be turned from a command line is not a control.
"""

from collections.abc import Mapping

from emblema.shared.adapters.synthetic.latent_factor_process import LatentFactorProcess
from emblema.shared.adapters.synthetic.sensor_layout import SensorLayout

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


# One dial apart from their coupled twins, in the data and not only in the specification: the
# same sensors responding to the same factors at the same instants, losing the same observations
# and off by the same amounts, because the reader addresses its draws by a unit's position and
# never by the name in its key. What differs is only where the signal comes from, so a difference
# in what transfer achieves has one explanation.
NULL_A = CONTROL_A.with_dials(name="null-a", coupling=0.0)
NULL_B = CONTROL_B.with_dials(name="null-b", coupling=0.0)

# The second layout of each pair with more units and nothing else turned: the corpus a transfer
# is measured on, where the width of the paired interval over held-out units is what decides
# whether an equivalence can be read at all. The first hundred and twenty units are the narrow
# layout's own, draw for draw, because a unit's draws are addressed by its position.
CONTROL_B_WIDE = CONTROL_B.with_dials(name="control-b-wide", units=800)
NULL_B_WIDE = NULL_B.with_dials(name="null-b-wide", units=800)

# The second layout watching the first's own factor trajectories, unit for unit: a leak, not a
# control, and therefore the ceiling of what transfer between the two can give. As many units as
# the first layout has, so that every trajectory here is one the first layout was pretrained on.
CONTROL_B_SHARED = CONTROL_B.with_dials(
    name="control-b-shared", units=CONTROL_A.units, trajectory_seed=CONTROL_A.trajectory_seed
)

# The first layout of the null pair with its signal drowned: the sensors report noise, with the
# same channels, cadence and losses. A backbone pretrained here learns the mechanics of reading a
# corpus — embeddings, normalisation, attention over a window — and nothing about any signal,
# which is the share of a transfer that any pretraining at all would give.
NOISE_A = NULL_A.with_dials(name="noise-a", noise=100.0)

LAYOUTS: Mapping[str, SensorLayout] = {
    layout.name: layout
    for layout in (
        CONTROL_A,
        CONTROL_B,
        NULL_A,
        NULL_B,
        CONTROL_B_WIDE,
        NULL_B_WIDE,
        CONTROL_B_SHARED,
        NOISE_A,
    )
}
