"""The synthetic corpora the tests generate: the shape of the control, cut to a test's budget.

``miniature`` takes a control layout down to a handful of short units, which is what a test that
publishes a corpus can afford. It is the one preset worth a name; anything else a test wants is
one ``with_dials`` call away, and a test taking fewer units without shortening them says so by
turning that dial alone — how much of a channel its factors explain depends on a unit being long
against the slowest of them, so a short unit is fitted well by any smooth curve and the shuffled
baseline a measurement is read against climbs.

``HOSTILE`` is the layout no preset is: barely any signal survives it, a unit reports nothing at
all and a channel of another stays silent, which is what the port contract has to hold for.
"""

from emblema.catalog.adapters.synthetic.latent_factor_process import LatentFactorProcess
from emblema.catalog.adapters.synthetic.sensor_layout import SensorLayout

PROCESS = LatentFactorProcess(
    factors=3, harmonics=2, shortest_period=4.0, longest_period=40.0, seed=7
)

HOSTILE = SensorLayout(
    name="hostile",
    channels=3,
    factors_per_channel=2,
    coupling=0.8,
    noise=0.1,
    missing=0.6,
    cadence=2,
    synchronous=False,
    units=4,
    shortest_unit=1,
    longest_unit=8,
    time_step=0.5,
    gain_spread=0.1,
    seed=24,
    trajectory_seed=13,
)


def miniature(layout: SensorLayout, *, units: int = 4) -> SensorLayout:
    """A control layout cut to a few short units: enough to travel the pipeline, not to measure."""
    return layout.with_dials(units=units, shortest_unit=96, longest_unit=160)
