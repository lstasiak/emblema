"""The corpus the spectral diagnostic can be read on: the dense control, watching faster factors.

The spectral diagnostic asks which frequencies of a channel hidden whole the model gives back — a
model that returns the slow ones and loses the fast ones has learnt that channels are smooth, not
how they move. On the control corpora it cannot answer: the fastest harmonic of the control
process completes about one cycle in the report's window of 32 steps, so once a constant and a
trend are fitted away every frequency above the first holds noise alone. This probe is the answer
to that, and nothing else.

It is ``control-a`` with one dial turned: the band the factors' periods are drawn from, moved
from 24 to 300 steps to 6 to 32, so that harmonics complete from about one to five cycles in a
window. The process keeps its seed and the layout keeps its seeds, so the probe draws the same
relative frequencies, the same sensors, the same instants, the same losses and the same noise as
the control — a difference in what the spectrum shows has one explanation. Six steps a cycle at a
cadence of one keeps the fastest harmonic sampled well above its Nyquist rate. The layout is cut to
half the control's units, which keeps a report run short and leaves twenty validation units for
the verdict to resample.

Kept out of the catalog's control registry on purpose. The registry states the control's claim —
two coupled layouts and their null twins — and is enumerated wherever the control is published or
measured; a probe is not a control, is never transferred to or from, and exists for one report.
"""

from emblema.catalog.adapters.synthetic.latent_factor_process import LatentFactorProcess
from emblema.catalog.adapters.synthetic.layouts import CONTROL_A, CONTROL_PROCESS
from emblema.catalog.adapters.synthetic.sensor_layout import SensorLayout

SPECTRAL_PROCESS: LatentFactorProcess = CONTROL_PROCESS.with_dials(
    shortest_period=6.0, longest_period=32.0
)
SPECTRAL_PROBE: SensorLayout = CONTROL_A.with_dials(name="spectral-probe", units=80)
