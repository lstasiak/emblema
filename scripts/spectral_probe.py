"""The corpus the spectral diagnostic can be read on: the dense control, watching faster factors.

On the controls the fastest harmonic completes about one cycle in the report's window of 32 steps,
so every higher frequency holds noise and the spectrum cannot tell structure from smoothness. The
probe is ``control-a`` with the factor periods drawn from 6 to 32 steps instead of 24 to 300 (one to
five cycles a window, well above Nyquist), on half the units. It keeps every seed and draws the same
sensors, instants, losses and noise, so a difference in the spectrum has one cause.

Kept out of the control registry on purpose: a probe is not a control and is never transferred.
"""

from emblema.shared.adapters.synthetic.latent_factor_process import LatentFactorProcess
from emblema.shared.adapters.synthetic.layouts import CONTROL_A, CONTROL_PROCESS
from emblema.shared.adapters.synthetic.sensor_layout import SensorLayout

SPECTRAL_PROCESS: LatentFactorProcess = CONTROL_PROCESS.with_dials(
    shortest_period=6.0, longest_period=32.0
)
SPECTRAL_PROBE: SensorLayout = CONTROL_A.with_dials(name="spectral-probe", units=80)
