import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from emblema.shared.adapters.synthetic.dials import Dials
from emblema.shared.adapters.synthetic.draws import Draws


class LatentFactorProcess(Dials):
    """The hidden factors a synthetic corpus is a view of: a few signals defined for every instant.

    A factor is a sum of harmonics whose frequencies are drawn once, from the seed of the process,
    and whose amplitudes and phases are drawn per unit: the frequencies are the structure a model
    could carry to sensors it has never seen, the amplitudes and phases one unit's realisation of
    it. Two layouts on one process share the structure and no trajectory, which keeps the transfer
    leg from being a memory test. Factors are functions of time, as layouts sample at instants of
    their own, and have unit variance, so how strongly a channel follows one is a property of the
    channel.

    Attributes:
        harmonics: Sinusoids each factor is the sum of.
        shortest_period: Period of the fastest harmonic any factor may be given, in time units.
        longest_period: Period of the slowest, in time units.
        seed: Seed the frequencies are drawn with; the identity of the shared structure.
    """

    factors: int = Field(ge=1)
    harmonics: int = Field(ge=1)
    shortest_period: float = Field(gt=0.0)
    longest_period: float = Field(gt=0.0)
    seed: int

    @model_validator(mode="after")
    def _periods_span_a_band(self) -> "LatentFactorProcess":
        if self.shortest_period >= self.longest_period:
            raise ValueError(
                f"shortest period must precede the longest, got {self.shortest_period} "
                f"and {self.longest_period}"
            )
        return self

    def values_at(
        self, times: NDArray[np.float64], *, trajectory_seed: int, unit: int
    ) -> NDArray[np.float64]:
        """Every factor at every instant of ``times``, as a ``(len(times), factors)`` array.

        The realisation is addressed by ``trajectory_seed`` and by the unit's position: two
        layouts asking under different seeds see different trajectories of the same factors, and
        asking twice under the same seed returns the same ones. A position rather than a name,
        so that what a unit is a realisation of does not move when the corpus it belongs to is
        called something else.
        """
        draws = Draws(trajectory_seed, unit)
        amplitudes = draws.normal(self.factors, self.harmonics)
        # Unit variance: a sinusoid of amplitude a contributes a² / 2, and the harmonics are
        # independent, so scaling by the root of their summed contribution normalises the factor.
        amplitudes /= np.sqrt(np.square(amplitudes).sum(axis=1, keepdims=True) / 2.0)
        phases = 2.0 * np.pi * draws.uniform(self.factors, self.harmonics)
        angles = 2.0 * np.pi * times[:, None, None] * self.frequencies() + phases
        return (amplitudes * np.sin(angles)).sum(axis=2)

    def frequencies(self) -> NDArray[np.float64]:
        """The frequency of every harmonic of every factor, drawn evenly across the band in log.

        The structure two layouts share: their trajectories differ, their sensors differ, and
        what both are built out of is this one set of frequencies.
        """
        fractions = Draws(self.seed, "frequencies").uniform(self.factors, self.harmonics)
        longest, shortest = np.log(self.longest_period), np.log(self.shortest_period)
        return 1.0 / np.exp(longest + fractions * (shortest - longest))
