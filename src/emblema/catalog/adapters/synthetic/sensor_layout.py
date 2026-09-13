from pydantic import Field, model_validator

from emblema.catalog.adapters.synthetic.dials import Dials
from emblema.shared.kernel.sampling import SamplingRegime


class SensorLayout(Dials):
    """One instrumented system watching the latent factors: how many sensors, how well, how often.

    A layout is a partial, noisy and unevenly sampled view of the factors. Every dial that makes
    two layouts different is here — the number of channels, how many factors a channel responds
    to, how closely it follows them, how often it reports and how much it drops — so that a pair
    of layouts can be stated rather than written.

    ``coupling`` is the dial the control turns. At one, a channel follows the shared factors; at
    zero, it follows factors of its own that no other layout sees. The two are mixed so that the
    signal keeps its size either way, which is what makes the zero case an honest null: a layout
    without shared structure is not also a layout with less to see, so transfer failing there
    cannot be put down to a fainter signal.

    Attributes:
        name: Name the corpus generated from this layout is registered under, and the prefix of
            its unit keys.
        factors_per_channel: Factors one sensor responds to; fewer than all makes the view partial.
        coupling: How much of a channel's signal comes from the shared factors rather than from
            factors private to this layout, between zero and one.
        noise: Standard deviation of the measurement noise, against a signal of unit variance.
        missing: Share of observations the layout fails to report.
        cadence: Mean gap between one observation of a channel and the next, in grid steps; one
            means every step.
        synchronous: Whether every channel is observed at the same instants.
        units: Independent systems of this layout the corpus holds.
        shortest_unit: Grid steps the shortest unit spans.
        longest_unit: Grid steps the longest unit spans.
        time_step: Distance between neighbouring grid steps, in the corpus's time unit.
        gain_spread: How far a unit's gain — the static feature scaling its shared signal — may
            stray from one.
        seed: Seed of everything this layout decides on its own: which factors each channel sees,
            its private factors, when it samples, what it drops and how much it is off by.
        trajectory_seed: Seed of the realisations of the shared factors this layout observes. Two
            layouts of one control differ here, so they share the structure and no trajectory.
    """

    name: str = Field(min_length=1)
    channels: int = Field(ge=1)
    factors_per_channel: int = Field(ge=1)
    coupling: float = Field(ge=0.0, le=1.0)
    noise: float = Field(ge=0.0)
    missing: float = Field(ge=0.0, lt=1.0)
    cadence: int = Field(ge=1)
    synchronous: bool
    units: int = Field(ge=1)
    shortest_unit: int = Field(ge=1)
    longest_unit: int = Field(ge=1)
    time_step: float = Field(gt=0.0)
    gain_spread: float = Field(ge=0.0, lt=1.0)
    seed: int
    trajectory_seed: int

    @model_validator(mode="after")
    def _lengths_span_a_range(self) -> "SensorLayout":
        if self.shortest_unit > self.longest_unit:
            raise ValueError(
                f"shortest unit must not exceed the longest, got {self.shortest_unit} "
                f"and {self.longest_unit}"
            )
        return self

    @property
    def channel_names(self) -> tuple[str, ...]:
        """The timed channels, named by position; the vocabulary keys them by corpus as well."""
        return tuple(f"s{number:02d}" for number in range(1, self.channels + 1))

    @property
    def sampling_regime(self) -> SamplingRegime:
        """Regular only where every channel reports on every step of the grid together."""
        if self.cadence == 1 and self.synchronous:
            return SamplingRegime.REGULAR
        return SamplingRegime.IRREGULAR
