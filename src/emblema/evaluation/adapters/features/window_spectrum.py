from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from emblema.shared.kernel.tokens import TokenWindow


@dataclass(frozen=True)
class WindowSpectrum:
    """How each channel of one window spreads its power over frequency, as a row per channel.

    The periodogram is Lomb-Scargle's, generalised to a floating mean (Zechmeister and Kürster,
    2009): at each frequency, the share of the channel's variance a sinusoid fitted to its
    readings explains. It is computed at the instants the readings were taken, so a channel
    observed at irregular times is read as it was observed rather than after resampling, which
    would invent readings between the real ones and move power towards low frequencies.

    Frequencies are whole cycles per window, because a token's time is a fraction of its
    window: one window's spectrum is comparable with another's of the same corpus, and a corpus
    with longer windows reads the same physical period at a higher count. The bands double in
    width, so each one is an octave and a slow drift and a fast oscillation weigh alike.

    A channel read ``n`` times in a window is read up to ``n / 2`` cycles and no further — the
    Nyquist frequency of evenly spaced readings, and the average one of uneven readings. Above
    it an evenly sampled channel holds only aliases of the frequencies below, which would count
    the same power twice; so an octave the channel cannot resolve holds no share of its power,
    and a sparsely read channel says so in its spectrum.

    A channel with fewer than three readings, or none that differ, has no spectrum and its row
    is ``NaN`` throughout, and one that resolves a single frequency has no entropy, for the
    reason a statistic it cannot support is absent rather than zero. A frequency the readings
    cannot tell apart from a constant — every instant at the same phase — explains nothing and
    holds no power.

    Attributes:
        channels: Identifiers of the channels the window holds a timed reading of, ascending.
        rows: One row per channel, one column per name in ``NAMES``.
    """

    channels: NDArray[np.int64]
    rows: NDArray[np.float64]

    CYCLES: ClassVar[NDArray[np.float64]] = np.arange(1.0, 33.0)
    BANDS: ClassVar[tuple[tuple[float, float], ...]] = (
        (1.0, 2.0),
        (2.0, 4.0),
        (4.0, 8.0),
        (8.0, 16.0),
        (16.0, 33.0),
    )
    NAMES: ClassVar[tuple[str, ...]] = (
        "band_1_2",
        "band_2_4",
        "band_4_8",
        "band_8_16",
        "band_16_32",
        "centroid",
        "entropy",
        "peak",
    )
    # Below this a determinant or a variance is rounding, not signal.
    _NEGLIGIBLE: ClassVar[float] = 1e-12

    @classmethod
    def of(cls, window: TokenWindow) -> Self:
        """The spectrum of every channel ``window`` holds a timed reading of."""
        timed = ~np.asarray(window.timeless, dtype=bool)
        ids = np.asarray(window.channel_ids, dtype=np.int64)[timed]
        order = np.argsort(ids, kind="stable")
        channels, starts, counts = np.unique(ids[order], return_index=True, return_counts=True)
        times = np.asarray(window.times, dtype=np.float64)[timed][order]
        values = np.asarray(window.values, dtype=np.float64)[timed][order]
        resolved = cls.CYCLES[None, :] <= counts[:, None] / 2.0
        power = np.where(resolved, cls._periodogram(times, values, starts, counts), 0.0)
        return cls(channels=channels, rows=cls._summarised(power, resolved))

    @classmethod
    def _periodogram(
        cls,
        times: NDArray[np.float64],
        values: NDArray[np.float64],
        starts: NDArray[np.int64],
        counts: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Normalised power per channel and frequency, ``NaN`` for a channel with no spectrum.

        Every sum is a mean over one channel's readings, taken for all channels at once by
        reducing over the runs the stable sort grouped them into.
        """
        phase = np.outer(times, 2.0 * np.pi * cls.CYCLES)
        cos, sin = np.cos(phase), np.sin(phase)
        held = counts[:, None].astype(np.float64)

        def mean(of: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.add.reduceat(of, starts, axis=0) / (held if of.ndim > 1 else counts)

        y = mean(values)[:, None]
        c, s = mean(cos), mean(sin)
        yy = mean(values * values)[:, None] - y * y
        yc = mean(values[:, None] * cos) - y * c
        ys = mean(values[:, None] * sin) - y * s
        cc = mean(cos * cos) - c * c
        ss = mean(sin * sin) - s * s
        cs = mean(cos * sin) - c * s
        determinant = cc * ss - cs * cs
        resolvable = determinant > cls._NEGLIGIBLE
        with np.errstate(invalid="ignore", divide="ignore"):
            power = np.where(
                resolvable,
                (ss * yc * yc + cc * ys * ys - 2.0 * cs * yc * ys) / (yy * determinant),
                0.0,
            )
        varies = (counts >= 3) & (yy[:, 0] > cls._NEGLIGIBLE)
        return np.where(varies[:, None], np.clip(power, 0.0, 1.0), np.nan)

    @classmethod
    def _summarised(
        cls, power: NDArray[np.float64], resolved: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """Share of power per octave, the centroid and entropy of the spread, and its peak.

        The entropy is taken over the frequencies the channel resolves, so a channel read
        sparsely is not called concentrated merely because it resolves few. A channel that holds
        no power anywhere has no spread to describe, so it joins the ones with no spectrum
        rather than dividing by nothing.
        """
        total = power.sum(axis=1, keepdims=True)
        absent = np.isnan(total[:, 0]) | (total[:, 0] <= cls._NEGLIGIBLE)
        share = power / np.where(absent[:, None], 1.0, total)
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.where(share > 0.0, share * np.log(share), 0.0)
        rows = np.column_stack(
            (
                *(
                    share[:, (low <= cls.CYCLES) & (high > cls.CYCLES)].sum(axis=1)
                    for low, high in cls.BANDS
                ),
                share @ cls.CYCLES,
                -logged.sum(axis=1) / np.log(np.maximum(resolved.sum(axis=1), 2)),
                power.max(axis=1),
            )
        )
        rows[absent] = np.nan
        # One resolved frequency is no spread at all, so there is no entropy to state.
        rows[resolved.sum(axis=1) < 2, cls.NAMES.index("entropy")] = np.nan
        return rows
