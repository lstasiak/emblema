from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class MiniRocketTransform:
    """MiniRocket's convolutions as one fit left them: which ones, where, against what bias.

    Dempster, Schmidt and Webb (2021). Every kernel is nine weights, three of them 2 and six of
    them -1, so the 84 ways of choosing the three are the whole family; each kernel is applied
    at dilations spread exponentially up to the length of the series, to a random combination of
    the channels, and every feature is the share of positions where the convolution exceeds a
    bias. The biases are the only thing learnt from data — quantiles of the convolution of one
    labelled series — so the transform is cheap to fit and has almost no variance to spend.

    The combination of channels is convolved as their sum. A convolution is linear, so that is
    the reference implementation's arithmetic reordered, and it lets one series per combination
    stand for all its channels and every window be convolved at once.

    Half the features read the whole convolution, zero-padded at both ends, and half only the
    positions where the kernel lies wholly inside the series, alternating from one kernel to the
    next — the reference's way of letting the edges count without letting them dominate.

    Attributes:
        length: Steps of every series this transforms.
        dilations: The dilations, ascending.
        per_dilation: How many features each kernel yields at each dilation.
        combination_sizes: How many channels each (dilation, kernel) pairing sums, in order.
        channels: The channels of every pairing, one run per pairing, in order.
        biases: One per feature, in feature order.
    """

    length: int
    dilations: NDArray[np.int64]
    per_dilation: NDArray[np.int64]
    combination_sizes: NDArray[np.int64]
    channels: NDArray[np.int64]
    biases: NDArray[np.float64]

    KERNELS: ClassVar[NDArray[np.int64]] = np.array(list(combinations(range(9), 3)))
    MAX_DILATIONS: ClassVar[int] = 32
    # The reference draws quantiles off the golden ratio, a low-discrepancy sequence that
    # spreads a kernel's biases over its distribution without repeating one.
    GOLDEN: ClassVar[float] = (np.sqrt(5.0) + 1.0) / 2.0

    @property
    def width(self) -> int:
        return len(self.biases)

    @classmethod
    def fitted(cls, series: NDArray[np.float64], features: int, rng: np.random.Generator) -> Self:
        """The transform fitted to ``series`` of shape ``[examples, channels, steps]``.

        ``features`` is rounded down to a whole multiple of the 84 kernels.

        Raises:
            ValueError: If a series is shorter than a kernel or there are fewer features than
                kernels.
        """
        examples, channel_count, length = series.shape
        kernels = len(cls.KERNELS)
        if length < 9 or features < kernels:
            raise ValueError(f"MiniRocket needs 9 steps and 84 features, got {length}, {features}")
        dilations, per_dilation = cls._dilations(length, features // kernels)
        pairings = kernels * len(dilations)
        widest = np.log2(min(channel_count, 9) + 1)
        sizes = (2.0 ** rng.uniform(0.0, widest, pairings)).astype(np.int64)
        channels = np.concatenate(
            [rng.choice(channel_count, size, replace=False) for size in sizes]
        ).astype(np.int64)
        quantiles = (np.arange(1, kernels * int(per_dilation.sum()) + 1) * cls.GOLDEN) % 1.0
        unfitted = cls(length, dilations, per_dilation, sizes, channels, np.zeros(0))
        biases = [
            np.quantile(convolved[0], quantiles[start : start + count])
            for convolved, start, count, _ in unfitted._convolutions(
                series[rng.integers(examples, size=pairings)], per_pairing=True
            )
        ]
        return cls(length, dilations, per_dilation, sizes, channels, np.concatenate(biases))

    def of(self, series: NDArray[np.float64]) -> NDArray[np.float64]:
        """One row of features per series of ``series``, shape ``[examples, channels, steps]``.

        Raises:
            ValueError: If the series are not as long as those the transform was fitted to.
        """
        if series.shape[2] != self.length:
            raise ValueError(f"series of {series.shape[2]} steps, fitted to {self.length}")
        rows = np.empty((series.shape[0], self.width), dtype=np.float64)
        for convolved, start, count, inner in self._convolutions(series, per_pairing=False):
            read = convolved[:, inner]
            biases = self.biases[start : start + count]
            rows[:, start : start + count] = (read[:, :, None] > biases).mean(axis=1)
        return rows

    @staticmethod
    def _dilations(length: int, per_kernel: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Dilations spread exponentially up to the series, and the features each one yields."""
        most = min(per_kernel, MiniRocketTransform.MAX_DILATIONS)
        exponent = np.log2((length - 1) / 8)
        dilations, counts = np.unique(
            np.logspace(0.0, exponent, most, base=2.0).astype(np.int64), return_counts=True
        )
        per_dilation = (counts * (per_kernel / most)).astype(np.int64)
        for index in range(per_kernel - int(per_dilation.sum())):
            per_dilation[index % len(per_dilation)] += 1
        return dilations, per_dilation

    def _convolutions(
        self, series: NDArray[np.float64], *, per_pairing: bool
    ) -> Iterator[tuple[NDArray[np.float64], int, int, slice]]:
        """Every (dilation, kernel) pairing's convolution, where its features start, how many.

        With ``per_pairing`` the i-th series of ``series`` is the one the i-th pairing is
        convolved over (the one example each bias is drawn from); otherwise every pairing is
        convolved over every series. One pairing at a time, because all of them at once over
        every window would hold hundreds of copies of the input.
        """
        feature, combination, channel = 0, 0, 0
        for dilation_index, (dilation, count) in enumerate(
            zip(self.dilations.tolist(), self.per_dilation.tolist(), strict=True)
        ):
            padding = 4 * dilation
            for kernel_index, heavy in enumerate(self.KERNELS):
                size = int(self.combination_sizes[combination])
                chosen = self.channels[channel : channel + size]
                over = series[combination : combination + 1] if per_pairing else series
                summed = over[:, chosen, :].sum(axis=1)
                edges = (dilation_index + kernel_index) % 2 == 0
                inner = slice(None) if edges else slice(padding, self.length - padding)
                yield self._convolved(summed, dilation, heavy), feature, count, inner
                feature += count
                combination += 1
                channel += size

    @staticmethod
    def _convolved(
        summed: NDArray[np.float64], dilation: int, heavy: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """One kernel at one dilation over ``summed``, zero-padded to the input's length.

        Every weight is -1 and the three heavy ones 2, so the output is minus the sum of the
        nine shifted series plus three times the three heavy ones.
        """
        length = summed.shape[1]
        shifted = np.zeros((9, *summed.shape), dtype=np.float64)
        for tap in range(9):
            offset = (tap - 4) * dilation
            if offset >= 0:
                shifted[tap, :, : length - offset] = summed[:, offset:]
            else:
                shifted[tap, :, -offset:] = summed[:, : length + offset]
        return np.asarray(-shifted.sum(axis=0) + 3.0 * shifted[heavy].sum(axis=0))
