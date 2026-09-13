import numpy as np
from numpy.typing import NDArray

from emblema.shared.kernel.ordering import seeded_rank


class Draws:
    """The random numbers one part of a synthetic corpus is made of, addressed by what it is.

    A unit is generated on its own, without the units before it, so its numbers cannot come from
    a stream that has to be advanced to reach them. Each draw is addressed instead: a seed and
    the names of what it belongs to — a layout, a unit, a channel — become the state the numbers
    come from, through the same rule the rest of the project turns a seed into an order with, so
    that a seed becomes randomness in one place and not two.

    Only uniform doubles are taken from the bit generator; the normal draws are built from them
    here. The numbers therefore follow from the bit stream alone, rather than from whichever
    algorithm the array library currently uses for a distribution — and a corpus regenerated
    later is the same corpus.
    """

    def __init__(self, seed: int, *parts: object) -> None:
        """Address the draws by ``seed`` and the identity of what they belong to."""
        state = int.from_bytes(seeded_rank(seed, *parts))
        self._generator = np.random.Generator(np.random.Philox(state))

    def uniform(self, *shape: int) -> NDArray[np.float64]:
        """Draws from ``[0, 1)``."""
        return self._generator.random(shape)

    def normal(self, *shape: int) -> NDArray[np.float64]:
        """Draws from the standard normal, by the Box-Muller transform of uniform pairs."""
        count = int(np.prod(shape))
        first, second = self.uniform(2, (count + 1) // 2)
        # The transform takes a logarithm, which the open end of the draw has to supply: the
        # numbers are drawn from [0, 1), so it is the lower end that has to be moved.
        radius = np.sqrt(-2.0 * np.log(1.0 - first))
        angle = 2.0 * np.pi * second
        pairs = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=1)
        return pairs.reshape(-1)[:count].reshape(shape)
