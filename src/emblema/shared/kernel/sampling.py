from enum import StrEnum


class SamplingRegime(StrEnum):
    """How observations of a corpus are spaced in time.

    A regime is a property of the data, not a promise about spacing: nothing downstream may
    derive a time step from it.

    Attributes:
        REGULAR: Observations arrive on a nominally fixed cadence on every channel.
        IRREGULAR: Observations arrive at arbitrary instants, sparsely and unevenly across
            channels.
    """

    REGULAR = "regular"
    IRREGULAR = "irregular"
