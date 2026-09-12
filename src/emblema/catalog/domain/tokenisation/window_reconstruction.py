from dataclasses import dataclass

from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature


@dataclass(frozen=True)
class WindowReconstruction:
    """What a token window says about the data it was cut from, back in the corpus's own units.

    Tokenising is read backwards here, so that what the representation keeps can be compared with
    what the reader delivered. What comes back is everything the window holds and nothing else:
    observations that fell outside every window were never tokenised and are in no reconstruction,
    and the window carries no trace of the unit it came from.

    Attributes:
        observations: Timed tokens as observations, in non-decreasing time order.
        static_features: Timeless tokens as static features, one per channel of the unit.
    """

    observations: tuple[Observation, ...]
    static_features: tuple[StaticFeature, ...]
