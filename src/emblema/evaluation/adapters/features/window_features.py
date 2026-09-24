"""What a classical candidate reads a window as, whichever scheme it was stated under.

A closed union rather than a protocol: the schemes are a closed enum, so the readings of a
window are a closed set too, and naming them here lets whoever holds one be typed without
inventing an interface that has exactly these implementations and no others.
"""

from emblema.evaluation.adapters.features.channel_aggregated_features import (
    ChannelAggregatedFeatures,
)
from emblema.evaluation.adapters.features.per_channel_features import PerChannelFeatures
from emblema.evaluation.adapters.features.spectral_features import SpectralFeatures

type WindowFeatures = PerChannelFeatures | SpectralFeatures | ChannelAggregatedFeatures
