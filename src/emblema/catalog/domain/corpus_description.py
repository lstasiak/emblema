from dataclasses import dataclass

from emblema.catalog.domain.channel_schema import ChannelSchema
from emblema.catalog.domain.corpus_content import CorpusContent
from emblema.shared.kernel.sampling import SamplingRegime


@dataclass(frozen=True)
class CorpusDescription:
    """How the data of a corpus is described and what it holds, as established by one reading.

    Attributes:
        channel_schema: Channels the data carries.
        sampling_regime: How the data is spaced in time.
        content: Checksum and counts of the validated data.
    """

    channel_schema: ChannelSchema
    sampling_regime: SamplingRegime
    content: CorpusContent
