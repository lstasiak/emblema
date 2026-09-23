from pathlib import Path

from emblema.evaluation.adapters.readers.cmapss_ground_truth import CmapssGroundTruth
from emblema.evaluation.adapters.readers.corpus_ground_truths import CorpusGroundTruths


class KnownGroundTruths:
    """The corpora a process can read the answers of, and where each one reads them from.

    Which corpora those are is a fact about the deployment rather than about any adapter: the
    answers of one are counted from run-to-failure records on disk and of another read from a
    file published beside it, and a process that has neither can still run a campaign over a
    corpus whose answers it does have. A register rather than a table read from configuration,
    because a corpus named in an environment file could be pointed at the wrong reader and the
    task would be answered instead of refused.

    Only the turbofans are here. The other corpora are published without answers this context
    knows how to read, and a task defined over one of them is refused by name rather than
    answered by whichever reader happened to be registered first.
    """

    CMAPSS = "cmapss"

    @classmethod
    def under(cls, corpora: Path) -> CorpusGroundTruths:
        """Every corpus this process can answer for, reading from the raw corpora in ``corpora``."""
        return CorpusGroundTruths({cls.CMAPSS: CmapssGroundTruth(corpora)})
