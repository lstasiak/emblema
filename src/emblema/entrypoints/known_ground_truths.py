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

    Where a corpus's files sit under the directory the process was given is part of the same
    fact. Each publisher ships its own shape and the archives are unpacked as they come, so the
    directory a reader binds to is nested differently for each of them; asking the process to be
    told that directory would be asking an operator to know one corpus's packaging. The marker
    below is what the turbofans are recognised by.

    Only the turbofans are here. The other corpora are published without answers this context
    knows how to read, and a task defined over one of them is refused by name rather than
    answered by whichever reader happened to be registered first.
    """

    CMAPSS = "cmapss"
    TURBOFAN_MARKER = "train_FD001.txt"

    @classmethod
    def under(cls, corpora: Path) -> CorpusGroundTruths:
        """Every corpus this process can answer for, reading from the raw corpora in ``corpora``."""
        return CorpusGroundTruths({cls.CMAPSS: CmapssGroundTruth(cls.turbofans_under(corpora))})

    @classmethod
    def turbofans_under(cls, corpora: Path) -> Path:
        """Where the run-to-failure records sit, wherever the archive put them.

        A corpus that was never fetched resolves to where it would have been, so what fails is
        the reading of a named file rather than the assembling of the process: a worker serving
        campaigns over another corpus has no business stopping because this one is absent.
        """
        root = corpora / cls.CMAPSS
        found = sorted(root.rglob(cls.TURBOFAN_MARKER)) if root.is_dir() else []
        return found[0].parent if found else root
