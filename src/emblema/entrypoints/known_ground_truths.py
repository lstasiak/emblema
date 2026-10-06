from pathlib import Path

from emblema.evaluation.adapters.readers.cmapss_ground_truth import CmapssGroundTruth
from emblema.evaluation.adapters.readers.corpus_ground_truths import CorpusGroundTruths
from emblema.evaluation.adapters.readers.physionet2012_ground_truth import (
    Physionet2012GroundTruth,
)
from emblema.evaluation.adapters.readers.physionet2019_ground_truth import (
    Physionet2019GroundTruth,
)
from emblema.evaluation.domain.labels.outcome_scheme import OutcomeScheme


class KnownGroundTruths:
    """The corpora a process can read the answers of, and where each one reads them from.

    Which corpora those are is a fact about the deployment rather than about any adapter: the
    answers of one are counted from run-to-failure records on disk and of another read from a
    file published beside it, and a process that has neither can still run a campaign over a
    corpus whose answers it does have. A register rather than a table read from configuration,
    because a corpus named in an environment file could be pointed at the wrong reader and the
    task would be answered instead of refused. Where each corpus's files sit under the directory
    the process was given is part of the same fact, so a reader is bound to the directory its
    marker file is found in rather than to one an operator would have to know.

    The turbofans and the intensive-care stays of 2012 and 2019 are here. The other corpora are
    published without answers this context knows how to read, and a task defined over one of
    them is refused by name rather than answered by whichever reader happened to be registered
    first. Each set of stays answers one outcome, named here once for the reader and for the task
    that asks it; the sepsis stays answer it at the end of their first day, and the length of that
    day is named here too, since the reader refuses every other window.
    """

    CMAPSS = "cmapss"
    TURBOFAN_MARKER = "train_FD001.txt"
    PHYSIONET = "physionet2012"
    OUTCOMES_MARKER = "Outcomes-a.txt"
    IN_HOSPITAL_DEATH = OutcomeScheme("In-hospital_death")
    PHYSIONET2019 = "physionet2019"
    STAYS_MARKER = "training_setA"
    SEPSIS = OutcomeScheme("SepsisLabel")
    FIRST_DAY_HOURS = 24.0

    @classmethod
    def under(cls, corpora: Path) -> CorpusGroundTruths:
        """Every corpus this process can answer for, reading from the raw corpora in ``corpora``."""
        return CorpusGroundTruths(
            {
                cls.CMAPSS: CmapssGroundTruth(cls.turbofans_under(corpora)),
                cls.PHYSIONET: Physionet2012GroundTruth(
                    cls._found_under(corpora, cls.PHYSIONET, cls.OUTCOMES_MARKER),
                    cls.IN_HOSPITAL_DEATH,
                ),
                cls.PHYSIONET2019: Physionet2019GroundTruth(
                    cls._found_under(corpora, cls.PHYSIONET2019, cls.STAYS_MARKER),
                    cls.SEPSIS,
                    cls.FIRST_DAY_HOURS,
                ),
            }
        )

    @classmethod
    def turbofans_under(cls, corpora: Path) -> Path:
        """Where the run-to-failure records sit, wherever the archive put them.

        A corpus that was never fetched resolves to where it would have been, so what fails is
        the reading of a named file rather than the assembling of the process: a worker serving
        campaigns over another corpus has no business stopping because this one is absent.
        """
        return cls._found_under(corpora, cls.CMAPSS, cls.TURBOFAN_MARKER)

    @staticmethod
    def _found_under(corpora: Path, corpus: str, marker: str) -> Path:
        """The directory under ``corpora/corpus`` holding ``marker``, or where it would be.

        The marker is a file the corpus ships, or the first of its sets where it ships one file per
        unit, since a set is what the reader is bound to the parent of.
        """
        root = corpora / corpus
        found = sorted(root.rglob(marker)) if root.is_dir() else []
        return found[0].parent if found else root
