from pathlib import Path

from emblema.catalog.adapters.readers.esa_ad_raw import ORIGIN_SECONDS, EsaAdRawCorpusReader
from emblema.evaluation.adapters.readers.cmapss_ground_truth import CmapssGroundTruth
from emblema.evaluation.adapters.readers.corpus_ground_truths import CorpusGroundTruths
from emblema.evaluation.adapters.readers.esa_ad_raw_ground_truth import EsaAdRawGroundTruth
from emblema.evaluation.adapters.readers.opssat_ad_ground_truth import OpssatAdGroundTruth
from emblema.evaluation.adapters.readers.physionet2012_ground_truth import (
    Physionet2012GroundTruth,
)
from emblema.evaluation.adapters.readers.physionet2019_ground_truth import (
    Physionet2019GroundTruth,
)
from emblema.evaluation.adapters.readers.skab_ground_truth import SkabGroundTruth
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

    The turbofans, the intensive-care stays of 2012 and 2019, the raw satellite telemetry, the
    testbed and the telemetry fragments are here. The other corpora are published without
    answers this context knows how to read, and a task defined over one of them is refused by
    name rather than answered by whichever reader happened to be registered first. Each set of
    stays answers one outcome, named here once for the reader and for the task that asks it;
    the sepsis stays answer it at the end of their first day, and the length of that day is
    named here too, since the reader refuses every other window. The telemetry's events are
    placed against windows whose length the task's publication fixed, so that length is named
    here once too, beside the channels the raw reader selects and the origin of its hours.
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
    ESA_AD_RAW = "esa_ad_raw"
    ESA_AD_RAW_DIRECTORY = "esa_ad"
    ESA_AD_RAW_MARKER = "ESA-Mission1"
    # The window the raw telemetry is published with for its tasks, in hours.
    ESA_AD_RAW_WINDOW_HOURS = 1.0
    SKAB = "skab"
    SKAB_MARKER = "anomaly-free"
    # The window the testbed's recordings are published with for their task, in seconds.
    SKAB_WINDOW_SECONDS = 100.0
    OPSSAT_AD = "opssat_ad"
    OPSSAT_AD_MARKER = "segments.csv"
    # The window the fragments are published with, the longest fragment's length in seconds;
    # the truth is the fragment's whole, so only the task checks it.
    OPSSAT_AD_WINDOW_SECONDS = 1340.0

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
                cls.ESA_AD_RAW: EsaAdRawGroundTruth(
                    cls._found_under(corpora, cls.ESA_AD_RAW_DIRECTORY, cls.ESA_AD_RAW_MARKER),
                    channels=EsaAdRawCorpusReader.POLICY.channels,
                    origin_seconds=ORIGIN_SECONDS,
                    window_hours=cls.ESA_AD_RAW_WINDOW_HOURS,
                ),
                cls.SKAB: SkabGroundTruth(
                    cls._found_under(corpora, cls.SKAB, cls.SKAB_MARKER),
                    window_seconds=cls.SKAB_WINDOW_SECONDS,
                ),
                cls.OPSSAT_AD: OpssatAdGroundTruth(
                    cls._found_under(corpora, cls.OPSSAT_AD, cls.OPSSAT_AD_MARKER)
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
