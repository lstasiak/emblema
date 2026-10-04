from dataclasses import dataclass
from typing import Self

from emblema.catalog.adapters.readers.utsd import UtsdCorpusReader
from emblema.catalog.domain.registry.corpus_source import CorpusSource
from emblema.catalog.domain.registry.licence import Licence
from emblema.shared.adapters.synthetic.layouts import LAYOUTS

# A corpus generated here carries the repository's own terms and may be redistributed: the control
# can travel with a published model, which is what lets a reader repeat it.
GENERATED_SOURCE = CorpusSource("Emblema", "https://github.com/lstasiak/emblema")
GENERATED_LICENCE = Licence(
    "Apache-2.0",
    permits_derivatives=True,
    url="https://www.apache.org/licenses/LICENSE-2.0",
)
# The time-series collection is one source under one licence, published as a corpus per dataset:
# the datasets measure unrelated things, and the vocabulary keeps the channels of corpora apart.
UTSD_SOURCE = CorpusSource(
    "THUML, Tsinghua University (Liu et al.)", "https://huggingface.co/datasets/thuml/UTSD"
)
UTSD_LICENCE = Licence(
    "Apache-2.0",
    permits_derivatives=True,
    url="https://www.apache.org/licenses/LICENSE-2.0",
)


@dataclass(frozen=True)
class KnownCorpus:
    """A corpus the process can publish: the facts about its source that no reader can state.

    Attributes:
        name: Name the corpus is registered under.
        source: Who publishes the data and where.
        licence: Terms the data was obtained under.
        raw_directory: Where the fetched data its reader is bound to sits, relative to the
            directory of raw corpora; the corpus's own name unless its publisher nests it.
    """

    name: str
    source: CorpusSource
    licence: Licence
    raw_directory: str = ""

    def __post_init__(self) -> None:
        if not self.raw_directory:
            object.__setattr__(self, "raw_directory", self.name)


class KnownCorpora:
    """The corpora the process can publish, by name."""

    def __init__(self, *corpora: KnownCorpus) -> None:
        self._by_name = {corpus.name: corpus for corpus in corpora}

    @classmethod
    def default(cls) -> Self:
        return cls(
            # NASA publishes C-MAPSS with no licence text. Whether derivatives may be
            # redistributed is undocumented, and an undocumented permission is recorded as its
            # absence.
            KnownCorpus(
                "cmapss",
                CorpusSource(
                    "NASA Prognostics Center of Excellence", "https://data.nasa.gov/dataset/"
                ),
                Licence("US Government Work", permits_derivatives=False),
            ),
            # The benchmark repository is GPL-3.0 and the data files sit inside it, so a
            # derivative may be redistributed under the same copyleft terms.
            KnownCorpus(
                "skab",
                CorpusSource("Skoltech (Katser and Kozitsin)", "https://github.com/waico/SKAB"),
                Licence(
                    "GPL-3.0",
                    permits_derivatives=True,
                    url="https://www.gnu.org/licenses/gpl-3.0.html",
                ),
            ),
            # The repository is MIT-licensed, but the metrics inside it were collected from a
            # company that stated no terms of its own; an undocumented permission is recorded as
            # its absence.
            KnownCorpus(
                "smd",
                CorpusSource(
                    "NetManAIOps (Su et al.)", "https://github.com/NetManAIOps/OmniAnomaly"
                ),
                Licence("No licence stated", permits_derivatives=False),
            ),
            # The Zenodo record states CC BY 3.0 IGO for the data itself, so a derivative may
            # be redistributed with attribution.
            KnownCorpus(
                "esa_ad",
                CorpusSource(
                    "European Space Agency (De Canio, Kotowski, Haskamp)",
                    "https://zenodo.org/records/15237121",
                ),
                Licence(
                    "CC-BY-3.0-IGO",
                    permits_derivatives=True,
                    url="https://creativecommons.org/licenses/by/3.0/igo/",
                ),
            ),
            # PhysioNet publishes the challenge data under the Open Data Commons Attribution
            # licence, so a derivative may be redistributed with attribution.
            KnownCorpus(
                "physionet2012",
                CorpusSource(
                    "PhysioNet (Silva, Moody, Scott, Celi, Mark)",
                    "https://physionet.org/content/challenge-2012/1.0.0/",
                ),
                Licence(
                    "ODC-By-1.0",
                    permits_derivatives=True,
                    url="https://opendatacommons.org/licenses/by/1-0/",
                ),
            ),
            # PhysioNet publishes the 2019 challenge data under CC BY 4.0, so a derivative may be
            # redistributed with attribution.
            KnownCorpus(
                "physionet2019",
                CorpusSource(
                    "PhysioNet (Reyna, Josef, Jeter, Shashikumar, Westover, Nemati, Clifford, "
                    "Sharma)",
                    "https://physionet.org/content/challenge-2019/1.0.0/",
                ),
                Licence(
                    "CC-BY-4.0",
                    permits_derivatives=True,
                    url="https://creativecommons.org/licenses/by/4.0/",
                ),
                raw_directory="physionet2019/training",
            ),
            # Harvard Dataverse records a public domain dedication with disclaimer, so a
            # derivative may be redistributed.
            KnownCorpus(
                "tep",
                CorpusSource(
                    "Harvard Dataverse (Rieth, Amsel, Tran, Cook)",
                    "https://doi.org/10.7910/DVN/6C3JR1",
                ),
                Licence(
                    "Public Domain Dedication",
                    permits_derivatives=True,
                    url="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1",
                ),
            ),
            *(
                KnownCorpus(
                    dataset.corpus_name, UTSD_SOURCE, UTSD_LICENCE, raw_directory="utsd/UTSD-12G"
                )
                for dataset in UtsdCorpusReader.DATASETS
            ),
            *(KnownCorpus(name, GENERATED_SOURCE, GENERATED_LICENCE) for name in LAYOUTS),
        )

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    def named(self, name: str) -> KnownCorpus:
        return self._by_name[name]
