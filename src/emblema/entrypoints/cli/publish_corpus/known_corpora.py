from dataclasses import dataclass
from typing import Self

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


@dataclass(frozen=True)
class KnownCorpus:
    """A corpus the process can publish: the facts about its source that no reader can state.

    Attributes:
        name: Name the corpus is registered under.
        source: Who publishes the data and where.
        licence: Terms the data was obtained under.
    """

    name: str
    source: CorpusSource
    licence: Licence


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
            *(KnownCorpus(name, GENERATED_SOURCE, GENERATED_LICENCE) for name in LAYOUTS),
        )

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    def named(self, name: str) -> KnownCorpus:
        return self._by_name[name]
