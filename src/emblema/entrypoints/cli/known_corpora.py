from dataclasses import dataclass
from typing import Self

from emblema.catalog.adapters.synthetic.layouts import LAYOUTS
from emblema.catalog.domain.registry.corpus_source import CorpusSource
from emblema.catalog.domain.registry.licence import Licence


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
        # The control corpora are generated here, so they carry the repository's own terms and
        # may be redistributed: the control can travel with a published model, which is what lets
        # a reader repeat it.
        ours = CorpusSource("Emblema", "https://github.com/lstasiak/emblema")
        apache = Licence(
            "Apache-2.0",
            permits_derivatives=True,
            url="https://www.apache.org/licenses/LICENSE-2.0",
        )
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
            *(KnownCorpus(name, ours, apache) for name in LAYOUTS),
        )

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    def named(self, name: str) -> KnownCorpus:
        return self._by_name[name]
