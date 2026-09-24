from collections.abc import Mapping, Sequence

from emblema.evaluation.domain.exceptions import UnknownGroundTruthError
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.ports.ground_truth import GroundTruth


class CorpusGroundTruths:
    """Every corpus a process knows the truth of, behind one port, chosen by the corpus asked about.

    A process serves whatever tasks its campaigns name, and the answers of one corpus are read
    from files a second corpus knows nothing about. Holding the readers in a register keyed by
    corpus is what lets a single run draw labels from several of them, which a classical
    candidate fitted across corpora does by definition. A corpus nobody registered is refused
    rather than answered by whichever reader happened to be there.
    """

    def __init__(self, by_corpus: Mapping[str, GroundTruth]) -> None:
        self._by_corpus = dict(by_corpus)

    def truths_of(self, corpus: str, windows: Sequence[TaskWindow]) -> Mapping[TaskWindow, float]:
        if corpus not in self._by_corpus:
            raise UnknownGroundTruthError(
                f"this process knows no ground truth of corpus {corpus!r}; it knows "
                f"{sorted(self._by_corpus)}"
            )
        return self._by_corpus[corpus].truths_of(corpus, windows)
