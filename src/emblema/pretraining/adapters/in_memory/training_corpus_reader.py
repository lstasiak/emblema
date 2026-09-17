from collections.abc import Sequence
from dataclasses import dataclass, replace

from emblema.pretraining.domain.backbone.pretraining_input import PretrainingInput
from emblema.pretraining.domain.training.corpus_share import CorpusShare
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.exceptions import ArtifactNotFoundError


@dataclass(frozen=True)
class _Published:
    """A corpus as handed to the reader, with the units a share is picked from."""

    described: PretrainingInput
    corpus: TrainingCorpus
    units: tuple[str, ...]
    empty_units: tuple[str, ...]

    @property
    def training_units(self) -> tuple[str, ...]:
        """Every training unit, sorted as a manifest lists its sides."""
        return tuple(sorted({*self.units, *self.empty_units}))


class InMemoryTrainingCorpusReader:
    """Reader over corpora handed to it whole: the fake of the port for application tests.

    A corpus is published into it as the description and the windows a manifest would yield,
    under the reference a manifest would have; nothing is decoded, so a test that wants to know
    what a caller does with a corpus does not pay for a block. A share is honoured over the
    training units the corpus was said to have — those its windows were cut from and those that
    yielded none — as the reader of a block honours it over the manifest's.
    """

    def __init__(self) -> None:
        self._published: dict[ArtifactRef, _Published] = {}

    def publish(
        self,
        manifest: ArtifactRef,
        described: PretrainingInput,
        corpus: TrainingCorpus,
        units: Sequence[str] | None = None,
        *,
        empty_units: Sequence[str] = (),
    ) -> None:
        """Make ``corpus`` readable under ``manifest``, described as ``described``.

        Args:
            manifest: The reference the corpus is read under.
            described: What ``describe`` says about it.
            corpus: The windows ``read`` gives back.
            units: The unit each training window was cut from, one per window. Where none are
                named, every training window counts as a unit of its own, so that a share still
                reads a share.
            empty_units: Training units that yielded no window; a share may pick them too.

        Raises:
            ValueError: If the units named are not one per training window.
        """
        if units is None:
            units = tuple(str(index) for index in range(len(corpus.training)))
        elif len(units) != len(corpus.training):
            raise ValueError(
                f"{len(units)} units named for {len(corpus.training)} training windows"
            )
        self._published[manifest] = _Published(described, corpus, tuple(units), tuple(empty_units))

    def describe(self, manifest: ArtifactRef) -> PretrainingInput:
        return self._found(manifest).described

    def read(self, manifest: ArtifactRef, share: CorpusShare) -> TrainingCorpus:
        published = self._found(manifest)
        if share.is_whole:
            return published.corpus
        chosen = set(share.select(published.training_units))
        return replace(
            published.corpus,
            training=[
                window
                for window, unit in zip(published.corpus.training, published.units, strict=True)
                if unit in chosen
            ],
        )

    def _found(self, manifest: ArtifactRef) -> _Published:
        try:
            return self._published[manifest]
        except KeyError:
            raise ArtifactNotFoundError(manifest.key) from None
