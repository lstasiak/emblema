from collections.abc import Sequence
from pathlib import Path
from typing import overload

import numpy as np
from numpy.typing import NDArray

from emblema.catalog.contracts.exceptions import MalformedManifestError
from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.pretraining.domain.backbone.pretraining_input import PretrainingInput
from emblema.pretraining.domain.exceptions import UnreadablePublishedCorpusError
from emblema.pretraining.domain.training.corpus_share import CorpusShare
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.pretraining.domain.training.window_place import WindowPlace
from emblema.shared.adapters.windows.block_workspace import BlockWorkspace
from emblema.shared.adapters.windows.exceptions import MalformedBlockError
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore


class BlockTrainingCorpusReader:
    """Reads a published corpus out of the store: the manifest as JSON, the windows as a block.

    The manifest is decoded by the Catalog's published codec and the block by the shared one, so
    nothing of the Catalog's interior is touched. The block is fetched to the workspace and mapped
    from there, under its digest — the layout the Catalog's archive keeps its own blocks in — so a
    machine that published a corpus reads it back without fetching it again.
    """

    def __init__(self, store: ArtifactStore, workspace: Path) -> None:
        self._store = store
        self._blocks = BlockWorkspace(store, workspace)
        self._json = PublishedCorpusManifestJson()

    def describe(self, manifest: ArtifactRef) -> PretrainingInput:
        published = self._manifest(manifest)
        return PretrainingInput(
            corpus=published.corpus,
            corpus_version=published.corpus_version,
            corpus_checksum=published.corpus_checksum,
            manifest=manifest,
            block_checksum=published.block.checksum,
            vocabulary_size=len(published.channels),
        )

    def channels(self, manifest: ArtifactRef) -> tuple[str, ...]:
        return self._channels(self._manifest(manifest))

    def read(self, manifest: ArtifactRef, share: CorpusShare) -> TrainingCorpus:
        published = self._manifest(manifest)
        try:
            block = self._blocks.open(published.block)
        except MalformedBlockError as error:
            raise UnreadablePublishedCorpusError(
                f"block {published.block.key!r} is not one this reads: {error}"
            ) from error
        positions = {key: index for index, key in enumerate(published.units)}
        training = self._positions(positions, share.select(published.training_units))
        validation = self._positions(positions, published.validation_units)
        axes = tuple(PublishedCorpusManifest.axis_of(unit) for unit in published.units)
        return TrainingCorpus(
            name=published.corpus,
            checksum=published.block.checksum,
            training=block.of_units(training),
            validation=block.of_units(validation),
            channels=self._channels(published),
            training_places=_Places(axes, *block.places_of(training)),
            validation_places=_Places(axes, *block.places_of(validation)),
        )

    @classmethod
    def _channels(cls, published: PublishedCorpusManifest) -> tuple[str, ...]:
        return tuple(
            cls.channel_name(channel.corpus, channel.channel) for channel in published.channels
        )

    @staticmethod
    def channel_name(corpus: str, channel: str) -> str:
        """What a training corpus calls a published channel: its corpus and itself, joined.

        The name carries the corpus because a vocabulary chained through several publications
        may hold one channel name under two corpora, and the mixture compares names.
        """
        return f"{corpus}/{channel}"

    @staticmethod
    def _positions(positions: dict[str, int], units: tuple[str, ...]) -> set[int]:
        # A unit that yielded no window is absent from the block and from the manifest's unit
        # order alike; asking for its windows is asking for nothing, not an error.
        return {positions[key] for key in units if key in positions}

    def _manifest(self, ref: ArtifactRef) -> PublishedCorpusManifest:
        try:
            return self._json.decode(self._store.get(ref))
        except MalformedManifestError as error:
            raise UnreadablePublishedCorpusError(
                f"artifact {ref.key!r} is not a published manifest: {error}"
            ) from error


class _Places(Sequence[WindowPlace]):
    """Where the windows of one side lie, from the block's columns, a place built when asked."""

    def __init__(
        self,
        axes: tuple[str, ...],
        units: NDArray[np.int64],
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
    ) -> None:
        self._axes = axes
        self._units = units
        self._starts = starts
        self._ends = ends

    def __len__(self) -> int:
        return len(self._units)

    @overload
    def __getitem__(self, index: int) -> WindowPlace: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[WindowPlace]: ...

    def __getitem__(self, index: int | slice) -> WindowPlace | Sequence[WindowPlace]:
        if isinstance(index, slice):
            return _Places(self._axes, self._units[index], self._starts[index], self._ends[index])
        return WindowPlace(
            axis=self._axes[int(self._units[index])],
            start=float(self._starts[index]),
            end=float(self._ends[index]),
        )
