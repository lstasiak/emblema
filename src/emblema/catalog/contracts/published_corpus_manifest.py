from collections import Counter
from dataclasses import dataclass

from emblema.catalog.contracts.exceptions import InvalidPublishedCorpusManifestError
from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum


@dataclass(frozen=True, kw_only=True)
class PublishedCorpusManifest:
    """Published description of a tokenised corpus: what a run reads to decide whether it suits.

    A tokenised corpus travels as two artifacts, the block of windows and this manifest. The
    manifest pins the data by version and checksum, names the block, carries the vocabulary the
    tokens are indexed by and lists the units in the order the block indexes them, with the side
    of the split each is on. Nothing of the Catalog's interior is here; unit keys are text, and
    the one thing a consumer may assume of their shape is that a corpus which arrives in parts
    keys a unit as ``<part>/<name>``.

    Invariants: channels carry identifiers ``1..n`` in order; units are unique and none is also
    said to be empty; the sides of the split are sorted, disjoint and together name exactly the
    units of the corpus, indexed and empty; counts are not negative.

    Attributes:
        corpus: Name of the corpus, as the vocabulary knows it.
        corpus_version: Version of the corpus the windows were cut from.
        corpus_checksum: Checksum of the source data of that version.
        block: Reference to the artifact holding the windows.
        window_length: Extent of one window, in the corpus's time unit.
        window_stride: Distance between the starts of consecutive windows.
        channels: The vocabulary, in identifier order, each channel with its statistics.
        units: Unit keys in the order the block's unit column indexes them.
        empty_units: Units of the corpus that yielded no window; absent from the block.
        training_units: Units whose data fitted the statistics, sorted.
        validation_units: Units held out from fitting, sorted.
        split_seed: Seed the split was drawn with; ``None`` where its units were named or a part
            of the corpus held out.
        window_count: How many windows the block holds.
        token_count: How many tokens the block holds in all.
    """

    corpus: str
    corpus_version: CorpusVersionId
    corpus_checksum: Checksum
    block: ArtifactRef
    window_length: float
    window_stride: float
    channels: tuple[PublishedChannel, ...]
    units: tuple[str, ...]
    empty_units: tuple[str, ...]
    training_units: tuple[str, ...]
    validation_units: tuple[str, ...]
    split_seed: int | None
    window_count: int
    token_count: int

    def __post_init__(self) -> None:
        identifiers = [channel.channel_id for channel in self.channels]
        if identifiers != list(range(1, len(identifiers) + 1)):
            raise InvalidPublishedCorpusManifestError(
                "channels must carry identifiers 1..n in order"
            )
        duplicates = sorted(key for key, count in Counter(self.units).items() if count > 1)
        if duplicates:
            raise InvalidPublishedCorpusManifestError(f"duplicate units: {duplicates}")
        both = sorted(set(self.units) & set(self.empty_units))
        if both:
            raise InvalidPublishedCorpusManifestError(
                f"units both indexed by the block and said to be empty: {both}"
            )
        for label, side in (
            ("training", self.training_units),
            ("validation", self.validation_units),
        ):
            if list(side) != sorted(set(side)):
                raise InvalidPublishedCorpusManifestError(
                    f"{label} units must be sorted and unique"
                )
        shared = sorted(set(self.training_units) & set(self.validation_units))
        if shared:
            raise InvalidPublishedCorpusManifestError(f"units on both sides of the split: {shared}")
        if set(self.training_units) | set(self.validation_units) != set(self.units) | set(
            self.empty_units
        ):
            raise InvalidPublishedCorpusManifestError(
                "the sides of the split must together name exactly the units of the corpus"
            )
        if self.window_count < 0 or self.token_count < 0:
            raise InvalidPublishedCorpusManifestError("counts cannot be negative")
