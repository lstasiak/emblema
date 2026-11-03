"""A detection task's two sides and a frozen encoder's states of them, as the campaign reads them.

Shared by the host reports that read the states of a detection campaign's encoders outside the
campaign: the task is defined over its published corpus as the worker defines it, its fitting
and scored sides are drawn by the same use case, the windows are laid out as the backbone read
them, and each encoder is built under the campaign's detector recipe and seeding, so a report's
states are the campaign's states. Imports torch, so a report imports this module inside the
step that measures and not at the top.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
from emblema.entrypoints.known_ground_truths import KnownGroundTruths
from emblema.entrypoints.restored_backbones import RestoredBackbones
from emblema.entrypoints.workers.known_detectors import KnownDetectors
from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.torch.frozen_embedding import FrozenEmbedding
from emblema.evaluation.application.use_cases.define_downstream_task import DefineDownstreamTask
from emblema.evaluation.application.use_cases.draw_detection_windows import (
    DrawDetectionWindows,
    DrawDetectionWindowsCommand,
)
from emblema.evaluation.application.use_cases.open_test_split import OpenTestSplit
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.score_reference import ScoreReference
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.arrays.laid_window import LaidWindow
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.adapters.system.clock import SystemClock
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore

Floats = NDArray[np.float64]

INITIALISATION = "initialisation"


def ref_of(key: str, checksum: str) -> ArtifactRef:
    return ArtifactRef(key, Checksum.parse(checksum))


def largest_reading(window: TokenWindow) -> float:
    """The magnitude of the window's largest timed reading."""
    values = np.abs(np.asarray(window.values, dtype=np.float64))
    return float(values[~np.asarray(window.timeless, dtype=bool)].max())


@dataclass(frozen=True)
class DetectionSides:
    """A detection task's fitting side and labelled scored side, laid out for the encoder.

    Attributes:
        task: The task's name.
        fitting: The fitting side's windows, as the block holds them.
        scored: The scored side's windows, in the order of the labels below.
        laid_fitting: The fitting windows laid out as cells.
        laid_scored: The scored windows laid out as cells.
        fitting_units: The unit of each fitting window.
        scored_units: The unit of each scored window.
        positions: Where the block holds each scored window.
        ends_at: When each scored window ends, in its unit's time.
        targets: Each scored window's event number, zero for a normal window.
        largest: Each scored window's largest reading.
        vocabulary: How many channels the published corpus names.
        layout: How windows are laid out: the backbone's bins.
    """

    task: str
    fitting: Sequence[TokenWindow]
    scored: Sequence[TokenWindow]
    laid_fitting: list[LaidWindow]
    laid_scored: list[LaidWindow]
    fitting_units: list[str]
    scored_units: list[str]
    positions: NDArray[np.int64]
    ends_at: Floats
    targets: Floats
    largest: Floats
    vocabulary: int
    layout: BinnedTokens


def detection_sides(
    task_name: str,
    manifest: ArtifactRef,
    *,
    store: ArtifactStore,
    layout: BinnedTokens,
    workspace: Path,
    corpora: Path,
) -> DetectionSides:
    """The task's two sides, drawn by the use case the campaign's worker draws them by."""
    known = KnownTasks.named(task_name)
    blocks = PublishedCorpusBlocks(store, workspace)
    corpus = BlockCorpusWindows(blocks)
    tasks = InMemoryDownstreamTaskRepository()
    ids = SequentialIdGenerator()
    task_id = DefineDownstreamTask(tasks, corpus, ids)(
        known.defined_over(manifest, corpus.describe(manifest))
    )
    drawn = DrawDetectionWindows(
        tasks,
        corpus,
        KnownGroundTruths.under(corpora),
        OpenTestSplit(tasks, ids, SystemClock(), InMemoryEventPublisher(InMemoryEventSubscriber())),
    )(DrawDetectionWindowsCommand(task=task_id))
    published = blocks.manifest_of(manifest)
    block = blocks.block_of(published)
    fitting = block.at([window.position for window in drawn.fitting])
    scored = block.at([labelled.window.position for labelled in drawn.scored])
    return DetectionSides(
        task=known.name,
        fitting=fitting,
        scored=scored,
        laid_fitting=layout.lay(fitting),
        laid_scored=layout.lay(scored),
        fitting_units=[str(window.unit) for window in drawn.fitting],
        scored_units=[str(labelled.window.unit) for labelled in drawn.scored],
        positions=np.array([labelled.window.position for labelled in drawn.scored]),
        ends_at=np.array([labelled.window.ends_at for labelled in drawn.scored]),
        targets=np.array([labelled.target for labelled in drawn.scored]),
        largest=np.array([largest_reading(window) for window in scored]),
        vocabulary=len(published.channels),
        layout=layout,
    )


def shared_shape(
    store: ArtifactStore, weights: Sequence[tuple[str, str, str]]
) -> RestoredBackbones:
    """The shape every backbone of ``weights`` (label, key, checksum) has, the first's.

    The initialisations and the layout are the first backbone's, as each campaign's are its
    own backbone's; that holds for every backbone only while they share one shape.

    Raises:
        SystemExit: If a backbone reads raw tokens, or has another shape than the first.
    """
    first = RestoredBackbones(store, ref_of(*weights[0][1:]))
    if not first.bins:
        raise SystemExit("the backbone reads raw tokens; this report lays windows out as cells")
    for label, key, checksum in weights[1:]:
        other = RestoredBackbones(store, ref_of(key, checksum))
        if (other.bins, other.width, other.layers) != (first.bins, first.width, first.layers):
            raise SystemExit(f"{label} has another shape than {weights[0][0]}")
    return first


@dataclass(frozen=True)
class EncoderToRead:
    """One encoder a report reads: pretrained weights, or the shape at its initialisation.

    Attributes:
        label: The backbone's label, or ``INITIALISATION``.
        seed: The detector's seed, which draws an initialisation's weights.
        weights: The pretrained weights; ``None`` for an initialisation.
        shape: The backbone whose shape the encoder has.
    """

    label: str
    seed: int
    weights: ArtifactRef | None
    shape: ArtifactRef

    @property
    def name(self) -> str:
        """The label, an initialisation by its seed too."""
        return f"{self.label}-{self.seed}" if self.weights is None else self.label


def encoders_to_read(
    weights: Sequence[tuple[str, str, str]], seeds: Sequence[int]
) -> list[EncoderToRead]:
    """The initialisations under ``seeds`` in the first backbone's shape, then every backbone."""
    shape = ref_of(*weights[0][1:])
    encoders = [EncoderToRead(INITIALISATION, seed, None, shape) for seed in seeds]
    for label, key, checksum in weights:
        encoders.append(EncoderToRead(label, 1, ref_of(key, checksum), ref_of(key, checksum)))
    return encoders


def detector_recipe(encoder: EncoderToRead, *, per_unit: bool) -> DetectorRecipe:
    """The campaign's recipe for the states detector over ``encoder``.

    Raises:
        SystemExit: If the detector's reading is not laid out as cells by the backbone, or its
            reference is not the one asked for.
    """
    wanted = KnownDetectors.UNTRAINED if encoder.weights is None else KnownDetectors.FROZEN
    arm = next(arm for arm in KnownDetectors.over(encoder.shape) if arm.ref == wanted)
    if per_unit:
        arm = arm.tuned("reference", "unit")
    recipe = arm.recipe(seed=encoder.seed)
    if recipe.reading is None or recipe.reading.encoder.bins is not None:
        raise SystemExit("the detector's reading is not the one this report lays out")
    if (recipe.reference is ScoreReference.OWN_UNIT) != per_unit:
        raise SystemExit("the reference is not the one asked for")
    return recipe


class FrozenStates:
    """Builds each encoder as the campaign's runtime does and reads its states on a device."""

    def __init__(
        self,
        store: ArtifactStore,
        sides: DetectionSides,
        *,
        device: str,
        batch_size: int,
    ) -> None:
        self._store = store
        self._sides = sides
        self._device = device
        self._batch_size = batch_size

    def embedding(self, encoder: EncoderToRead, recipe: DetectorRecipe) -> FrozenEmbedding:
        """The frozen embedding of ``encoder``, seeded as the runtime seeds it.

        Raises:
            ValueError: If the recipe reads no encoder.
        """
        if recipe.reading is None:
            raise ValueError("the recipe reads no encoder")
        torch.manual_seed(recipe.seed)
        embedding = FrozenEmbedding.under(
            recipe.reading,
            RestoredBackbones(self._store, encoder.shape),
            vocabulary_size=self._sides.vocabulary,
            layout=self._sides.layout,
        ).to(self._device)
        embedding.eval()
        return embedding

    def states(self, embedding: FrozenEmbedding, laid: Sequence[LaidWindow]) -> Floats:
        """One state per laid window, on the host in double precision, in the order given."""
        rows = []
        layout = self._sides.layout
        for start in range(0, len(laid), self._batch_size):
            batch = TokenTensors.of(layout.batch(laid[start : start + self._batch_size]))
            # Moved before it is widened: an accelerator without double precision hands back
            # zeros when asked to do both at once.
            rows.append(embedding.embed(batch.to(self._device)).to("cpu").to(torch.float64).numpy())
        return np.concatenate(rows)
