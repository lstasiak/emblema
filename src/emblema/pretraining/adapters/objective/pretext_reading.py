from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from emblema.pretraining.adapters.objective.cell_reconstruction import CellReconstruction
from emblema.pretraining.adapters.objective.token_masking import TokenMasking
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.adapters.training.cell_batches import CellBatches
from emblema.pretraining.adapters.training.exceptions import UnanswerablePretextError
from emblema.pretraining.adapters.training.joined_cell_batches import JoinedCellBatches
from emblema.pretraining.adapters.training.trained_model import TrainedModel
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.arrays.joined_windows import JoinedWindow
from emblema.shared.adapters.arrays.laid_window import LaidWindow
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.tokens import TokenWindow

# Seeds a generator takes: torch reads one as a signed 64-bit integer.
SEEDS = 2**63
# Cells a batch holds at most: attention over a window costs memory by the square of its cells,
# and a window of 48 hours over 48 channels took a gigabyte of the Mac's accelerator at this
# size, so a batch is as many windows as fit this budget, never a fixed count of them.
CELLS_PER_BATCH = 8192


def packed(cells: Sequence[int], budget: int) -> list[range]:
    """Consecutive runs of windows whose cells together stay within ``budget``, in order.

    A window larger than the budget is a run of its own: it is read, not refused.
    """
    runs: list[range] = []
    start, held = 0, 0
    for at, count in enumerate(cells):
        if at > start and held + count > budget:
            runs.append(range(start, at))
            start, held = at, 0
        held += count
    if start < len(cells):
        runs.append(range(start, len(cells)))
    return runs


@dataclass(frozen=True)
class ReadCells:
    """One draw of masks over one window's cells, and what the decoder answered under it.

    Every array stands cell for cell in the order the layout gives a window's cells, without
    padding.

    Attributes:
        full: ``[cells, features]`` — the cells laid out from every reading.
        visible: ``[cells, features]`` — the same cells laid out from the readings left visible.
        hidden: ``[cells]`` — the cells the draw hid.
        channel_ids: ``[cells]`` — the channel of every cell.
        timeless: ``[cells]`` — the cells of static features, which have no time.
        answers: ``[cells, 5]`` — the decoder's mean, minimum, maximum, first and last of every
            cell, in normalised units; ``None`` where the draw was asked for without them.
    """

    full: NDArray[np.float64]
    visible: NDArray[np.float64]
    hidden: NDArray[np.bool_]
    channel_ids: NDArray[np.int64]
    timeless: NDArray[np.bool_]
    answers: NDArray[np.float64] | None


class PretextReading:
    """A pretrained objective over cells asked what it would fill in, one window at a time.

    The masks are the run's own strategy, but each window's are drawn from its own generator over
    that window alone, so what a window is asked never depends on the windows read beside it or
    on how the batch is padded. The objective is read as stored, its decoder included, and only
    over the channels it learnt rows for. An objective trained over a history of windows is asked
    over windows joined with theirs, as its run joined them, and over nothing else. The windows
    are read in batches of as many as hold ``cells_per_batch`` cells together.
    """

    def __init__(
        self,
        model: CellReconstruction,
        masking: MaskingStrategy,
        *,
        vocabulary_size: int,
        device: str = "cpu",
        cells_per_batch: int = CELLS_PER_BATCH,
    ) -> None:
        self._model = model.to(device).eval()
        self._masking = TokenMasking(masking)
        architecture = model.encoder.architecture
        self._cells = CellBatches(BinnedTokens(architecture.bins))
        self._joined = JoinedCellBatches(architecture.bins) if architecture.history > 1 else None
        self._vocabulary_size = vocabulary_size
        self._device = device
        self._cells_per_batch = cells_per_batch

    @classmethod
    def of(cls, trained: TrainedModel, *, device: str = "cpu") -> "PretextReading":
        """The stored objective read back, with the masks it was trained under.

        Raises:
            UnanswerablePretextError: If the model does not read cells, or its bytes do not say
                what it hid.
            UnreadableTrainedModelError: If the weights do not fit the shape stored with them.
        """
        if trained.masking is None or trained.loss is None:
            raise UnanswerablePretextError(
                "the stored model does not say what it hid and how it scored a miss: "
                "it reads no cells, or was stored before they were kept"
            )
        model = trained.build()
        if not isinstance(model, CellReconstruction):
            raise UnanswerablePretextError("the stored objective does not fill in cells")
        return cls(model, trained.masking, vocabulary_size=trained.vocabulary_size, device=device)

    def drawn(
        self,
        windows: Sequence[TokenWindow | JoinedWindow],
        seeds: Sequence[int],
        *,
        answered: bool,
        tail: bool = False,
    ) -> list[ReadCells]:
        """One draw over each window, from a generator seeded by the seed beside it.

        A window read under several seeds is given once per seed. Without ``answered`` the
        decoder is not run, and the draws are those a run with it would have made. With ``tail``
        nothing is drawn: the last span of every channel is hidden and the rest left visible, the
        seeds unread.

        Raises:
            ValueError: If the windows and seeds are not paired, or a seed is not one a
                generator takes: an integer in ``[0, 2**63)``.
            UnanswerablePretextError: If a window holds a channel the objective has no row for,
                or is not joined with its history where the objective read one, or the reverse.
        """
        if len(windows) != len(seeds):
            raise ValueError(f"{len(seeds)} seeds were given for {len(windows)} windows")
        outside = [seed for seed in seeds if not 0 <= seed < SEEDS]
        if outside:
            raise ValueError(f"a seed lies outside [0, 2**63): {outside[0]}")
        joined = self._joined is not None
        if any(isinstance(window, JoinedWindow) is not joined for window in windows):
            raise UnanswerablePretextError(
                "the objective reads windows joined with their history"
                if joined
                else "the objective reads each published window alone"
            )
        read: list[ReadCells] = []
        # A window asked under several seeds is laid out once, not once a draw: its cells for
        # the batch and, alone, for its masks.
        alone: dict[int, tuple[LaidWindow | None, TokenTensors]] = {}
        for run in packed([self._cells_of(window) for window in windows], self._cells_per_batch):
            read.extend(
                self._read(
                    windows[run.start : run.stop],
                    seeds[run.start : run.stop],
                    alone,
                    answered=answered,
                    tail=tail,
                )
            )
        return read

    def _cells_of(self, window: TokenWindow | JoinedWindow) -> int:
        """How many cells the window is laid out as: a row per channel it holds, over the bins."""
        if isinstance(window, JoinedWindow):
            return int(window.cells.shape[0] * window.cells.shape[1])
        return len(set(window.channel_ids)) * self._cells.layout.bins

    def _read(
        self,
        windows: Sequence[TokenWindow | JoinedWindow],
        seeds: Sequence[int],
        alone: dict[int, tuple[LaidWindow | None, TokenTensors]],
        *,
        answered: bool,
        tail: bool,
    ) -> list[ReadCells]:
        for window in windows:
            held = window.rows if isinstance(window, JoinedWindow) else window.channel_ids
            beyond = [int(c) for c in held if c > self._vocabulary_size]
            if beyond:
                raise UnanswerablePretextError(
                    f"a window holds the channel {beyond[0]}, beyond the "
                    f"{self._vocabulary_size} the objective learnt rows for"
                )
        laid = [self._laid_alone(window, alone) for window in windows]
        batch, layout = self._batch(windows, [cells for cells, _ in laid])
        tokens = batch.padding_mask.shape[1]
        masks = self._stacked(
            [
                self._tail_alone(own, tokens) if tail else self._drawn_alone(own, seed, tokens)
                for (_, own), seed in zip(laid, seeds, strict=True)
            ]
        )
        cells = layout.scored(batch, masks)
        answers = None
        if answered:
            with torch.no_grad():
                prediction = self._model(cells.to(self._device), masks.to(self._device))
            # Off the accelerator first, then widened: both in one call reads zeros on MPS.
            answers = (
                prediction[..., BinnedTokens.VALUE_FEATURES].to("cpu").to(torch.float64).numpy()
            )
        # Widened and taken off torch once for the batch: a call per row cost more than the model.
        hidden = masks.hidden.numpy()
        full = cells.features.to(torch.float64).numpy()
        visible = cells.visible.to(torch.float64).numpy()
        channel_ids = cells.channel_ids.numpy().astype(np.int64)
        timeless = cells.timeless.numpy().astype(np.bool_)
        lengths = (~batch.padding_mask).sum(dim=1).tolist()
        return [
            ReadCells(
                full=full[row, :length].copy(),
                visible=visible[row, :length].copy(),
                hidden=hidden[row, :length].copy(),
                channel_ids=channel_ids[row, :length].copy(),
                timeless=timeless[row, :length].copy(),
                answers=None if answers is None else answers[row, :length].copy(),
            )
            for row, length in enumerate(lengths)
        ]

    def _laid_alone(
        self,
        window: TokenWindow | JoinedWindow,
        alone: dict[int, tuple[LaidWindow | None, TokenTensors]],
    ) -> tuple[LaidWindow | None, TokenTensors]:
        """The window's cells, and the batch of it alone its masks are drawn over.

        A joined window's cells are joined again with the batch, which costs less than keeping
        them; a published window's are laid once.
        """
        found = alone.get(id(window))
        if found is None:
            if isinstance(window, JoinedWindow):
                found = (None, self._joined_layout.collate([window]))
            else:
                (cells,) = self._cells.layout.lay([window])
                found = (cells, self._cells.collate_laid([window], [cells]))
            alone[id(window)] = found
        return found

    def _batch(
        self, windows: Sequence[TokenWindow | JoinedWindow], laid: Sequence[LaidWindow | None]
    ) -> tuple[TokenTensors, CellBatches | JoinedCellBatches]:
        """The batch of ``windows`` and the layout that scores it."""
        joined = [window for window in windows if isinstance(window, JoinedWindow)]
        if joined:
            return self._joined_layout.collate(joined), self._joined_layout
        tokens = [window for window in windows if isinstance(window, TokenWindow)]
        cells = [cells for cells in laid if cells is not None]
        return self._cells.collate_laid(tokens, cells), self._cells

    @property
    def _joined_layout(self) -> JoinedCellBatches:
        if self._joined is None:  # pragma: no cover - refused before any window is laid
            raise UnanswerablePretextError("the objective reads each published window alone")
        return self._joined

    def _drawn_alone(self, laid: TokenTensors, seed: int, tokens: int) -> TokenMasks:
        """The masks of one window drawn over it alone, padded to the batch's ``tokens``."""
        drawn = self._masking.draw(laid, torch.Generator().manual_seed(seed))
        own = drawn.channel.shape[1]

        def padded(mask: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(mask, (0, tokens - own), value=False)

        return TokenMasks(
            channel=padded(drawn.channel),
            block=padded(drawn.block),
            token=padded(drawn.token),
            horizon=padded(drawn.horizon),
        )

    def _tail_alone(self, laid: TokenTensors, tokens: int) -> TokenMasks:
        """The last span of every channel hidden, as the horizon a training draw would hide."""
        last = laid.timestamps > 1.0 - 1.0 / self._cells.layout.bins
        horizon = last & ~laid.timeless & ~laid.padding_mask
        padded = torch.nn.functional.pad(horizon, (0, tokens - horizon.shape[1]), value=False)
        none = torch.zeros_like(padded)
        return TokenMasks(channel=none, block=none, token=none, horizon=padded)

    @staticmethod
    def _stacked(masks: Sequence[TokenMasks]) -> TokenMasks:
        return TokenMasks(
            channel=torch.cat([m.channel for m in masks]),
            block=torch.cat([m.block for m in masks]),
            token=torch.cat([m.token for m in masks]),
            horizon=torch.cat([m.horizon for m in masks]),
        )
