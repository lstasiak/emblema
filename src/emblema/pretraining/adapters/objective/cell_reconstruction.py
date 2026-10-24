from typing import ClassVar

import torch
from torch import Tensor
from torch.nn import functional

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.cell_tensors import CellTensors
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.objective_score import RECONSTRUCTION, ObjectiveScore
from emblema.pretraining.adapters.objective.reconstruction_loss import elementwise, losses_dtype
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.domain.training.objective_loss import ObjectiveLoss
from emblema.pretraining.domain.training.observation_term import ObservationTerm
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.tensors.token_tensors import TokenTensors

# The names of the terms beside the hidden cells' own in the score, as ``RECONSTRUCTION`` names it.
PRESENCE, COUNT = "presence", "count"


class CellReconstruction(MaskedReconstruction):
    """Hide cells of a window, encode the visible ones, predict what the hidden cells held.

    A token is one channel over one span of the window. A hidden cell leaves the encoder's
    input as a hidden reading does, and the cells around it are laid out from the readings that
    stay visible, so nothing of it stands in its neighbours. The decoder puts a placeholder at
    every hidden position, labelled with the channel and time encodings, and answers seven
    things about it, in the order a cell's features stand: whether anything was measured there,
    as a logit; ``log(1 + count)``; and the mean, minimum, maximum, first and last of what was.
    The five summaries are scored under the run's reading of a miss, averaged over the five,
    where the cell held a reading, and that is the hidden cells' own term, named
    ``reconstruction``, which validation reads; the presence is scored at every hidden cell by
    the cross-entropy and the count where the cell held a reading, each weighed by the
    observation term, and logged beside. The count that divides the sums is the hidden cells,
    held or not, one divisor for the three terms; the summaries' mean per hidden cell is thus
    scaled by the share of hidden cells that held a reading, and their ratio to the trivial
    predictor is not.

    What a model that learnt nothing would cost: the channel's mean for the summaries, zero in
    normalised units; for the presence, the share of the channel's visible timed cells in the
    window that held a reading, the window's share where the channel has no visible cell, each
    smoothed by one imagined cell of either kind so that a channel seen in every visible cell of
    six does not wager certainty; for the count, the visible mean of the channel's
    ``log(1 + count)`` the same way, unsmoothed.

    Attributes:
        encoder: The backbone being pretrained; what remains when the objective is done.
        decoder: The part that is thrown away.
        term: What the presence and the count of a hidden cell weigh.
    """

    OUTPUTS: ClassVar[int] = (
        2 + BinnedTokens.VALUE_FEATURES.stop - BinnedTokens.VALUE_FEATURES.start
    )
    # A share without a model is kept off the ends, where its cross-entropy is infinite.
    _EDGE: ClassVar[float] = 1e-6

    def __init__(
        self,
        encoder: SetEncoder,
        *,
        decoder_layers: int,
        observation: ObservationTerm,
        dropout: float = 0.0,
    ) -> None:
        if not encoder.architecture.reads_cells:
            raise ValueError(
                "this objective hides cells, and the encoder reads one token per reading"
            )
        super().__init__(
            encoder, decoder_layers=decoder_layers, dropout=dropout, outputs=self.OUTPUTS
        )
        self.term = observation

    def encoded(self, batch: TokenTensors, padding_mask: Tensor) -> Tensor:
        """The encoder's states of every position, fed the visible layout, the hidden unseen."""
        cells = self._cells(batch)
        states: Tensor = self.encoder(
            cells.visible, cells.channel_ids, cells.timestamps, cells.timeless, padding_mask
        )
        return states

    def score(self, batch: TokenTensors, masks: TokenMasks, loss: ObjectiveLoss) -> ObjectiveScore:
        """The summaries' error at the hidden cells that held a reading, and the terms beside.

        Raises:
            ValueError: If the batch carries no visible layout beside the full one.
        """
        cells = self._cells(batch)
        prediction = self(cells, masks)
        dtype = losses_dtype(prediction)
        target = cells.features.to(dtype)
        hidden = masks.hidden & ~cells.padding_mask
        present = target[..., BinnedTokens.PRESENCE] > 0.5
        weight = (hidden & present).to(dtype)

        summaries = prediction[..., BinnedTokens.VALUE_FEATURES].to(dtype)
        answers = target[..., BinnedTokens.VALUE_FEATURES]
        error = (elementwise(loss, summaries, answers).mean(-1) * weight).sum()
        trivial = (elementwise(loss, torch.zeros_like(answers), answers).mean(-1) * weight).sum()
        tokens = hidden.sum()
        if not self.term.is_stated:
            return ObjectiveScore(error=error, trivial=trivial.detach(), tokens=tokens)

        terms = {RECONSTRUCTION: error.detach()}
        trivial_terms = {RECONSTRUCTION: trivial.detach()}
        total, trivial_total = error, trivial
        # The timed cells the encoder saw: what a guess without a model may read.
        visible = ~cells.padding_mask & ~masks.hidden & ~cells.timeless
        entries = int(cells.channel_ids.max().item()) + 1
        if self.term.asks_presence:
            logits = prediction[..., BinnedTokens.PRESENCE].to(dtype)
            was_there = present.to(dtype)
            cost = functional.binary_cross_entropy_with_logits(logits, was_there, reduction="none")
            guess = self._visible_mean(
                cells,
                cells.visible[..., BinnedTokens.PRESENCE],
                visible,
                0.5,
                entries,
                smoothing=1.0,
            )
            trivial_cost = functional.binary_cross_entropy(
                guess.to(dtype).clamp(self._EDGE, 1.0 - self._EDGE), was_there, reduction="none"
            )
            presence = (cost * hidden.to(dtype)).sum() * self.term.presence_weight
            trivial_presence = (trivial_cost * hidden.to(dtype)).sum() * self.term.presence_weight
            total, trivial_total = total + presence, trivial_total + trivial_presence
            terms[PRESENCE], trivial_terms[PRESENCE] = presence.detach(), trivial_presence.detach()
        if self.term.asks_count:
            counted = prediction[..., BinnedTokens.LOG_COUNT].to(dtype)
            answer = target[..., BinnedTokens.LOG_COUNT]
            guess = self._visible_mean(
                cells,
                cells.visible[..., BinnedTokens.LOG_COUNT],
                visible & (cells.visible[..., BinnedTokens.PRESENCE] > 0.5),
                0.0,
                entries,
            ).to(dtype)
            count = (elementwise(loss, counted, answer) * weight).sum() * self.term.count_weight
            trivial_count = (
                elementwise(loss, guess, answer) * weight
            ).sum() * self.term.count_weight
            total, trivial_total = total + count, trivial_total + trivial_count
            terms[COUNT], trivial_terms[COUNT] = count.detach(), trivial_count.detach()
        return ObjectiveScore(
            error=total,
            trivial=trivial_total.detach(),
            tokens=tokens,
            terms=terms,
            trivial_terms=trivial_terms,
        )

    @staticmethod
    def _cells(batch: TokenTensors) -> CellTensors:
        if not isinstance(batch, CellTensors):
            raise ValueError(
                "this objective scores cells laid out twice, and the batch carries one layout"
            )
        return batch

    @staticmethod
    def _visible_mean(
        cells: CellTensors,
        values: Tensor,
        over: Tensor,
        fallback: float,
        entries: int,
        *,
        smoothing: float = 0.0,
    ) -> Tensor:
        """Per position, the mean of ``values`` over the cells of its channel in ``over``.

        The window's mean over ``over`` where the channel has no such cell, and ``fallback``
        where the window has none: what a guess without a model knows of a hidden cell from the
        visible cells of the same channel, and failing that of the same window. ``entries`` is
        one past the largest channel identifier in the batch. A ``smoothing`` adds twice that
        many imagined cells at ``fallback`` to every mean, so a share read off a few cells is
        kept off the ends.
        """
        ids = cells.channel_ids
        rows = ids.shape[0]
        counted = over.to(values.dtype)
        prior = 2.0 * smoothing * fallback
        summed = torch.zeros(rows, entries, dtype=values.dtype, device=ids.device).scatter_add(
            1, ids, values * counted
        )
        counts = torch.zeros(rows, entries, dtype=values.dtype, device=ids.device).scatter_add(
            1, ids, counted
        )
        window_counts = counts.sum(dim=1, keepdim=True)
        window_mean = ((values * counted).sum(dim=1, keepdim=True) + prior) / (
            window_counts + 2.0 * smoothing
        ).clamp(min=1e-12)
        window_mean = torch.where(
            window_counts > 0, window_mean, torch.full_like(window_mean, fallback)
        )
        channel_mean = torch.where(
            counts > 0, (summed + prior) / (counts + 2.0 * smoothing).clamp(min=1e-12), window_mean
        )
        return channel_mean.gather(1, ids)
