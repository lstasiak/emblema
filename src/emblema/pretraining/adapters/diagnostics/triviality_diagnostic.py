from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Self

import torch
from torch import Tensor

from emblema.pretraining.adapters.diagnostics.mask_kind_verdict import MaskKindVerdict
from emblema.pretraining.adapters.objective.reconstruction_loss import ReconstructionLoss
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.shared.adapters.tensors.token_tensors import TokenTensors

# Squared error of the model, squared error of the baseline, tokens scored.
_Tally = tuple[float, float, int]


@dataclass(frozen=True)
class TrivialityDiagnostic:
    """Does each kind of mask teach anything a trivial baseline does not already know?

    For every kind of mask the model's squared error is tallied against the baseline matched to
    it: linear interpolation within the channel for a block or a single token, the cross-channel
    regression for a channel hidden whole. Both are scored by the objective's own loss over the
    same hidden tokens, so the comparison is between two answers to one question. Batches are
    observed one at a time and the tallies added; ``verdicts`` turns them into one row per kind.

    Attributes:
        channels_apart: Channel identifiers whose tokens are tallied in rows of their own —
            constant channels, whose value every method recovers, and timeless ones, which no
            interpolation reaches.
        tallies: Running sums per kind and per side of ``channels_apart``.
    """

    channels_apart: frozenset[int] = frozenset()
    tallies: Mapping[tuple[MaskKind, bool], _Tally] = field(default_factory=dict)

    def observe(
        self,
        batch: TokenTensors,
        masks: TokenMasks,
        model: Tensor,
        interpolation: Tensor,
        ridge: Tensor,
    ) -> Self:
        """The diagnostic with ``batch`` tallied in, given the three predictions over it."""
        apart = torch.isin(
            batch.channel_ids,
            torch.tensor(
                sorted(self.channels_apart), dtype=torch.int64, device=batch.channel_ids.device
            ),
        )
        tallies = dict(self.tallies)
        for kind in MaskKind:
            baseline = ridge if kind is MaskKind.CHANNEL else interpolation
            for side in (False, True):
                positions = masks.of_kind(kind) & ~batch.padding_mask & (apart == side)
                count = int(positions.sum())
                if count == 0:
                    continue
                so_far = tallies.get((kind, side), (0.0, 0.0, 0))
                tallies[(kind, side)] = (
                    so_far[0] + float(ReconstructionLoss.over(model, batch, positions)) * count,
                    so_far[1] + float(ReconstructionLoss.over(baseline, batch, positions)) * count,
                    so_far[2] + count,
                )
        return replace(self, tallies=tallies)

    def verdicts(self) -> tuple[MaskKindVerdict, ...]:
        """One row per kind of mask, the channels reported apart after the rest."""
        return tuple(
            MaskKindVerdict(
                kind=kind,
                apart=side,
                tokens=count,
                model_error=model / count,
                baseline_error=baseline / count,
            )
            for side in (False, True)
            for kind in MaskKind
            for (model, baseline, count) in (self.tallies.get((kind, side), (0.0, 0.0, 0)),)
            if count
        )
