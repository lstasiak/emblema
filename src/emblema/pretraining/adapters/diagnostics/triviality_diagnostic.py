from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Self

import torch
from torch import Tensor

from emblema.pretraining.adapters.objective.reconstruction_loss import ReconstructionLoss
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


@dataclass(frozen=True)
class TrivialityDiagnostic:
    """Does each kind of mask teach anything a trivial baseline does not already know?

    For every kind of mask the model's squared error is tallied beside the baseline the plan
    matches to it — linear interpolation within the channel for a block or a single token, the
    cross-channel regression for a channel hidden whole — beside the strongest linear baseline on
    the same inputs, and beside the channel mean. All of them are scored by the objective's own
    squared error over the same hidden tokens, so the comparison is between answers to one
    question. Batches are observed one at a time and the sums added per group of windows the
    caller names; ``tallies`` hands them over. What the sums mean is judged elsewhere, with the
    uncertainty only groups can give.

    Attributes:
        channels_apart: Channel identifiers whose tokens are tallied in rows of their own —
            constant channels, whose value every method recovers, and timeless ones, which no
            interpolation reaches.
        noise_variance: Per vocabulary entry, indexed by channel identifier with padding at zero,
            the variance of the measurement noise in normalised units; ``None`` where the corpus
            states no noise.
        sums: Running tallies per kind, side of ``channels_apart`` and group.
    """

    channels_apart: frozenset[int] = frozenset()
    noise_variance: tuple[float, ...] | None = None
    sums: Mapping[tuple[MaskKind, bool, str], MaskKindTally] = field(default_factory=dict)

    def observe(
        self,
        batch: TokenTensors,
        masks: TokenMasks,
        groups: Sequence[str],
        *,
        model: Tensor,
        interpolation: Tensor,
        ridge: Tensor,
        combined: Tensor,
    ) -> Self:
        """The diagnostic with ``batch`` tallied in, given every prediction over it.

        Args:
            batch: The windows.
            masks: What was hidden, the same for the model and every baseline.
            groups: The group each window of the batch belongs to, in row order.
            model: The model's prediction.
            interpolation: The interpolation baseline's.
            ridge: The cross-channel regression's.
            combined: The regression on the other channels and the channel's own line.

        Raises:
            ValueError: If ``groups`` does not name one group per window, or a channel of the
                batch has no noise variance.
        """
        if len(groups) != batch.batch_size:
            raise ValueError(
                f"groups must name one group per window: {len(groups)} for {batch.batch_size}"
            )
        ids = batch.channel_ids
        apart = torch.isin(
            ids, torch.tensor(sorted(self.channels_apart), dtype=torch.int64, device=ids.device)
        )
        floor = self._floor_of(ids)
        errors = {
            name: ReconstructionLoss.squared_error(prediction.to(torch.float64), batch)
            for name, prediction in (
                ("model", model),
                ("interpolation", interpolation),
                ("ridge", ridge),
                ("combined", combined),
                ("mean", torch.zeros_like(model, dtype=torch.float64)),
            )
        }
        sums = dict(self.sums)
        for kind in MaskKind:
            channel = kind is MaskKind.CHANNEL
            matched = errors["ridge" if channel else "interpolation"]
            linear = errors["ridge" if channel else "combined"]
            for side in (False, True):
                positions = (masks.of_kind(kind) & ~batch.padding_mask & (apart == side)).to(
                    torch.float64
                )
                tokens = positions.sum(dim=1).to(torch.int64).tolist()
                model_sums = _per_window(errors["model"], positions)
                mean_sums = _per_window(errors["mean"], positions)
                matched_sums = _per_window(matched, positions)
                linear_sums = _per_window(linear, positions)
                floor_sums = None if floor is None else _per_window(floor, positions)
                for row, group in enumerate(groups):
                    if tokens[row] == 0:
                        continue
                    added = MaskKindTally(
                        kind=kind,
                        apart=side,
                        group=group,
                        tokens=tokens[row],
                        model=model_sums[row],
                        matched=matched_sums[row],
                        linear=linear_sums[row],
                        mean=mean_sums[row],
                        floor=None if floor_sums is None else floor_sums[row],
                    )
                    key = (kind, side, group)
                    sums[key] = sums[key] + added if key in sums else added
        return replace(self, sums=sums)

    def tallies(self) -> tuple[MaskKindTally, ...]:
        """Every tally, the channels reported apart after the rest, then by kind and group."""
        kinds = list(MaskKind)
        return tuple(
            sorted(
                self.sums.values(),
                key=lambda tally: (tally.apart, kinds.index(tally.kind), tally.group),
            )
        )

    def _floor_of(self, ids: Tensor) -> Tensor | None:
        if self.noise_variance is None:
            return None
        if int(ids.max()) >= len(self.noise_variance):
            raise ValueError(
                f"noise variance covers {len(self.noise_variance)} entries, the batch reaches "
                f"channel {int(ids.max())}"
            )
        variance = torch.tensor(self.noise_variance, dtype=torch.float64, device=ids.device)
        return variance[ids]


def _per_window(values: Tensor, positions: Tensor) -> list[float]:
    """Per window, the sum of ``values`` over ``positions``, given as weights of zero or one."""
    summed: list[float] = (values * positions).sum(dim=1).tolist()
    return summed
