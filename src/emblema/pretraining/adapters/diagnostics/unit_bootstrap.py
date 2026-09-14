from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from emblema.pretraining.domain.assessment.interval import CONFIDENCE, Interval
from emblema.pretraining.domain.assessment.kind_summary import KindSummary
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally
from emblema.pretraining.domain.mask_kind import MaskKind


@dataclass(frozen=True)
class UnitBootstrap:
    """Summarises tallies with intervals from a bootstrap over the groups they were tallied in.

    Both comparisons of a kind are ratios of sums over hidden tokens, and tokens of one unit are
    not independent — windows overlap and a unit is one realisation of the process — so their
    uncertainty is a bootstrap over validation units, the level at which they are independent. The
    masks are drawn once over the validation windows, so an interval holds the variation between
    units under that draw and not the variation between draws. Both intervals of a kind are read
    off the same resamples.

    Attributes:
        resamples: Bootstrap resamples per kind of mask.
        seed: Seed of the resampling, so that one run always gets the same intervals.
    """

    resamples: int = 2000
    seed: int = 1

    def summarise(self, tallies: Iterable[MaskKindTally]) -> tuple[KindSummary, ...]:
        """One summary per kind of mask and side of the channels apart, the channels apart last."""
        groups: dict[tuple[MaskKind, bool], list[MaskKindTally]] = {}
        for tally in tallies:
            groups.setdefault((tally.kind, tally.apart), []).append(tally)
        summaries = []
        for apart in (False, True):
            for kind in MaskKind:
                rows = sorted(groups.get((kind, apart), []), key=lambda tally: tally.group)
                if rows:
                    summaries.append(
                        _summary(kind, apart, rows, resamples=self.resamples, seed=self.seed)
                    )
        return tuple(summaries)


def _summary(
    kind: MaskKind, apart: bool, rows: Sequence[MaskKindTally], *, resamples: int, seed: int
) -> KindSummary:
    tokens = np.array([row.tokens for row in rows], dtype=np.float64)
    model = np.array([row.model for row in rows])
    picks = np.random.default_rng(seed).integers(0, len(rows), size=(resamples, len(rows)))
    drawn_tokens = tokens[picks].sum(axis=1)
    drawn_model = model[picks].sum(axis=1)
    tail = (1.0 - CONFIDENCE) / 2.0

    def interval(baseline: NDArray[np.float64]) -> Interval:
        excess = (baseline[picks].sum(axis=1) - drawn_model) / drawn_tokens
        low, high = np.quantile(excess, [tail, 1.0 - tail])
        return Interval(float(low), float(high))

    matched = np.array([row.matched for row in rows])
    linear = np.array([row.linear for row in rows])
    total = float(tokens.sum())
    floors = [row.floor for row in rows]
    return KindSummary(
        kind=kind,
        apart=apart,
        tokens=int(total),
        units=len(rows),
        model_error=float(model.sum()) / total,
        matched_error=float(matched.sum()) / total,
        linear_error=float(linear.sum()) / total,
        mean_error=sum(row.mean for row in rows) / total,
        noise_floor=None
        if any(floor is None for floor in floors)
        else sum(floor or 0.0 for floor in floors) / total,
        matched_excess=interval(matched),
        linear_excess=interval(linear),
    )
