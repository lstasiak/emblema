"""Masked-reconstruction runs built from numbers, for the tests that judge and store them.

No model is trained: the rules read numbers, and numbers are what these give them.
"""

from dataclasses import replace

import numpy as np

from emblema.pretraining.domain.assessment.curve import Curve
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally
from emblema.pretraining.domain.assessment.results import Results
from emblema.pretraining.domain.assessment.spectrum import Spectrum
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy

FLOOR = 0.01
UNITS = 20
STRATEGY = MaskingStrategy(channel_rate=0.15, block_rate=0.6, block_span=0.5, token_rate=0.1)
LEVELLED = (0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
SETTINGS = {"corpus": "control-a", "code": "abc", "seed": "1", "date": "2026-09-13 10:00:00"}


def tallies(
    kind: MaskKind,
    *,
    model: float,
    matched: float,
    linear: float | None = None,
    mean: float = 1.0,
    floor: float | None = FLOOR,
    units: int = UNITS,
    tokens: int = 100,
    spread: float = 0.0,
    apart: bool = False,
) -> list[MaskKindTally]:
    """Per-unit sums whose means are the given errors, the model's scattered from unit to unit."""
    offsets = np.linspace(-spread, spread, units)
    return [
        MaskKindTally(
            kind=kind,
            apart=apart,
            group=f"u/{index:02d}",
            tokens=tokens,
            model=float(model + offsets[index]) * tokens,
            matched=matched * tokens,
            linear=(matched if linear is None else linear) * tokens,
            mean=mean * tokens,
            floor=None if floor is None else floor * tokens,
        )
        for index in range(units)
    ]


def learnt_everywhere(*, mean: float = 1.0, floor: float | None = FLOOR) -> list[MaskKindTally]:
    return [
        *tallies(MaskKind.CHANNEL, model=0.1, matched=0.3, mean=mean, floor=floor),
        *tallies(MaskKind.BLOCK, model=0.05, matched=0.25, mean=mean, floor=floor),
        *tallies(MaskKind.TOKEN, model=0.02, matched=0.04, mean=mean, floor=floor),
    ]


def with_token(
    *, model: float, matched: float, linear: float | None = None, mean: float = 1.0
) -> list[MaskKindTally]:
    return [
        *tallies(MaskKind.CHANNEL, model=0.1, matched=0.3),
        *tallies(MaskKind.BLOCK, model=0.05, matched=0.25),
        *tallies(MaskKind.TOKEN, model=model, matched=matched, linear=linear, mean=mean),
    ]


def results(
    *,
    unit_tallies: list[MaskKindTally] | None = None,
    validation: tuple[float, ...] = LEVELLED,
    training: tuple[float, ...] | None = None,
    realised: float = 0.46,
    truth: tuple[float, ...] = (100.0, 20.0, 5.0),
    fitted: int = 50,
    skipped: int = 0,
    strategy: MaskingStrategy = STRATEGY,
) -> Results:
    return Results(
        settings=dict(SETTINGS),
        strategy=strategy,
        realised_ratio=realised,
        curve=Curve(
            training=training if training is not None else validation,
            validation=validation,
            seconds=(1.0,) * len(validation),
        ),
        tallies=tuple(unit_tallies if unit_tallies is not None else learnt_everywhere()),
        spectrum=Spectrum(
            truth=truth,
            model_residual=tuple(energy / 10 for energy in truth),
            ridge_residual=tuple(energy / 2 for energy in truth),
            fitted=fitted,
            skipped=skipped,
        ),
    )


def epochs_of(run: Results, epochs: int) -> Results:
    """The same run, its curve cut to its first ``epochs`` epochs."""
    curve = run.curve
    return replace(
        run,
        curve=Curve(curve.training[:epochs], curve.validation[:epochs], curve.seconds[:epochs]),
    )


def half_of(run: Results) -> Results:
    """The same configuration trained for half the epochs, and ending where this one ends."""
    half = epochs_of(run, run.epochs // 2)
    ending = (*half.curve.validation[:-1], run.curve.validation[-1])
    return replace(half, curve=replace(half.curve, validation=ending))
