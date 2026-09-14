from dataclasses import dataclass


@dataclass(frozen=True)
class Curve:
    """The losses of a run, epoch by epoch."""

    training: tuple[float, ...]
    validation: tuple[float, ...]
    seconds: tuple[float, ...]
