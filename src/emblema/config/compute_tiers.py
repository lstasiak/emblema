import tomllib
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from emblema.shared.kernel.compute import ComputeTier

WindowChoice = Literal["default", "longest"]

TIERS_FILE = files("emblema.config").joinpath("compute_tiers.toml")


class ComputeTierProfile(BaseModel):
    """One compute tier: the hardware it assumes and the scale an experiment runs at there."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: ComputeTier
    device: str = Field(description="Hardware and precision the tier runs on")
    tflops: float = Field(gt=0, description="Sustained throughput assumed for the device, TFLOP/s")
    width: int = Field(gt=0, description="Model width")
    heads: int = Field(gt=0, description="Attention heads per block")
    layers: int = Field(gt=0, description="Encoder layers")
    feedforward_width: int = Field(
        gt=0, description="Hidden width of the feed-forward network in each block"
    )
    time_frequencies: int = Field(gt=0, description="Fourier frequencies of the time encoding")
    corpus_fraction: float = Field(
        gt=0,
        le=1,
        description="Share of each corpus's training units a run reads unless its experiment says",
    )
    window: WindowChoice = Field(description="Window variant of each corpus the tier trains on")


class ComputeTiers(BaseModel):
    """Every compute tier, each stated exactly once, in the order the file states them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tiers: tuple[ComputeTierProfile, ...]

    @model_validator(mode="after")
    def check_every_tier_once(self) -> Self:
        names = [tier.name for tier in self.tiers]
        if sorted(names) != sorted(ComputeTier):
            raise ValueError(
                f"every compute tier must be stated exactly once, got {[str(n) for n in names]}"
            )
        return self

    @classmethod
    def load(cls, source: Path | Traversable = TIERS_FILE) -> Self:
        """The tiers stated in ``source``, the file shipped with the package by default."""
        with source.open("rb") as handle:
            return cls.model_validate(tomllib.load(handle))

    def profile(self, tier: ComputeTier) -> ComputeTierProfile:
        return next(profile for profile in self.tiers if profile.name == tier)
