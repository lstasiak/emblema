"""The budget file, read once, and the encoder shape each of its compute tiers describes.

Numbers live in ``corpus_budget.toml``; a report script that needs one asks here, so the file is
parsed in one place and a tier becomes an architecture in one place.
"""

import tomllib
from functools import cache
from pathlib import Path
from typing import Any

from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture

BUDGET = Path(__file__).resolve().parent / "corpus_budget.toml"


@cache
def budget() -> dict[str, Any]:
    with BUDGET.open("rb") as handle:
        return tomllib.load(handle)


def tier_named(name: str) -> dict[str, Any]:
    return next(tier for tier in budget()["tiers"] if tier["name"] == name)


def architecture_of(tier: dict[str, Any]) -> EncoderArchitecture:
    return EncoderArchitecture(
        width=tier["width"],
        heads=tier["heads"],
        layers=tier["layers"],
        feedforward_width=tier["feedforward_width"],
        time_frequencies=tier["time_frequencies"],
    )
