"""The corpus budget file, read once, for the reports that need its corpus facts.

Numbers live in ``corpus_budget.toml``; a report script that needs one asks here, so the file is
parsed in one place. The compute tiers are configuration and are read from ``emblema.config``.
"""

import tomllib
from functools import cache
from pathlib import Path
from typing import Any

BUDGET = Path(__file__).resolve().parent / "corpus_budget.toml"


@cache
def budget() -> dict[str, Any]:
    with BUDGET.open("rb") as handle:
        return tomllib.load(handle)


def vocabulary_size() -> int:
    """Channels of every measured corpus together: the table one backbone over the mix carries."""
    return sum(
        int(corpus["measured"]["channels"])
        for corpus in budget()["corpora"].values()
        if "measured" in corpus
    )
