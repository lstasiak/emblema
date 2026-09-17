"""Derive the units a corpus should hold out, where a seeded draw would put them on one side.

A corpus whose units differ only in the draw is split by a seed. One whose units differ in kind is
not: the satellite corpus's held-out months carry excursions no training month repeats, and a
draw that puts them together leaves a side no model trained on the other reaches. ADR-0028 states
the rule for such a corpus, and this applies it to a published version's own values, as the
excursion report weighed them: rank the units by the mean square of their tokens, alternate the
ones past the line between the sides so that neither side holds them all, and draw the rest by
the seed until the held-out side is the share asked for.

The list it prints is the publication's, not this script's: it is derived once from one version's
block and stated to the next publication, which fits its statistics afresh and would therefore
weigh its units differently.

    uv run scripts/held_out_units.py data/report/saturation/excursions
    uv run python -m emblema.entrypoints.cli.publish_corpus --corpus esa_ad \
        --window 1 --stride 1 --hold-out $(uv run scripts/held_out_units.py <sides> --plain)
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.shared.kernel.ordering import seeded_rank
from scripts.excursion_report import WindowMass, read_masses
from scripts.reporting import table

# Where a unit stops being ordinary, in mean squares. The training side of a published corpus is
# normalised to one, so a unit above it is heavier than the data the statistics were fitted on.
LINE = 1.0
# The share of units the held-out side takes, as the publications of this project draw it.
FRACTION = 0.2
SEED = 1


@dataclass(frozen=True)
class WeighedUnit:
    """One unit of a corpus, weighed by what its tokens hold.

    Attributes:
        unit: The unit's key.
        tokens: Tokens it holds.
        mean_square: The channel-mean predictor's error over them, which is what a squared loss
            over this unit is decided by.
    """

    unit: str
    tokens: int
    mean_square: float


def weighed(masses: Sequence[WindowMass]) -> list[WeighedUnit]:
    """Every unit of both sides, heaviest first, ties broken by name so the order is fixed."""
    tokens: dict[str, int] = {}
    squares: dict[str, float] = {}
    for row in masses:
        tokens[row.unit] = tokens.get(row.unit, 0) + row.tokens
        squares[row.unit] = squares.get(row.unit, 0.0) + row.squares
    units = [WeighedUnit(unit, tokens[unit], squares[unit] / tokens[unit]) for unit in tokens]
    return sorted(units, key=lambda weighed: (-weighed.mean_square, weighed.unit))


def held_out(
    units: Sequence[WeighedUnit],
    *,
    line: float = LINE,
    fraction: float = FRACTION,
    seed: int = SEED,
) -> tuple[str, ...]:
    """The units to hold out: the heavy ones alternated, the rest drawn, sorted by name.

    The heaviest unit goes to the training side, so that a model has the kind of behaviour it
    will be asked about; the next goes to the held-out side, and so on down the line. Whatever
    the alternation leaves to find is drawn from the ordinary units by the seed, ranked as the
    Catalog ranks them, so the draw is the one a reader of this project would expect.

    Raises:
        ValueError: If the alternation alone already holds out more than the fraction asks for,
            which no draw can undo.
    """
    heavy = [unit for unit in units if unit.mean_square > line]
    alternated = [unit.unit for index, unit in enumerate(heavy) if index % 2 == 1]
    wanted = round(fraction * len(units))
    if len(alternated) > wanted:
        raise ValueError(
            f"alternating {len(heavy)} units past {line:g} holds out {len(alternated)}, "
            f"more than the {wanted} a share of {fraction:g} asks for"
        )
    ordinary = [unit.unit for unit in units if unit.mean_square <= line]
    drawn = sorted(ordinary, key=lambda unit: (seeded_rank(seed, unit), unit))
    return tuple(sorted([*alternated, *drawn[: wanted - len(alternated)]]))


def sides_of(units: Sequence[WeighedUnit], chosen: Sequence[str]) -> dict[str, list[WeighedUnit]]:
    held = set(chosen)
    return {
        "training": [unit for unit in units if unit.unit not in held],
        "validation": [unit for unit in units if unit.unit in held],
    }


def render(units: Sequence[WeighedUnit], chosen: Sequence[str], *, line: float) -> str:
    """What the rule did, as a note records it: the heavy units, then what each side holds."""
    held = set(chosen)
    heavy = table(
        ("Unit", "Tokens", "Mean square", "Side"),
        [
            (
                f"`{unit.unit}`",
                f"{unit.tokens:,}",
                f"{unit.mean_square:.3f}",
                "held out" if unit.unit in held else "training",
            )
            for unit in units
            if unit.mean_square > line
        ],
    )
    summary = table(
        ("Side", "Units", "Tokens", "Mean square", f"Units past {line:g}"),
        [
            (
                side,
                f"{len(members):,}",
                f"{sum(unit.tokens for unit in members):,}",
                f"{_mean_square(members):.3f}",
                f"{sum(1 for unit in members if unit.mean_square > line)}",
            )
            for side, members in sides_of(units, chosen).items()
        ],
    )
    return "\n\n".join(
        [
            f"Units past {line:g} mean squares, alternated between the sides, heaviest first:",
            heavy,
            "What each side then holds, weighed under the published version's own statistics:",
            summary,
            f"Held out ({len(chosen)} of {len(units)}):",
            " ".join(chosen),
        ]
    )


def _mean_square(units: Sequence[WeighedUnit]) -> float:
    tokens = sum(unit.tokens for unit in units)
    return sum(unit.mean_square * unit.tokens for unit in units) / tokens if tokens else 0.0


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "excursions", type=Path, help="directory the excursion report weighed the block into"
    )
    parser.add_argument(
        "--line",
        type=float,
        default=LINE,
        help="mean square past which a unit is not ordinary; the normalisation's own unit",
    )
    parser.add_argument(
        "--fraction", type=float, default=FRACTION, help="share of units the held-out side takes"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="seed the ordinary units are drawn by"
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="print the units alone, space separated, to pass to the publication",
    )
    arguments = parser.parse_args(argv)
    units = weighed(read_masses(arguments.excursions))
    if not units:
        parser.error(f"no weighed window in {arguments.excursions}")
    chosen = held_out(units, line=arguments.line, fraction=arguments.fraction, seed=arguments.seed)
    print(" ".join(chosen) if arguments.plain else render(units, chosen, line=arguments.line))


if __name__ == "__main__":
    main()
