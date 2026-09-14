"""Show that the positive control carries the structure it claims, before anything uses it.

A permutation test: every channel of every unit is fitted on its own unit's latent factors and on
another unit's. The second fit is not zero, since all units share the process's frequencies, so the
evidence of shared structure is the excess of the first over the second. The coupled corpora must
clear the excess threshold and the null ones must not, and the report says which happened.

    uv sync --all-extras
    uv run scripts/synthetic_control_report.py

Prints markdown for the verification note and writes the figures beside it. Why the control exists
is stated in ``emblema.catalog.adapters.synthetic.layouts``.
"""

import argparse
import io
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

# A report writes files and never opens a window; the backend has to be chosen before pyplot is.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emblema.catalog.adapters.synthetic.latent_factor_process import LatentFactorProcess
from emblema.catalog.adapters.synthetic.layouts import CONTROL_PROCESS, LAYOUTS
from emblema.catalog.adapters.synthetic.sensor_layout import SensorLayout
from emblema.catalog.adapters.synthetic.synthetic_corpus_reader import SyntheticCorpusReader
from emblema.catalog.domain.identifiers import UnitKey
from scripts.reporting import table

FIGURES = REPO_ROOT / "docs" / "verification" / "figures"

# What a coupled corpus must clear and a null corpus must stay under. Between them is a gap wide
# enough that a run landing inside it is a result to look at rather than a threshold to argue
# about. Set from the design of the control, not from a measurement: a channel that follows the
# shared factors explains most of itself with them, a channel that does not explains none.
COUPLED_EXCESS = 0.4
NULL_EXCESS = 0.1

# A fit needs more observations than it has parameters to mean anything; this many times more.
OBSERVATIONS_PER_PARAMETER = 4


@dataclass(frozen=True)
class LayoutRecovery:
    """How much of a corpus's channels its own latent factors explain, over a matched shuffle."""

    layout: str
    coupling: float
    fits: int
    skipped: int
    own: float
    shuffled: float
    weakest_own: float

    @property
    def excess(self) -> float:
        return self.own - self.shuffled

    @property
    def holds(self) -> bool:
        """Whether the corpus came out as its coupling says it should."""
        if self.coupling > 0.0:
            return self.excess >= COUPLED_EXCESS
        return abs(self.excess) <= NULL_EXCESS


@dataclass(frozen=True)
class Report:
    """What one run of the script established."""

    process: LatentFactorProcess
    units: int
    recoveries: Sequence[LayoutRecovery]
    figures: Sequence[Path]


def explained(design: NDArray[np.float64], values: NDArray[np.float64]) -> float:
    """Share of the variance of ``values`` that a least-squares fit on ``design`` accounts for.

    Zero where the design says nothing about the values, one where it says everything. A channel
    that never varied has no variance to explain and is reported as nothing explained.
    """
    spread = float(np.square(values - values.mean()).sum())
    if spread == 0.0:
        return 0.0
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficients
    return 1.0 - float(np.square(residual).sum()) / spread


def recover(process: LatentFactorProcess, layout: SensorLayout) -> LayoutRecovery:
    """Fit every channel of every unit of ``layout`` on its own factors and on another unit's."""
    if layout.units < 2:
        # With one unit the shuffled baseline is the fit itself, so every corpus reads as an
        # excess of nothing and a coupled one is condemned by arithmetic rather than measured.
        raise SystemExit("the shuffled baseline needs a second unit; ask for --units 2 or more")
    reader = SyntheticCorpusReader(process, layout)
    keys = [unit.key.value for unit in reader.read_units()]
    enough = OBSERVATIONS_PER_PARAMETER * (process.factors + 1)
    own: list[float] = []
    shuffled: list[float] = []
    skipped = 0
    for position, key in enumerate(keys):
        other = (position + 1) % len(keys)
        for times, values in by_channel(reader, key).values():
            if len(times) < enough:
                skipped += 1
                continue
            own.append(explained(design_on(process, layout, times, position), values))
            shuffled.append(explained(design_on(process, layout, times, other), values))
    if not own:
        # Every channel was too short to fit, so there is no measurement to report; averaging
        # nothing would hand the verdict a nan and read as a corpus that failed.
        raise SystemExit(
            f"no channel of {layout.name} reached {enough} observations; ask for longer units"
        )
    return LayoutRecovery(
        layout=layout.name,
        coupling=layout.coupling,
        fits=len(own),
        skipped=skipped,
        own=float(np.mean(own)),
        shuffled=float(np.mean(shuffled)),
        weakest_own=float(np.min(own)),
    )


def design_on(
    process: LatentFactorProcess, layout: SensorLayout, times: NDArray[np.float64], unit: int
) -> NDArray[np.float64]:
    """The factors of the unit at ``unit``, at ``times``, with a column of ones for the offset."""
    factors = process.values_at(times, trajectory_seed=layout.trajectory_seed, unit=unit)
    return np.column_stack([factors, np.ones(len(times))])


def by_channel(
    reader: SyntheticCorpusReader, key: str
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """One unit's observations gathered per channel, as instants and values."""
    gathered: dict[str, list[tuple[float, float]]] = {}
    for observation in reader.read_observations(UnitKey(key)):
        gathered.setdefault(observation.channel, []).append((observation.time, observation.value))
    return {
        channel: (
            np.array([time for time, _ in pairs]),
            np.array([value for _, value in pairs]),
        )
        for channel, pairs in gathered.items()
    }


def draw(process: LatentFactorProcess, layout: SensorLayout, directory: Path) -> Path:
    """The hidden factors of one unit above the channels that watch them, for a person to judge."""
    reader = SyntheticCorpusReader(process, layout)
    unit = next(reader.read_units())
    channels = by_channel(reader, unit.key.value)
    dense = np.linspace(unit.extent.start, unit.extent.end, 600)
    factors = process.values_at(dense, trajectory_seed=layout.trajectory_seed, unit=0)

    figure, (above, below) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for number in range(process.factors):
        above.plot(dense, factors[:, number], linewidth=1.0, label=f"factor {number + 1}")
    above.set_title(f"{layout.name}: hidden factors of {unit.key}, coupling {layout.coupling:g}")
    above.legend(fontsize="x-small", ncol=process.factors)
    for channel, (times, values) in sorted(channels.items()):
        below.plot(times, values, marker=".", markersize=3, linewidth=0.6, label=channel)
    below.set_title(f"what its {layout.channels} sensors reported")
    below.set_xlabel("time")
    below.legend(fontsize="x-small", ncol=min(layout.channels, 8))
    figure.tight_layout()

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"synthetic-control-{layout.name}.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def measure(arguments: argparse.Namespace) -> Report:
    """Every corpus asked for, cut to the units asked for, measured and optionally drawn."""
    directory = Path(arguments.out)
    # A report reads a sample of the corpus, not all of it; a unit is the same unit either way,
    # because a corpus is generated per key rather than in a run.
    chosen = [LAYOUTS[name].with_dials(units=arguments.units) for name in arguments.layout]
    drawn = (
        [draw(CONTROL_PROCESS, layout, directory) for layout in chosen] if arguments.figures else []
    )
    return Report(
        process=CONTROL_PROCESS,
        units=arguments.units,
        recoveries=[recover(CONTROL_PROCESS, layout) for layout in chosen],
        figures=drawn,
    )


def shown(path: Path) -> str:
    """A path as the note carries it: relative to the repository when it lies inside one."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def heading(report: Report) -> str:
    process = report.process
    periods = 1.0 / process.frequencies()
    rows = [
        ("Machine", f"{platform.platform()}, {platform.processor() or 'unknown CPU'}"),
        ("Python", platform.python_version()),
        ("numpy", version("numpy")),
        (
            "Process",
            f"{process.factors} factors of {process.harmonics} harmonics, seed {process.seed}",
        ),
        ("Periods drawn", f"{periods.min():.1f} to {periods.max():.1f} steps"),
        ("Units fitted", f"{report.units} per corpus"),
    ]
    return f"## {datetime.now():%Y-%m-%d} — {platform.system()} {platform.machine()}\n\n" + table(
        ("", ""), rows
    )


def recovery_section(report: Report) -> str:
    rows = [
        (
            recovery.layout,
            f"{recovery.coupling:g}",
            f"{recovery.fits:,}"
            + (f" (+{recovery.skipped} too short)" if recovery.skipped else ""),
            f"{recovery.own:.3f}",
            f"{recovery.shuffled:.3f}",
            f"{recovery.excess:+.3f}",
            f"{recovery.weakest_own:.3f}",
        )
        for recovery in report.recoveries
    ]
    return "### Factor recovery\n\n" + table(
        (
            "Corpus",
            "Coupling",
            "Channel fits",
            "Own factors",
            "Another unit's",
            "Excess",
            "Weakest own",
        ),
        rows,
    )


def verdict_section(report: Report) -> str:
    """The question the script exists to answer, answered."""
    lines = ["### Verdict", ""]
    for recovery in report.recoveries:
        threshold = (
            f"at least {COUPLED_EXCESS:g}"
            if recovery.coupling > 0.0
            else f"no more than {NULL_EXCESS:g} either way"
        )
        held = "as designed" if recovery.holds else "**NOT AS DESIGNED**"
        lines.append(
            f"- `{recovery.layout}` came out {held}: excess {recovery.excess:+.3f}, {threshold}."
        )
    if all(recovery.holds for recovery in report.recoveries):
        lines.append(
            "- The control carries the structure it claims, and the null carries none. A pipeline "
            "that finds nothing here is at fault."
        )
    else:
        lines.append(
            "- **The generator does not behave as specified. Nothing below it is usable.**"
        )
    for path in report.figures:
        lines.append(f"- Figure: `{shown(path)}`")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--layout",
        action="append",
        choices=sorted(LAYOUTS),
        help="corpus to measure; repeatable, all four unless given",
    )
    parser.add_argument("--units", type=int, default=12, help="units per corpus to fit on")
    parser.add_argument("--out", default=str(FIGURES), help="directory the figures are written to")
    parser.add_argument("--no-figures", dest="figures", action="store_false")
    arguments = parser.parse_args()
    arguments.layout = arguments.layout or sorted(LAYOUTS)
    if isinstance(sys.stdout, io.TextIOWrapper):
        # The headings use characters a Windows console's default code page lacks, and the note
        # this output is pasted into is LF-only.
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    report = measure(arguments)
    print("\n\n".join((heading(report), recovery_section(report), verdict_section(report))))


if __name__ == "__main__":
    main()
