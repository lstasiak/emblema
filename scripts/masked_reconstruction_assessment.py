"""Store what a masked-reconstruction run measured, find the run it is compared with, print it.

The rules that judge a run live in ``emblema.pretraining`` (``AssessReconstructionRun``); this
module is what surrounds them until the training loop and the experiment tracker exist: the
tables the report prints and the CSV files each run is stored in, a directory of its own and a
line in an index of every run, so that the next run of a configuration can be compared with it.

    uv run scripts/masked_reconstruction_assessment.py data/report/results/control-a-20260913-101500
"""

import argparse
import csv
import io
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.pretraining.adapters.diagnostics.unit_bootstrap import UnitBootstrap
from emblema.pretraining.application.use_cases.assess_reconstruction_run import (
    AssessReconstructionRun,
)
from emblema.pretraining.domain.assessment.assessment import Assessment
from emblema.pretraining.domain.assessment.curve import Curve
from emblema.pretraining.domain.assessment.interval import CONFIDENCE
from emblema.pretraining.domain.assessment.kind_summary import (
    BEYOND_LINEAR,
    MATCHED_BASELINE,
    KindSummary,
)
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally
from emblema.pretraining.domain.assessment.results import Results
from emblema.pretraining.domain.assessment.spectrum import Spectrum
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from scripts.reporting import table

# ------------------------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------------------------


def kinds_table(summaries: Sequence[KindSummary]) -> str:
    """The kinds of mask against their baselines, as the report and the command line print them."""
    rows = [
        (
            summary.kind.value + (" (apart)" if summary.apart else ""),
            f"{summary.tokens:,}",
            str(summary.units),
            f"{summary.model_error:.4f}",
            f"{MATCHED_BASELINE[summary.kind]} {summary.matched_error:.4f}",
            f"{summary.excess:+.4f} {summary.matched_excess}",
            summary.verdict.value,
            f"{summary.linear_error:.4f}",
            f"{summary.linear_excess} {summary.linear_excess.verdict.value}"
            if summary.kind in BEYOND_LINEAR
            else "—",
            f"{summary.mean_error:.4f}",
            "—" if summary.noise_floor is None else f"{summary.noise_floor:.4f}",
        )
        for summary in summaries
    ]
    header = (
        "Kind",
        "Tokens",
        "Units",
        "Model",
        "Matched baseline",
        f"Excess [{CONFIDENCE:.0%}]",
        "Verdict",
        "Linear",
        f"Beyond linear [{CONFIDENCE:.0%}]",
        "Channel mean",
        "Noise floor",
    )
    return (
        table(header, rows)
        + "\n\nErrors are mean squared errors on hidden tokens, in normalised units. Each interval "
        "is a bootstrap over validation units under one draw of the masks, and a kind is `learnt` "
        "only when all of it lies above zero. Both baselines decide: the matched one, which the "
        "plan names, and the linear one, which reads the channel's own line and the other channels "
        "together — the linear one wherever the noise floor leaves room above it."
    )


def assessment_section(assessment: Assessment) -> str:
    """The decision, every check and what to do, as the report prints them under its verdict."""
    decision = assessment.decision
    checks = table(
        ("Area", "Check", "Status", "Measured", "Expected"),
        (
            (check.area.value, check.name, check.status.value, check.measured, check.expected)
            for check in assessment.checks
        ),
    )
    actions = "\n".join(f"- {action}" for action in decision.actions) or "- nothing to do"
    return (
        "### Assessment\n\n"
        f"**{decision.outcome.value}** — {decision.headline}\n\n" + checks + "\n\n" + actions
    )


# ------------------------------------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------------------------------------

RUN = "run.csv"
EPOCHS = "epochs.csv"
TALLIES = "tallies.csv"
SPECTRUM = "spectrum.csv"
KINDS = "mask_kinds.csv"
CHECKS = "checks.csv"
INDEX = "runs.csv"
STRATEGY_FIELDS = ("channel_rate", "block_rate", "block_span", "token_rate")
INDEX_FIELDS = (
    "run",
    "corpus",
    "code",
    "device",
    "shape",
    "epochs",
    "warmup_epochs",
    "final_lr_fraction",
    "batch_size",
    "learning_rate",
    "seed",
    "last_validation",
    *(
        f"{kind.value}_{column}"
        for kind in MaskKind
        for column in ("model", "matched", "excess_low", "excess_high", "verdict")
    ),
    "compared_with_epochs",
    "outcome",
)
# How many runs of one corpus a single second may hold before storing gives up on a name.
SAME_SECOND = 100


def store(assessment: Assessment, root: Path) -> Path:
    """Write the run under a directory of its own in ``root`` and add it to the index.

    The directory is named after the corpus and the date of the run and created afresh: a second
    run stored in the same second gets a numbered name, never the first one's files.

    Raises:
        ValueError: If the index in ``root`` was written with other columns.
        FileExistsError: If every name for the run's second is taken.
    """
    results = assessment.results
    index = root / INDEX
    _check_index(index)
    stamp = datetime.strptime(results.settings["date"], "%Y-%m-%d %H:%M:%S")
    base = f"{results.settings['corpus']}-{stamp:%Y%m%d-%H%M%S}"
    root.mkdir(parents=True, exist_ok=True)
    directory = _fresh_directory(root, base)
    _write_measured(results, directory)
    _write_derived(assessment, directory)
    _append_to_index(assessment, directory.name, index)
    return directory


def read(directory: Path) -> Results:
    """A stored run, from the files that hold what it measured and never from what was concluded.

    A changed rule therefore reassesses an old run rather than trusting the old rule's verdict.
    """
    run = {row["key"]: row["value"] for row in _read_rows(directory / RUN)}
    epochs = _read_rows(directory / EPOCHS)
    spectrum = _read_rows(directory / SPECTRUM)
    reserved = {*STRATEGY_FIELDS, "realised_ratio", "spectrum_fitted", "spectrum_skipped"}
    return Results(
        settings={key: value for key, value in run.items() if key not in reserved},
        strategy=MaskingStrategy(
            channel_rate=float(run["channel_rate"]),
            block_rate=float(run["block_rate"]),
            block_span=float(run["block_span"]),
            token_rate=float(run["token_rate"]),
        ),
        realised_ratio=float(run["realised_ratio"]),
        curve=Curve(
            training=tuple(float(row["training_loss"]) for row in epochs),
            validation=tuple(float(row["validation_loss"]) for row in epochs),
            seconds=tuple(float(row["seconds"]) for row in epochs),
        ),
        tallies=tuple(
            MaskKindTally(
                kind=MaskKind(row["kind"]),
                apart=row["apart"] == "true",
                group=row["unit"],
                tokens=int(row["tokens"]),
                model=float(row["model_sse"]),
                matched=float(row["matched_sse"]),
                linear=float(row["linear_sse"]),
                mean=float(row["mean_sse"]),
                floor=float(row["floor_sse"]) if row["floor_sse"] else None,
            )
            for row in _read_rows(directory / TALLIES)
        ),
        spectrum=Spectrum(
            truth=tuple(float(row["truth_energy"]) for row in spectrum),
            model_residual=tuple(float(row["model_residual_energy"]) for row in spectrum),
            ridge_residual=tuple(float(row["ridge_residual_energy"]) for row in spectrum),
            fitted=int(run["spectrum_fitted"]),
            skipped=int(run["spectrum_skipped"]),
        ),
    )


def find_shorter(results: Results, root: Path) -> Results | None:
    """The stored run of this configuration with the most epochs that are at most half of these.

    Among runs of equally many epochs the latest wins. Only directories holding a run file are
    read, and one that holds a run file but cannot be read is an error rather than a run to skip.

    Raises:
        ValueError: If a stored run cannot be read.
    """
    if not root.is_dir():
        return None
    candidates = []
    for run_file in sorted(root.glob(f"*/{RUN}")):
        try:
            stored = read(run_file.parent)
        except (KeyError, ValueError, OSError) as error:
            raise ValueError(f"{run_file.parent} is not a readable run: {error}") from error
        if results.configuration_of(stored) and stored.epochs <= results.epochs // 2:
            candidates.append(stored)
    if not candidates:
        return None
    return max(candidates, key=lambda stored: (stored.epochs, stored.settings["date"]))


def _fresh_directory(root: Path, base: str) -> Path:
    for attempt in range(SAME_SECOND):
        directory = root / (base if attempt == 0 else f"{base}-{attempt + 1}")
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        return directory
    raise FileExistsError(f"every name for {base} in {root} is taken")


def _write_measured(results: Results, directory: Path) -> None:
    strategy = results.strategy
    run = {
        **results.settings,
        "channel_rate": _number(strategy.channel_rate),
        "block_rate": _number(strategy.block_rate),
        "block_span": _number(strategy.block_span),
        "token_rate": _number(strategy.token_rate),
        "realised_ratio": _number(results.realised_ratio),
        "spectrum_fitted": str(results.spectrum.fitted),
        "spectrum_skipped": str(results.spectrum.skipped),
    }
    _write_rows(directory / RUN, ("key", "value"), run.items())
    curve = results.curve
    _write_rows(
        directory / EPOCHS,
        ("epoch", "training_loss", "validation_loss", "seconds"),
        (
            (str(index + 1), _number(training), _number(validation), _number(seconds))
            for index, (training, validation, seconds) in enumerate(
                zip(curve.training, curve.validation, curve.seconds, strict=True)
            )
        ),
    )
    _write_rows(
        directory / TALLIES,
        (
            "kind",
            "apart",
            "unit",
            "tokens",
            "model_sse",
            "matched_sse",
            "linear_sse",
            "mean_sse",
            "floor_sse",
        ),
        (
            (
                tally.kind.value,
                str(tally.apart).lower(),
                tally.group,
                str(tally.tokens),
                _number(tally.model),
                _number(tally.matched),
                _number(tally.linear),
                _number(tally.mean),
                "" if tally.floor is None else _number(tally.floor),
            )
            for tally in results.tallies
        ),
    )
    spectrum = results.spectrum
    _write_rows(
        directory / SPECTRUM,
        ("cycles", "truth_energy", "model_residual_energy", "ridge_residual_energy"),
        (
            (str(k + 1), _number(truth), _number(model), _number(ridge))
            for k, (truth, model, ridge) in enumerate(
                zip(spectrum.truth, spectrum.model_residual, spectrum.ridge_residual, strict=True)
            )
        ),
    )


def _write_derived(assessment: Assessment, directory: Path) -> None:
    """What the assessment concluded, for reading in a spreadsheet; ``read`` never looks here."""
    _write_rows(
        directory / KINDS,
        (
            "kind",
            "apart",
            "tokens",
            "units",
            "model_error",
            "matched_baseline",
            "matched_error",
            "excess_low",
            "excess_high",
            "verdict",
            "linear_error",
            "linear_excess_low",
            "linear_excess_high",
            "mean_error",
            "noise_floor",
        ),
        (
            (
                summary.kind.value,
                str(summary.apart).lower(),
                str(summary.tokens),
                str(summary.units),
                _number(summary.model_error),
                MATCHED_BASELINE[summary.kind],
                _number(summary.matched_error),
                _number(summary.matched_excess.low),
                _number(summary.matched_excess.high),
                summary.verdict.value,
                _number(summary.linear_error),
                _number(summary.linear_excess.low),
                _number(summary.linear_excess.high),
                _number(summary.mean_error),
                "" if summary.noise_floor is None else _number(summary.noise_floor),
            )
            for summary in assessment.summaries
        ),
    )
    _write_rows(
        directory / CHECKS,
        ("area", "check", "status", "measured", "expected", "reading"),
        (
            (
                check.area.value,
                check.name,
                check.status.value,
                check.measured,
                check.expected,
                check.reading,
            )
            for check in assessment.checks
        ),
    )


def _check_index(index: Path) -> None:
    if not index.exists():
        return
    with index.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    if tuple(header) != INDEX_FIELDS:
        raise ValueError(
            f"{index} was written with other columns; move it aside to start a new index rather "
            "than append rows its header does not describe"
        )


def _append_to_index(assessment: Assessment, name: str, index: Path) -> None:
    results = assessment.results
    settings = results.settings
    line = {
        "run": name,
        **{
            key: settings.get(key, "")
            for key in (
                "corpus",
                "code",
                "device",
                "shape",
                "warmup_epochs",
                "final_lr_fraction",
                "batch_size",
                "learning_rate",
                "seed",
            )
        },
        "epochs": str(results.epochs),
        "last_validation": f"{results.curve.validation[-1]:.6g}",
        "compared_with_epochs": ""
        if assessment.shorter is None
        else str(assessment.shorter.epochs),
        "outcome": assessment.decision.outcome.value,
    }
    for summary in assessment.summaries:
        if summary.apart:
            continue
        prefix = summary.kind.value
        line[f"{prefix}_model"] = f"{summary.model_error:.6g}"
        line[f"{prefix}_matched"] = f"{summary.matched_error:.6g}"
        line[f"{prefix}_excess_low"] = f"{summary.matched_excess.low:.6g}"
        line[f"{prefix}_excess_high"] = f"{summary.matched_excess.high:.6g}"
        line[f"{prefix}_verdict"] = summary.verdict.value
    fresh = not index.exists()
    with index.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS, restval="")
        if fresh:
            writer.writeheader()
        writer.writerow(line)


def _number(value: float) -> str:
    """A number as text that reads back to the same float, whatever numeric type it arrived as."""
    return repr(float(value))


def _write_rows(path: Path, header: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="directories the report stored runs in")
    arguments = parser.parse_args(argv)
    assess = AssessReconstructionRun(UnitBootstrap())
    sections = []
    for directory in arguments.runs:
        results = read(directory)
        assessment = assess(results, find_shorter(results, directory.parent))
        sections.append(
            f"## {directory.name} — {results.settings.get('corpus', '?')}\n\n"
            + kinds_table(assessment.summaries)
            + "\n\n"
            + assessment_section(assessment)
        )
    if isinstance(sys.stdout, io.TextIOWrapper):
        # The tables use ×, → and —; the note this output is pasted into is UTF-8 and LF-only.
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
