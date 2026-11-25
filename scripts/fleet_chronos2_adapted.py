r"""The published forecaster fine-tuned for a turbine or a farm read as new, and its answers.

``fit`` adapts Chronos-2 by low-rank adapters (``--mode full``: every weight) on the series
``fleet_adaptation.py`` builds: the target farm's other turbines (``--fold N --folds K``: the
fold's turbines are neither read nor answered by any other fit), with another farm's turbines
beside them (``--beside``), or the other farm's turbines alone (``--other``), with the
missions' hourly means beside them in a share of the series (``--missions``). Every series ends
before ``--until``; a draw reads at most ``--context`` hours, by default the thirty days of a
span; with ``--complete`` the spans are cut from the hours whose conditions all read.
``answer`` hands each cell to the one fit of an arm that may answer it, in the normal-behaviour
form, on two rungs as ``fleet_chronos2_answers.py`` reads the zero-shot model:

- ``<arm>_nbm``: the 47 hours before, with the hour's operating conditions known;
- ``<arm>_nbm_own``: every hour since the start, up to the most the model reads.

The forecaster and its adapters are not dependencies of this package, so this runs in its own
environment and reads nothing of it:

    uv run --isolated --no-project --python 3.12 --with chronos-forecasting==2.3.2 \\
        --with peft==0.18.1 python -I scripts/fleet_chronos2_adapted.py fit --farm F \\
        --hourly DIR (--fold N --folds K [--beside NAME DIR] | --other NAME DIR) \\
        [--missions DIR --share 0.5] [--complete] \\
        --until 2020-01-01 --arm ARM --out FITS/NAME [--rank 8 --steps 2000 ...]
    uv run ... scripts/fleet_chronos2_adapted.py answer --hourly DIR \\
        --cells DIR/answers-<label>.csv --fits FITS/NAME... --out DIR/<arm>-<label>.csv

``fit`` writes the adapters under ``FITS/NAME/checkpoint`` and ``fit.json``, the record of what
it read; with ``--save-every N`` also every N steps (``checkpoint-<step>``), which under
``--schedule constant`` are the fits of fewer steps, read by ``answer --checkpoint``. ``answer``
writes one row per cell: window, axis, channel and each rung's median.
"""

import argparse
import faulthandler
import sys
import time
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Run in the forecaster's own environment: only the sibling scripts are reached from the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.chronos2_answers import median_index  # noqa: E402
from scripts.chronos2_ladder import (  # noqa: E402
    CONTEXT,
    Input,
    Key,
    Predict,
    load,
    read_cells,
    write,
)
from scripts.fleet_adaptation import (  # noqa: E402
    SPAN,
    FitRecord,
    Series,
    refused,
    scope_of,
    training_series,
)
from scripts.fleet_chronos2_answers import NBM, NBM_OWN, STARTUP_SECONDS, answered  # noqa: E402
from scripts.fleet_hours import another_farm, loaded_farm, parsed_axis  # noqa: E402

RUNGS = {NBM: "_nbm", NBM_OWN: "_nbm_own"}
# The library's own choice of adapted modules: every attention projection and the output layer.
ADAPTED = (
    "self_attention.q",
    "self_attention.k",
    "self_attention.v",
    "self_attention.o",
    "output_patch_embedding.output_layer",
)


def assigned(records: Sequence[FitRecord], cells: Sequence[Key]) -> dict[str, list[Key]]:
    """Each fit's cells: the one fit of the arm that answers a cell's turbine.

    Raises:
        ValueError: If the fits belong to more than one arm, no fit or two answer a turbine,
            or a fit may not answer its cells.
    """
    arms = {r.arm for r in records}
    if len(arms) != 1:
        raise ValueError(f"the fits belong to {sorted(arms)}, not to one arm")
    out: dict[str, list[Key]] = {r.name: [] for r in records}
    for key in cells:
        turbine = parsed_axis(key[1])[0]
        owners = [r.name for r in records if turbine in r.answers]
        if len(owners) != 1:
            raise ValueError(f"{len(owners)} fits answer turbine {turbine}")
        out[owners[0]].append(key)
    reasons = [why for r in records for why in refused(r, out[r.name])]
    if reasons:
        raise ValueError("; ".join(reasons))
    return out


def named_as(records: Sequence[FitRecord], checkpoint: str, named: str | None) -> str:
    """The answers' name: the fits' arm, or ``named`` for adapters kept on the way.

    Raises:
        ValueError: If adapters kept on the way are read without a name of their own, or from
            a fit whose rate was not constant, where they are no fit of fewer steps.
    """
    if checkpoint != "checkpoint" and (not named or any(r.schedule != "constant" for r in records)):
        raise ValueError("adapters kept on the way are read under a constant rate and a name")
    return named or records[0].arm


def prepared(series: Sequence[Series]) -> list[Any]:  # pragma: no cover - needs the model
    """Each series in the library's prepared form, one at a time.

    The library refuses a list of series whose targets or covariates differ — a turbine's
    seventeen temperatures with its conditions beside a mission's channels alone — but trains on
    prepared series of any shape, each a group of its own; a series handed again is prepared once.
    """
    import chronos.chronos2.preprocess as preprocess  # type: ignore[import-not-found,unused-ignore]  # ty: ignore[unresolved-import]

    done: dict[int, Any] = {}
    out = []
    for one in series:
        if id(one) not in done:
            (done[id(one)],) = preprocess.from_list_of_dicts([one], prediction_length=1)
        out.append(done[id(one)])
    return out


def saving(every: int | None) -> dict[str, object]:
    """The trainer's settings that keep the adapters every ``every`` steps, or none.

    Under a constant rate, the adapters kept on the way are the fits of fewer steps, so one run
    reads the dose of steps.
    """
    if not every:
        return {}
    return {"save_strategy": "steps", "save_steps": every, "save_total_limit": None}


def said(name: str) -> Callable[[str], None]:
    def say(text: str) -> None:
        print(f"{name} {text}", file=sys.stderr, flush=True)

    return say


def predicting(pipeline: Any, cap: int) -> Predict:  # pragma: no cover - needs the model
    """The pipeline's answers for a batch of inputs, one hour ahead from up to ``cap`` hours."""

    def predict(batch: Sequence[Input]) -> list[NDArray[np.float32]]:
        outputs = pipeline.predict(list(batch), prediction_length=1, context_length=cap)
        return [t.cpu().numpy() for t in outputs]

    return predict


def fit(arguments: argparse.Namespace) -> None:  # pragma: no cover - needs the model
    import chronos  # type: ignore[import-not-found,unused-ignore]  # ty: ignore[unresolved-import]
    import torch  # type: ignore[import-not-found,unused-ignore]

    if arguments.mode == "lora":
        try:
            import peft  # type: ignore[import-not-found,unused-ignore]  # ty: ignore[unresolved-import]  # noqa: F401
        except ImportError as error:
            # Without it the library falls back to tuning every weight, with a warning only.
            raise SystemExit("peft is not installed: the fit would tune every weight") from error

    if bool(arguments.mission_names) != bool(arguments.missions):
        raise SystemExit("missions are named with their directory")
    target = loaded_farm(arguments.hourly)
    until = date.fromisoformat(arguments.until)
    try:
        answers, left_out = scope_of(
            arguments.farm,
            [t.name for t in target],
            fold=arguments.fold,
            of=arguments.folds,
            other=arguments.other[0] if arguments.other else None,
            beside=arguments.beside[0] if arguments.beside else None,
            missions=bool(arguments.missions),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    farms = {} if arguments.other else {arguments.farm: target}
    for name, directory in filter(None, (arguments.other, arguments.beside)):
        farms[name] = another_farm(Path(directory), arguments.hourly)
    missions = {m: load(arguments.missions, m) for m in arguments.mission_names}
    trained = training_series(
        farms,
        left_out=left_out,
        until=until,
        missions=missions or None,
        share=arguments.share,
        seed=arguments.seed,
        complete=arguments.complete,
    )
    record = FitRecord(
        name=arguments.out.name,
        arm=arguments.arm,
        farm=arguments.farm,
        answers=answers,
        turbines=trained.turbines,
        missions=tuple(missions),
        share=arguments.share if missions else 0.0,
        turbine_series=trained.turbine_series,
        mission_series=trained.mission_series,
        until=until.isoformat(),
        model=arguments.model,
        mode=arguments.mode,
        rank=arguments.rank if arguments.mode == "lora" else 0,
        steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        schedule=arguments.schedule,
        context=arguments.context,
        batch=arguments.batch,
        save_every=arguments.save_every or 0,
        seed=arguments.seed,
        complete=arguments.complete,
    )
    arguments.out.mkdir(parents=True, exist_ok=True)
    if (arguments.out / "fit.json").exists():
        if FitRecord.read(arguments.out) != record:
            raise SystemExit(f"{arguments.out} holds another fit")
        if (arguments.out / "checkpoint").exists():
            print(f"{record.name} is fitted already")
            return
    # Written before training, so that adapters kept on the way are never without their record.
    record.write(arguments.out)
    print(
        f"{record.name}: {trained.turbine_series} series of {trained.turbines}, "
        f"{trained.mission_series} of {list(missions)}"
    )
    started = time.monotonic()
    faulthandler.dump_traceback_later(STARTUP_SECONDS, exit=True)
    pipeline = chronos.Chronos2Pipeline.from_pretrained(
        arguments.model, device_map=arguments.device, dtype=torch.float32
    )
    faulthandler.cancel_dump_traceback_later()
    adapters = (
        {"r": arguments.rank, "lora_alpha": 2 * arguments.rank, "target_modules": list(ADAPTED)}
        if arguments.mode == "lora"
        else None
    )
    pipeline.fit(
        prepared(trained.series),
        prediction_length=1,
        finetune_mode=arguments.mode,
        lora_config=adapters,
        context_length=arguments.context,
        learning_rate=arguments.learning_rate,
        num_steps=arguments.steps,
        batch_size=arguments.batch,
        output_dir=arguments.out,
        # A draw reads at least the context of a cell on its first days.
        min_past=CONTEXT,
        finetuned_ckpt_name="checkpoint",
        seed=arguments.seed,
        data_seed=arguments.seed,
        optim="adamw_torch_fused",
        logging_steps=max(arguments.steps // 20, 1),
        lr_scheduler_type=arguments.schedule,
        **saving(arguments.save_every),
    )
    print(f"{record.name} fitted in {time.monotonic() - started:.0f} s")


def answer(arguments: argparse.Namespace) -> None:  # pragma: no cover - needs the model
    import chronos  # type: ignore[import-not-found,unused-ignore]  # ty: ignore[unresolved-import]
    import torch  # type: ignore[import-not-found,unused-ignore]

    records = [FitRecord.read(d) for d in arguments.fits]
    turbines = loaded_farm(arguments.hourly)
    cells = read_cells(arguments.cells)
    farms = {r.farm for r in records}
    if len(farms) != 1:
        raise SystemExit(f"the fits answer {sorted(farms)}, not one farm")
    by_fit = assigned(records, cells)
    try:
        arm = named_as(records, arguments.checkpoint, arguments.named)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    out: dict[str, dict[Key, float]] = {arm + suffix: {} for suffix in RUNGS.values()}
    started = time.monotonic()
    for record, directory in zip(records, arguments.fits, strict=True):
        mine = by_fit[record.name]
        if not mine:
            continue
        faulthandler.dump_traceback_later(STARTUP_SECONDS, exit=True)
        pipeline = chronos.Chronos2Pipeline.from_pretrained(
            str(directory / arguments.checkpoint), device_map=arguments.device, dtype=torch.float32
        )
        faulthandler.cancel_dump_traceback_later()
        cap = int(pipeline.model_context_length)

        given = answered(
            mine,
            turbines,
            predicting(pipeline, cap),
            median_index(pipeline.quantiles),
            cap=cap,
            batch=arguments.batch,
            rungs=tuple(RUNGS),
            say=said(record.name),
        )
        for rung, suffix in RUNGS.items():
            out[arm + suffix].update(given[rung])
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    write(arguments.out, cells, out, tuple(out))
    print(
        f"{len(cells)} cells answered by {len(records)} fits in {time.monotonic() - started:.0f} s"
    )


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - needs the model
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("fit")
    fitting.add_argument("--farm", required=True, help="the farm whose cells the fit answers")
    fitting.add_argument("--hourly", type=Path, required=True, help="that farm's hourly means")
    fitting.add_argument("--fold", type=int)
    fitting.add_argument("--folds", type=int)
    fitting.add_argument("--other", nargs=2, metavar=("FARM", "DIR"), help="read it instead")
    fitting.add_argument("--beside", nargs=2, metavar=("FARM", "DIR"), help="read it as well")
    fitting.add_argument(
        "--complete", action="store_true", help="spans of the hours whose conditions all read"
    )
    fitting.add_argument("--missions", type=Path, help="the missions' hourly means")
    fitting.add_argument(
        "--mission-names", nargs="*", default=[], help="e.g. ESA-Mission1 ESA-Mission2"
    )
    fitting.add_argument("--share", type=float, default=0.5, help="the missions' series")
    fitting.add_argument("--until", required=True, help="the first day not read")
    fitting.add_argument("--arm", required=True)
    fitting.add_argument("--mode", choices=("lora", "full"), default="lora")
    fitting.add_argument("--rank", type=int, default=8, help="the adapters', under lora")
    fitting.add_argument("--steps", type=int, default=2000)
    fitting.add_argument("--learning-rate", type=float, default=1e-5)
    fitting.add_argument("--context", type=int, default=SPAN, help="hours a draw reads at most")
    fitting.add_argument(
        "--batch", type=int, default=128, help="rows per step: a series' targets and covariates"
    )
    fitting.add_argument("--seed", type=int, default=1)
    fitting.add_argument("--schedule", choices=("linear", "constant"), default="linear")
    fitting.add_argument("--save-every", type=int, help="keep the adapters every N steps")
    fitting.add_argument("--out", type=Path, required=True)
    answering = commands.add_parser("answer")
    answering.add_argument("--hourly", type=Path, required=True)
    answering.add_argument("--cells", type=Path, required=True, help="fleet_cold_start_cells's")
    answering.add_argument("--fits", type=Path, nargs="+", required=True)
    answering.add_argument("--batch", type=int, default=32, help="windows handed at once")
    answering.add_argument("--checkpoint", default="checkpoint", help="e.g. checkpoint-500")
    answering.add_argument("--named", help="the answers' name instead of the fits' arm")
    answering.add_argument("--out", type=Path, required=True)
    fitting.add_argument("--model", default="amazon/chronos-2")
    for command in (fitting, answering):
        command.add_argument("--device", default="mps")
    arguments = parser.parse_args(argv)
    if arguments.command == "fit":
        fit(arguments)
    else:
        answer(arguments)


if __name__ == "__main__":  # pragma: no cover
    main()
