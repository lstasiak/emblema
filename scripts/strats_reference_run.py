"""Train the official STraTS on one of the files strats_reference_data.py writes, and answer.

The network's own code at commit e936cda (https://github.com/sindhura97/STraTS, MIT), checked out
beside this project and imported from there: its dataset, its model and its evaluator, unchanged.
Its training loop lives in its entry point, which also imports an optimiser the current
``transformers`` no longer has, so the loop is restated here step for step: AdamW at the
learning rate with ``transformers``' defaults (no weight decay, epsilon 1e-6), gradients clipped
at 0.3, a validation after every epoch by the sum of the areas under the ROC and the
precision-recall curves, the best checkpoint kept, ten epochs of patience, fifty at most. Two
things differ: the device is an argument rather than CUDA, and the stays scored are answered once,
by the best checkpoint, rather than at every validation. The setting is the one its run script
gives the network without self-supervision on this corpus.

``--ablate`` turns parts of it towards this project's network from nothing, one name a part
(``ABLATIONS``; ``ours`` names them all): the class weight off; thirty epochs and the last weights
instead of the early stop; this project's rate schedule on top of that; no dropout; this project's
width, depth and heads; a value embedded by one linear map; the static features read among the
readings; the mean over the readings instead of the learnt attention. The clone is not edited:
each part is a setting, a module put in place of one of its model's, or a subclass of its dataset.

Its dataset reads ``../data/processed/<dataset>.pkl`` relative to where it runs, so each run gets
a directory laid out that way, the file linked in.

    uv run --with transformers --with pytz --with tqdm --with scikit-learn
        scripts/strats_reference_run.py --strats DIR --data FILE --seed N --out DIR [--device mps]
        [--ablate NAME ...]
"""

import argparse
import csv
import importlib
import json
import os
import pickle
import sys
import time
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# Run from anywhere: the project's package is installed, and the rate schedule is taken from it so
# that the ablation runs the schedule this project's network trains under, not a copy of it.
from emblema.shared.kernel.learning_rate_schedule import LearningRateSchedule

HID_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 16
DROPOUT = 0.2
ATTENTION_DROPOUT = 0.2
LEARNING_RATE = 5e-4
MAX_OBS = 880
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
MAX_EPOCHS = 50
PATIENCE = 10
CLIP = 0.3
# The defaults of the optimiser its entry point imports from transformers 4.35.
ADAM_EPSILON = 1e-6
WEIGHT_DECAY = 0.0
DATASET = "physionet_2012"
SCORED = "test"
STRATS_MODULES = ("dataset", "evaluator", "modeling_strats", "utils")

ABLATIONS = (
    "unweighted",
    "fixed-epochs",
    "our-schedule",
    "no-dropout",
    "our-size",
    "linear-value",
    "statics-among",
    "mean-pooling",
)
# The network from nothing at every stay, as its campaigns run it: thirty epochs, no early stop,
# the peak rate its selections chose, a tenth of the run warming up and a cosine decay to a
# hundredth, no clipping, 256 wide in six blocks of four heads.
FIXED_EPOCHS = 30
OUR_LEARNING_RATE = 0.000333
OUR_WARMUP_FRACTION = 0.1
OUR_FINAL_FRACTION = 0.01
OUR_HID_DIM = 256
OUR_NUM_LAYERS = 6
OUR_NUM_HEADS = 4


class LinearValue(nn.Module):
    """A value embedded by one linear map, as this project's encoder embeds it, in place of CVE."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(1, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        embedded: torch.Tensor = self.linear(values.unsqueeze(-1))
        return embedded


class MeanWeights(nn.Module):
    """Equal weights over the observed triplets, in place of the learnt fusion attention.

    Called as the attention is, with the states and the mask of observed triplets, and returning
    one weight per triplet, so the pooled state is the mean over the observed ones.
    """

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        observed = mask.to(states.dtype)
        return observed / observed.sum(dim=-1, keepdim=True).clamp(min=1.0)


def ablations_of(names: Sequence[str]) -> frozenset[str]:
    """The parts turned by ``names``; ``ours`` stands for every one.

    This project's schedule is a schedule over a fixed number of epochs, so it brings them.

    Raises:
        ValueError: If a name is none of the parts.
    """
    unknown = sorted(set(names) - {*ABLATIONS, "ours"})
    if unknown:
        raise ValueError(f"no part is called {unknown}; the parts are {ABLATIONS} and ours")
    turned = set(ABLATIONS) if "ours" in names else set(names)
    if "our-schedule" in turned:
        turned.add("fixed-epochs")
    return frozenset(turned)


def our_schedule(total_steps: int) -> LearningRateSchedule:
    """This project's rate over a run of ``total_steps``, as its adaptation schedule states it."""
    return LearningRateSchedule(
        warmup_steps=min(round(OUR_WARMUP_FRACTION * total_steps), total_steps - 1),
        total_steps=total_steps,
        final_fraction=OUR_FINAL_FRACTION,
    )


# Any: STraTS's dataset class carries no types to name here.
def statics_among(dataset_class: Any) -> Any:
    """Its dataset with the static features left among the triplets and no demographics.

    The demographics become one zero per stay, so its path for them carries nothing, and the
    static variables stay in the data as triplets at the minute they were read.
    """

    class StaticsAmong(dataset_class):  # type: ignore[misc]
        def get_static_varis(self, dataset: str) -> list[str]:
            return []

        def get_static_data(self, data: Any) -> Any:
            self.demo = np.zeros((self.N, 1))
            self.args.D = 1
            return data

    return StaticsAmong


# Any: STraTS's logger, dataset and model classes carry no types to name here.
def settings(
    device: str, seed: int, out: Path, logger: Any, ablations: frozenset[str] = frozenset()
) -> Namespace:
    """The arguments its dataset, model and evaluator read, as its entry point would parse them."""
    dropout = 0.0 if "no-dropout" in ablations else DROPOUT
    sized = "our-size" in ablations
    return Namespace(
        dataset=DATASET,
        train_frac=1.0,
        run="1o1",
        model_type="strats",
        load_ckpt_path=None,
        max_obs=MAX_OBS,
        hid_dim=OUR_HID_DIM if sized else HID_DIM,
        num_layers=OUR_NUM_LAYERS if sized else NUM_LAYERS,
        num_heads=OUR_NUM_HEADS if sized else NUM_HEADS,
        dropout=dropout,
        attention_dropout=0.0 if "no-dropout" in ablations else ATTENTION_DROPOUT,
        pretrain=0,
        seed=seed,
        max_epochs=FIXED_EPOCHS if "fixed-epochs" in ablations else MAX_EPOCHS,
        patience=PATIENCE,
        lr=OUR_LEARNING_RATE if "our-schedule" in ablations else LEARNING_RATE,
        train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        eval_batch_size=EVAL_BATCH_SIZE,
        output_dir=str(out),
        logger=logger,
        device=torch.device(device),
    )


def answers(
    model: torch.nn.Module, dataset: Any, args: Namespace
) -> list[tuple[int, float, float]]:
    """Each scored stay's index, label and answer, from the model as it stands."""
    model.eval()
    rows: list[tuple[int, float, float]] = []
    indices = dataset.splits[SCORED]
    with torch.no_grad():
        for start in range(0, len(indices), args.eval_batch_size):
            batch_indices = indices[start : start + args.eval_batch_size]
            batch = dataset.get_batch(batch_indices)
            labels = batch.pop("labels")
            batch = {key: value.to(args.device) for key, value in batch.items()}
            predicted = model(**batch).cpu()
            rows.extend(
                (int(index), float(label), float(answer))
                for index, label, answer in zip(batch_indices, labels, predicted, strict=True)
            )
    return rows


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--strats", type=Path, required=True, help="its checkout")
    parser.add_argument("--data", type=Path, required=True, help="a file of the data script")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory of this run")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-epochs", type=int, help="fewer only for a smoke run")
    parser.add_argument(
        "--ablate", nargs="+", default=[], metavar="NAME", help=f"of {ABLATIONS} or ours"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    try:
        ablations = ablations_of(arguments.ablate)
    except ValueError as error:
        sys.exit(str(error))
    arguments.data = arguments.data.resolve()
    arguments.strats = arguments.strats.resolve()
    out = arguments.out.resolve()
    processed = out / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    linked = processed / f"{DATASET}.pkl"
    linked.unlink(missing_ok=True)
    linked.symlink_to(arguments.data)
    (out / "src").mkdir(exist_ok=True)
    sys.path.insert(0, str(arguments.strats / "src"))
    os.chdir(out / "src")

    # Its modules, imported by name once the path to them is set; they carry no types.
    strats = {name: importlib.import_module(name) for name in STRATS_MODULES}
    logger = strats["utils"].Logger(str(out), "log.txt")

    args = settings(arguments.device, arguments.seed, out, logger, ablations)
    if arguments.max_epochs is not None:
        args.max_epochs = arguments.max_epochs
    strats["utils"].set_all_seeds(args.seed + int(args.run.split("o")[0]))
    dataset_class = strats["dataset"].Dataset
    if "statics-among" in ablations:
        dataset_class = statics_among(dataset_class)
    dataset = dataset_class(args)
    if "unweighted" in ablations:
        # Its model reads the weight when it is built.
        args.pos_class_weight = 1.0
    model = strats["modeling_strats"].Strats(args)
    if "linear-value" in ablations:
        model.cve_value = LinearValue(args.hid_dim)
    if "mean-pooling" in ablations:
        model.fusion_att = MeanWeights()
    model = model.to(args.device)
    fixed = "fixed-epochs" in ablations
    kept_path = out / "checkpoint.bin"

    batches_per_epoch = len(dataset.splits["train"]) / args.train_batch_size
    max_steps = int(round(batches_per_epoch) * args.max_epochs)
    validate_every = int(np.ceil(batches_per_epoch))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
    )
    rate = (
        torch.optim.lr_scheduler.LambdaLR(optimizer, our_schedule(max_steps).factor)
        if "our-schedule" in ablations
        else None
    )
    evaluator = strats["evaluator"].Evaluator(args)
    wait, best, best_step, started = args.patience, -np.inf, -1, time.perf_counter()
    model.train()
    for step in range(max_steps):
        batch = {key: value.to(args.device) for key, value in dataset.get_batch().items()}
        loss = model(**batch)
        if not torch.isnan(loss):
            loss.backward()
            if rate is None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            optimizer.zero_grad()
        if rate is not None:
            rate.step()
        if (step + 1) % validate_every == 0:
            result = evaluator.evaluate(model, dataset, "val", train_step=step)
            model.train(True)
            metric = result["auprc"] + result["auroc"]
            if metric > best:
                best, best_step, wait = metric, step + 1, args.patience
                if not fixed:
                    torch.save(model.state_dict(), kept_path)
            elif not fixed:
                wait -= 1
                if wait == 0:
                    break
    seconds = time.perf_counter() - started
    if fixed:
        # The run's own network as it ended, which is what this project's network is scored as.
        torch.save(model.state_dict(), kept_path)

    model.load_state_dict(torch.load(kept_path, map_location=args.device))
    stays = dataset_stays(arguments.data, dataset)
    with (out / "predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "unit", "target", "predicted"])
        writer.writerows(
            (args.seed, stays[index], label, answer)
            for index, label, answer in answers(model, dataset, args)
        )
    (out / "run.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "data": str(arguments.data),
                "ablations": sorted(ablations),
                "learning_rate": args.lr,
                "shape": [args.hid_dim, args.num_layers, args.num_heads],
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "dropout": [args.dropout, args.attention_dropout],
                "device": arguments.device,
                "torch": torch.__version__,
                "best_validation": best,
                "best_step": best_step,
                "steps_per_epoch": validate_every,
                "last_step": step + 1,
                "seconds": seconds,
                "sides": {side: len(ids) for side, ids in dataset.splits.items()},
                "positive_weight": float(args.pos_class_weight),
            },
            indent=2,
        )
    )
    print(f"seed {args.seed}: best {best:.4f} at step {best_step}, {seconds:.0f} s")


def dataset_stays(path: Path, dataset: Any) -> list[str]:
    """The stay each index of its dataset stands for, as its dataset orders them.

    The dataset keeps no map back to the stays, so its order is restated here: the stays of each
    side that hold a variable the learning side observed, each side sorted, learnt, stopped on,
    scored. Every stay's label and count of readings, at most the readings the network keeps, are
    checked against what the dataset holds at its index, so a restatement that drifted from its
    order stops the run instead of mislabelling answers.
    """
    with path.open("rb") as file:
        data, outcomes, train_ids, val_ids, test_ids = pickle.load(file)
    observed = data.loc[data.ts_id.isin(train_ids)].variable.unique()
    kept = data.loc[data.variable.isin(observed)]
    stays = [
        str(stay)
        for side in (train_ids, val_ids, test_ids)
        for stay in np.intersect1d(side, kept.ts_id.unique())
    ]
    labels = dict(zip(outcomes.ts_id, outcomes.in_hospital_mortality, strict=True))
    series = kept.loc[~kept.variable.isin(dataset.static_varis)].ts_id.value_counts()
    readings = [min(int(series.get(stay, 0)), MAX_OBS) for stay in stays]
    if (
        len(stays) != dataset.N
        or any(labels[stay] != label for stay, label in zip(stays, dataset.y, strict=True))
        or readings != [len(values) for values in dataset.values]
    ):
        raise SystemExit("the dataset's order of stays is not the one restated here")
    return stays


if __name__ == "__main__":
    main()
