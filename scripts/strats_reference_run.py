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

Its dataset reads ``../data/processed/<dataset>.pkl`` relative to where it runs, so each run gets
a directory laid out that way, the file linked in.

    uv run --with transformers --with pytz --with tqdm --with scikit-learn
        scripts/strats_reference_run.py --strats DIR --data FILE --seed N --out DIR [--device mps]
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


# Any: STraTS's logger, dataset and model classes carry no types to name here.
def settings(device: str, seed: int, out: Path, logger: Any) -> Namespace:
    """The arguments its dataset, model and evaluator read, as its entry point would parse them."""
    return Namespace(
        dataset=DATASET,
        train_frac=1.0,
        run="1o1",
        model_type="strats",
        load_ckpt_path=None,
        max_obs=MAX_OBS,
        hid_dim=HID_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        attention_dropout=ATTENTION_DROPOUT,
        pretrain=0,
        seed=seed,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        lr=LEARNING_RATE,
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
    parser.add_argument(
        "--max-epochs", type=int, default=MAX_EPOCHS, help="fewer only for a smoke run"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
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

    args = settings(arguments.device, arguments.seed, out, logger)
    args.max_epochs = arguments.max_epochs
    strats["utils"].set_all_seeds(args.seed + int(args.run.split("o")[0]))
    dataset = strats["dataset"].Dataset(args)
    model = strats["modeling_strats"].Strats(args).to(args.device)
    best_path = out / "checkpoint_best.bin"

    batches_per_epoch = len(dataset.splits["train"]) / args.train_batch_size
    max_steps = int(round(batches_per_epoch) * args.max_epochs)
    validate_every = int(np.ceil(batches_per_epoch))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
    )
    evaluator = strats["evaluator"].Evaluator(args)
    wait, best, best_step, started = args.patience, -np.inf, -1, time.perf_counter()
    model.train()
    for step in range(max_steps):
        batch = {key: value.to(args.device) for key, value in dataset.get_batch().items()}
        loss = model(**batch)
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            optimizer.zero_grad()
        if (step + 1) % validate_every == 0:
            result = evaluator.evaluate(model, dataset, "val", train_step=step)
            model.train(True)
            metric = result["auprc"] + result["auroc"]
            if metric > best:
                best, best_step, wait = metric, step + 1, args.patience
                torch.save(model.state_dict(), best_path)
            else:
                wait -= 1
                if wait == 0:
                    break
    seconds = time.perf_counter() - started

    model.load_state_dict(torch.load(best_path, map_location=args.device))
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
