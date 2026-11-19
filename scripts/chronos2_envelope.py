r"""A published forecaster's envelope for each channel's hour, from the 47 hourly means before it.

Chronos-2 is asked, for every loaded window and every channel with a reading in the hours
before it, for the hour's mean: the channels of a window are handed together as the variates of
one series, as the forecast was read on 2026-11-15, and its 10th, 50th and 90th percentiles are
kept. The detector built on them reads how far the hour's mean falls from the median, over the
spread between the outer two. Windows are answered in chunks written as they finish, so a run
cut short starts again at the first chunk missing.

The forecaster is not a dependency of this package, so this runs in its own environment:

    uv run --isolated --no-project --python 3.12 --with chronos-forecasting==2.3.2 \\
        python -I scripts/chronos2_envelope.py --contexts DIR/contexts.npy --out DIR/envelope

Reads ``[windows, rows, hours]`` hourly means, missing as NaN; writes ``envelope.npy``,
``[windows, rows, 3]``, missing where a channel had no reading in the hours before.
"""

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

LEVELS = (0.1, 0.5, 0.9)
CHUNK = 4096
Floats = NDArray[np.float32]
# Takes ``[variates, hours]`` arrays and gives, for each, ``[variates, quantiles, 1]``.
Predict = Callable[[Sequence[Floats]], Sequence[Floats]]


def level_indices(quantiles: Sequence[float]) -> list[int]:
    """Where the model's quantile levels hold the three this envelope keeps.

    Raises:
        ValueError: If the model gives one of them no level.
    """
    found = []
    for level in LEVELS:
        matches = [k for k, q in enumerate(quantiles) if abs(q - level) < 1e-9]
        if not matches:
            raise ValueError(f"the model's quantiles hold no {level}: {list(quantiles)}")
        found.append(matches[0])
    return found


def envelope_of(contexts: Floats, predict: Predict, quantiles: Sequence[float]) -> Floats:
    """``[windows, rows, 3]`` for one chunk: each window's channels with a reading handed together.

    Raises:
        ValueError: If the model answers another number of variates than handed, or a value
            that is not finite.
    """
    at = level_indices(quantiles)
    out = np.full((*contexts.shape[:2], len(LEVELS)), np.nan, dtype=np.float32)
    read = np.isfinite(contexts).any(axis=2)
    windows = [w for w in range(len(contexts)) if read[w].any()]
    if not windows:
        return out
    answers = predict([contexts[w, read[w]] for w in windows])
    for w, answer in zip(windows, answers, strict=True):
        if answer.shape[0] != int(read[w].sum()):
            raise ValueError(f"the model answered {answer.shape[0]} variates for {read[w].sum()}")
        kept = answer[:, at, 0]
        if not np.isfinite(kept).all():
            raise ValueError(f"the model answered a value that is not finite in window {w}")
        out[w, read[w]] = kept
    return out


def chunk_path(out: Path, start: int) -> Path:
    return out / f"chunk-{start:07d}.npy"


def fingerprint(contexts: Floats, chunk: int) -> dict[str, object]:
    """The contexts' shape and a digest of every chunk's first window, which ties chunks to them."""
    sample = np.ascontiguousarray(contexts[::chunk])
    return {"shape": list(contexts.shape), "sha256": hashlib.sha256(sample.tobytes()).hexdigest()}


def run(
    contexts: Floats,
    out: Path,
    predict: Predict,
    quantiles: Sequence[float],
    *,
    chunk: int = CHUNK,
    say: Callable[[str], None] = lambda _: None,
) -> Floats:
    """The envelope of every window, each chunk written as it finishes and skipped if written.

    Raises:
        SystemExit: If the chunks already written came from other contexts.
    """
    out.mkdir(parents=True, exist_ok=True)
    source = out / "contexts.json"
    held = fingerprint(contexts, chunk) | {"chunk": chunk}
    if source.exists() and json.loads(source.read_text()) != held:
        raise SystemExit(f"{out} holds chunks of other contexts")
    source.write_text(json.dumps(held) + "\n")
    started = time.monotonic()
    for start in range(0, len(contexts), chunk):
        path = chunk_path(out, start)
        if path.exists():
            continue
        part = envelope_of(np.asarray(contexts[start : start + chunk]), predict, quantiles)
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, part)
        temporary.replace(path)
        say(f"{start + len(part)} of {len(contexts)} windows in {time.monotonic() - started:.0f} s")
    return np.concatenate([np.load(chunk_path(out, s)) for s in range(0, len(contexts), chunk)])


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - needs the model
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="a directory for the chunks")
    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=256, help="windows handed at once")
    arguments = parser.parse_args(argv)

    import chronos  # type: ignore[import-not-found,unused-ignore]  # ty: ignore[unresolved-import]
    import torch  # type: ignore[import-not-found,unused-ignore]

    contexts = np.load(arguments.contexts, mmap_mode="r")
    pipeline = chronos.Chronos2Pipeline.from_pretrained(
        arguments.model, device_map=arguments.device, dtype=torch.float32
    )

    def predict(batch: Sequence[Floats]) -> list[Floats]:
        answers: list[Floats] = []
        for start in range(0, len(batch), arguments.batch):
            handed = list(batch[start : start + arguments.batch])
            answers += [t.cpu().numpy() for t in pipeline.predict(handed, prediction_length=1)]
        return answers

    envelope = run(
        contexts,
        arguments.out,
        predict,
        pipeline.quantiles,
        say=lambda text: print(text, file=sys.stderr, flush=True),
    )
    np.save(arguments.out / "envelope.npy", envelope)
    print(f"{len(envelope)} windows, {int(np.isfinite(envelope[..., 1]).sum())} channel hours")


if __name__ == "__main__":  # pragma: no cover
    main()
