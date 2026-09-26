"""What the frozen states of a window pool to, checked token by token on a small encoder."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from scripts.frozen_representations import (  # noqa: E402
    TOLERANCE,
    channels_read,
    pooled,
    represent,
)
from scripts.head_and_representation_report import (  # noqa: E402
    CHANNEL_LAST,
    CHANNEL_MEAN,
    MEAN,
    save_arrays,
    tail_name,
)
from tests.support.encoders import SMALL, small_encoder  # noqa: E402
from tests.support.published import FIRST, HELD_OUT, OTHER, SECOND  # noqa: E402

pytestmark = pytest.mark.ml

# The four windows of the shared test corpus, and the channels the three "tuning" ones observe.
WINDOWS = (FIRST, SECOND, HELD_OUT, OTHER)
CHANNELS = np.array([1, 2, 3], dtype=np.int64)
WIDTH = SMALL.width


def test_the_channels_read_are_those_the_tuning_windows_observe_ascending() -> None:
    np.testing.assert_array_equal(channels_read((FIRST, SECOND, OTHER)), CHANNELS)
    np.testing.assert_array_equal(channels_read((OTHER,)), np.array([2]))


def test_every_pooling_reads_the_tokens_it_says_it_reads() -> None:
    encoder = small_encoder()
    batch = TokenTensors.from_windows(WINDOWS)
    with torch.no_grad():
        states = encoder(*batch.args)
        found = pooled(states, batch, torch.as_tensor(CHANNELS))

    # FIRST holds channel 1 at times 0 and 1, channel 2 at 0.5, in that canonical order.
    first = states[0]
    torch.testing.assert_close(found[MEAN][0], first[:3].mean(dim=0))
    torch.testing.assert_close(found[tail_name(0.1)][0], first[2])
    torch.testing.assert_close(found[tail_name(0.2)][0], first[2])
    per_channel_mean = found[CHANNEL_MEAN][0].reshape(len(CHANNELS), WIDTH)
    torch.testing.assert_close(per_channel_mean[0], (first[0] + first[2]) / 2)
    torch.testing.assert_close(per_channel_mean[1], first[1])
    assert per_channel_mean[2].abs().sum() == 0.0
    per_channel_last = found[CHANNEL_LAST][0].reshape(len(CHANNELS), WIDTH)
    torch.testing.assert_close(per_channel_last[0], first[2])
    torch.testing.assert_close(per_channel_last[1], first[1])
    assert per_channel_last[2].abs().sum() == 0.0
    # SECOND holds the timeless channel 3 first, then channel 1 at 0.25: the tail keeps only the
    # static feature, which is current at every instant.
    second = states[1]
    torch.testing.assert_close(found[tail_name(0.1)][1], second[0])
    torch.testing.assert_close(found[CHANNEL_LAST][1].reshape(len(CHANNELS), WIDTH)[2], second[0])
    # HELD_OUT holds two tokens and is padded to three: the padding pools into nothing.
    held = states[2]
    torch.testing.assert_close(found[MEAN][2], held[:2].mean(dim=0))
    torch.testing.assert_close(found[CHANNEL_LAST][2].reshape(len(CHANNELS), WIDTH)[0], held[0])


def test_a_tail_at_the_edge_of_its_share_keeps_the_instant_on_the_edge() -> None:
    encoder = small_encoder()
    batch = TokenTensors.from_windows((FIRST,))
    # Single precision puts 0.9 a hair below the share's edge; the tolerance puts it back.
    batch = TokenTensors(
        features=batch.features,
        channel_ids=batch.channel_ids,
        timestamps=torch.tensor([[0.0, 0.9 - TOLERANCE / 2, 1.0]], dtype=torch.float32),
        timeless=batch.timeless,
        padding_mask=batch.padding_mask,
    )
    with torch.no_grad():
        states = encoder(*batch.args)
        found = pooled(states, batch, torch.as_tensor(CHANNELS), tails=(0.1,))

    torch.testing.assert_close(found[tail_name(0.1)][0], states[0, 1:].mean(dim=0))


def test_representing_in_batches_gives_the_rows_one_batch_gives() -> None:
    encoder = small_encoder()
    batched = represent(encoder, WINDOWS, channels=CHANNELS, device="cpu", batch_size=3)
    whole = represent(encoder, WINDOWS, channels=CHANNELS, device="cpu", batch_size=4)

    assert set(batched) == {
        MEAN,
        CHANNEL_MEAN,
        CHANNEL_LAST,
        *(tail_name(s) for s in (0.02, 0.1, 0.2)),
    }
    assert batched[MEAN].shape == (4, WIDTH)
    assert batched[CHANNEL_MEAN].shape == (4, len(CHANNELS) * WIDTH)
    assert batched[MEAN].dtype == np.float32
    for name, rows in whole.items():
        np.testing.assert_allclose(batched[name], rows, atol=1e-5)


def test_the_archive_is_read_back_without_pickling(tmp_path: Path) -> None:
    arrays = {
        "states": np.arange(6, dtype=np.float32).reshape(2, 3),
        "names": np.array(["a", "b"]),
        "channels": np.array([1, 2], dtype=np.int64),
    }

    save_arrays(tmp_path / "arrays.npz", arrays)

    with np.load(tmp_path / "arrays.npz", allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for name, array in arrays.items():
            np.testing.assert_array_equal(loaded[name], array)
