import pytest

from emblema.evaluation.domain.exceptions import InvalidEncoderSettingError
from emblema.evaluation.domain.transfer.encoder_shape import EncoderShape


def test_a_shape_holds_its_four_counts() -> None:
    shape = EncoderShape(width=64, heads=16, layers=2, feedforward_width=128)

    assert (shape.width, shape.heads, shape.layers, shape.feedforward_width) == (64, 16, 2, 128)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ({"width": 0, "heads": 1, "layers": 1, "feedforward_width": 1}, "width must be positive"),
        ({"width": 8, "heads": 0, "layers": 1, "feedforward_width": 1}, "heads must be positive"),
        ({"width": 8, "heads": 1, "layers": 0, "feedforward_width": 1}, "layers must be positive"),
        (
            {"width": 8, "heads": 1, "layers": 1, "feedforward_width": 0},
            "feedforward_width must be positive",
        ),
        ({"width": 10, "heads": 4, "layers": 1, "feedforward_width": 8}, "multiple of heads"),
    ],
)
def test_a_shape_with_no_count_or_a_width_its_heads_cannot_split_is_refused(
    counts: dict[str, int], message: str
) -> None:
    with pytest.raises(InvalidEncoderSettingError, match=message):
        EncoderShape(**counts)
