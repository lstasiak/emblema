import pytest

from emblema.evaluation.domain.exceptions import InvalidLayerReadingError
from emblema.evaluation.domain.transfer.layer_reading import LayerCombination, LayerReading


@pytest.mark.parametrize("text", ["last", "0", "3", "mean", "concat"])
def test_a_reading_is_named_as_it_was_read(text: str) -> None:
    assert str(LayerReading.of(text)) == text


def test_only_the_last_layer_is_the_reading_every_head_had() -> None:
    assert LayerReading.last().is_last
    assert not LayerReading(layer=0).is_last
    assert not LayerReading(combination=LayerCombination.MEAN).is_last


@pytest.mark.parametrize("text", ["", "-1", "1.5", "all", "Last", " 2", "02", "²", "٣"])
def test_a_name_that_is_no_reading_is_refused(text: str) -> None:
    with pytest.raises(InvalidLayerReadingError):
        LayerReading.of(text)


def test_one_layer_and_a_combination_at_once_are_refused() -> None:
    with pytest.raises(InvalidLayerReadingError, match="not layer 2 and mean"):
        LayerReading(layer=2, combination=LayerCombination.MEAN)


def test_a_negative_layer_is_refused() -> None:
    with pytest.raises(InvalidLayerReadingError, match="not negative"):
        LayerReading(layer=-1)


def test_only_a_concatenation_is_wider_than_one_layer() -> None:
    assert LayerReading(combination=LayerCombination.CONCATENATION).width_factor(6) == 6
    assert LayerReading(combination=LayerCombination.MEAN).width_factor(6) == 1
    assert LayerReading(layer=2).width_factor(6) == 1
