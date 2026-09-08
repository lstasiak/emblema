import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.catalog.domain.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.exceptions import InvalidChannelSchemaError

names = st.text(alphabet=st.characters(categories=["L", "N"]), min_size=1, max_size=8)


def test_names_follow_declaration_order() -> None:
    schema = ChannelSchema((Channel("b"), Channel("a")))

    assert schema.names == ("b", "a")


def test_len_and_iteration_expose_the_channels() -> None:
    channels = (Channel("a"), Channel("b", "K"))
    schema = ChannelSchema(channels)

    assert len(schema) == 2
    assert tuple(schema) == channels


def test_rejects_empty_schema() -> None:
    with pytest.raises(InvalidChannelSchemaError, match="at least one"):
        ChannelSchema(())


def test_rejects_duplicate_names_and_names_them() -> None:
    with pytest.raises(InvalidChannelSchemaError, match=r"\['a'\]"):
        ChannelSchema((Channel("a"), Channel("b"), Channel("a")))


def test_uniqueness_is_by_name_regardless_of_unit() -> None:
    with pytest.raises(InvalidChannelSchemaError):
        ChannelSchema((Channel("t", "K"), Channel("t", "C")))


@pytest.mark.parametrize("name", ["", " ", " t", "t ", "\tt"])
def test_channel_rejects_blank_or_padded_name(name: str) -> None:
    with pytest.raises(InvalidChannelSchemaError):
        Channel(name)


@given(st.lists(names, min_size=1, max_size=6, unique=True))
def test_any_unique_names_form_a_schema_in_any_order(unique_names: list[str]) -> None:
    forward = ChannelSchema(tuple(Channel(name) for name in unique_names))
    backward = ChannelSchema(tuple(Channel(name) for name in reversed(unique_names)))

    assert set(forward.names) == set(backward.names) == set(unique_names)
    assert forward.names == tuple(reversed(backward.names))
