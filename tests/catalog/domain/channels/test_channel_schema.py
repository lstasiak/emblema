import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.catalog.domain.channels.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.exceptions import InvalidChannelSchemaError

names = st.text(alphabet=st.characters(categories=["L", "N"]), min_size=1, max_size=8)


def schema_of(*channel_names: str) -> ChannelSchema:
    return ChannelSchema(frozenset(Channel(name) for name in channel_names))


def test_a_channel_is_timed_unless_declared_otherwise() -> None:
    assert not Channel("T2").timeless
    assert Channel("age", timeless=True).timeless


def test_names_are_sorted_regardless_of_declaration_order() -> None:
    assert schema_of("b", "a").names == ("a", "b")


def test_len_and_iteration_expose_the_channels_sorted_by_name() -> None:
    schema = ChannelSchema(frozenset({Channel("b", "K"), Channel("a")}))

    assert len(schema) == 2
    assert tuple(schema) == (Channel("a"), Channel("b", "K"))


def test_rejects_empty_schema() -> None:
    with pytest.raises(InvalidChannelSchemaError, match="at least one"):
        ChannelSchema(frozenset())


def test_rejects_one_name_under_two_units_and_names_it() -> None:
    with pytest.raises(InvalidChannelSchemaError, match=r"\['t'\]"):
        ChannelSchema(frozenset({Channel("t", "K"), Channel("t", "C")}))


@pytest.mark.parametrize("name", ["", " ", " t", "t ", "\tt"])
def test_channel_rejects_blank_or_padded_name(name: str) -> None:
    with pytest.raises(InvalidChannelSchemaError):
        Channel(name)


@given(st.lists(names, min_size=1, max_size=6, unique=True))
def test_declaration_order_does_not_change_the_schema(unique_names: list[str]) -> None:
    forward = schema_of(*unique_names)
    backward = schema_of(*reversed(unique_names))

    assert forward == backward
    assert hash(forward) == hash(backward)
    assert forward.names == tuple(sorted(unique_names))
