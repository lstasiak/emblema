from collections import Counter

import pytest

from emblema.catalog.adapters.synthetic.latent_factor_process import LatentFactorProcess
from emblema.catalog.adapters.synthetic.layouts import (
    CONTROL_A,
    CONTROL_B,
    CONTROL_PROCESS,
    NULL_A,
)
from emblema.catalog.adapters.synthetic.sensor_layout import SensorLayout
from emblema.catalog.adapters.synthetic.synthetic_corpus_reader import (
    GAIN,
    SyntheticCorpusReader,
)
from emblema.catalog.domain.channels.channel_schema import Channel
from emblema.catalog.domain.exceptions import UnknownUnitError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.shared.kernel.sampling import SamplingRegime
from tests.support.synthetic import HOSTILE, PROCESS, miniature

SMALL = miniature(CONTROL_A)

# Pinned, one per code path the generator has — a synchronous layout on every step, and one that
# skips steps, reports its channels apart and leaves a unit empty. The corpus is a function of its
# specification and of the arithmetic that turns it into values, so a machine that produces other
# bytes would freeze a different corpus version under the same name, and a change to either has to
# be a deliberate change here.
PINNED = {
    "control-a": "sha256:5abe7209308231c5f53ed7651764510e4e106f0510f336d4a8b009b7f8f1c780",
    "control-b": "sha256:4073015ee3828a1a10db2340701b21662fe7e15a7181728428b2c3fb2ceb3c3b",
    "hostile": "sha256:5c39a621d33c284e2a4f5f3745e0f831ff624c8296bf93e63b6d84609d58192f",
}


def reader(layout: SensorLayout = SMALL) -> SyntheticCorpusReader:
    return SyntheticCorpusReader(CONTROL_PROCESS, layout)


def observations(corpus: SyntheticCorpusReader) -> list[tuple[str, str, float, float]]:
    return [
        (unit.key.value, observation.channel, observation.time, observation.value)
        for unit in corpus.read_units()
        for observation in corpus.read_observations(unit.key)
    ]


def test_the_same_specification_gives_the_same_corpus() -> None:
    assert reader().describe() == reader().describe()


@pytest.mark.parametrize(
    ("process", "layout"),
    [
        (CONTROL_PROCESS, SMALL),
        (CONTROL_PROCESS, miniature(CONTROL_B)),
        (PROCESS, HOSTILE),
    ],
    ids=list(PINNED),
)
def test_the_corpus_is_the_same_bytes_wherever_it_is_generated(
    process: LatentFactorProcess, layout: SensorLayout
) -> None:
    corpus = SyntheticCorpusReader(process, layout)

    assert str(corpus.describe().content.checksum) == PINNED[layout.name]


@pytest.mark.parametrize(
    ("dial", "turned_to"),
    [
        ("noise", 0.5),
        ("coupling", 0.5),
        ("missing", 0.5),
        ("cadence", 3),
        ("seed", 999),
        ("trajectory_seed", 999),
    ],
)
def test_a_turned_dial_gives_another_corpus(dial: str, turned_to: float) -> None:
    turned = SMALL.with_dials(**{dial: turned_to})

    assert reader(turned).describe().content.checksum != reader().describe().content.checksum


def test_the_schema_carries_a_timeless_channel_beside_the_sensors() -> None:
    schema = reader().describe().channel_schema

    assert len(schema) == SMALL.channels + 1
    assert Channel(GAIN, timeless=True) in schema.channels
    assert all(not channel.timeless for channel in schema if channel.name != GAIN)


def test_the_gain_of_a_unit_is_its_only_static_feature() -> None:
    for unit in reader().read_units():
        assert [feature.channel for feature in unit.static_features] == [GAIN]
        assert abs(unit.static_features[0].value - 1.0) <= SMALL.gain_spread


def test_every_instant_sits_on_the_grid() -> None:
    for _, _, time, _ in observations(reader()):
        assert time % SMALL.time_step == pytest.approx(0.0)


def test_a_regular_layout_reports_every_channel_on_every_step_it_keeps() -> None:
    corpus = reader()

    assert corpus.describe().sampling_regime is SamplingRegime.REGULAR
    for unit in corpus.read_units():
        instants = Counter(observation.time for observation in corpus.read_observations(unit.key))
        # Nothing but the missing share parts the channels of a synchronous layout, and that
        # share is small enough here that most instants still carry every one of them.
        assert instants.most_common(1)[0][1] == SMALL.channels


def test_an_asynchronous_layout_reports_its_channels_apart() -> None:
    apart = SMALL.with_dials(cadence=3, synchronous=False)
    corpus = reader(apart)
    unit = next(corpus.read_units())

    by_channel: dict[str, set[float]] = {}
    for observation in corpus.read_observations(unit.key):
        by_channel.setdefault(observation.channel, set()).add(observation.time)

    assert corpus.describe().sampling_regime is SamplingRegime.IRREGULAR
    assert len({frozenset(instants) for instants in by_channel.values()}) > 1


def test_switching_off_the_coupling_changes_the_values_and_nothing_else() -> None:
    coupled, null = observations(reader()), observations(reader(SMALL.with_dials(coupling=0.0)))

    assert [reading[:3] for reading in coupled] == [reading[:3] for reading in null]
    assert [reading[3] for reading in coupled] != [reading[3] for reading in null]


def test_a_named_null_corpus_is_its_control_unit_for_unit() -> None:
    # The pair the control actually uses differs in name as well as in coupling, and the name is
    # the prefix of every unit key. Nothing may be drawn from that key: a null whose units were
    # other lengths, sampled at other instants and off by other gains would differ from its twin
    # in far more than the one dial the comparison rests on.
    control, null = reader(miniature(CONTROL_A)), reader(miniature(NULL_A))

    for ours, theirs in zip(control.read_units(), null.read_units(), strict=True):
        assert (ours.extent, ours.static_features) == (theirs.extent, theirs.static_features)
        here = [(o.channel, o.time) for o in control.read_observations(ours.key)]
        there = [(o.channel, o.time) for o in null.read_observations(theirs.key)]
        assert here == there

    assert [reading[1:] for reading in observations(control)] != [
        reading[1:] for reading in observations(null)
    ]


def test_units_keep_their_extents_when_the_coupling_goes() -> None:
    coupled = [(unit.key, unit.extent) for unit in reader().read_units()]
    null = [(unit.key, unit.extent) for unit in reader(SMALL.with_dials(coupling=0.0)).read_units()]

    assert coupled == null


@pytest.mark.parametrize("coupling", [1.0, 0.0])
def test_the_gain_scales_a_unit_whichever_way_the_coupling_is_set(coupling: float) -> None:
    # Switching the coupling off must not also turn the volume down, or the null corpus would be
    # the fainter one as well as the unstructured one, and transfer failing on it would have two
    # explanations. A spread of zero fixes every gain at one, which is the corpus to hold the
    # gained one against.
    plain = reader(SMALL.with_dials(coupling=coupling, noise=0.0, gain_spread=0.0))
    gained = reader(SMALL.with_dials(coupling=coupling, noise=0.0))
    gains = []

    for unit in gained.read_units():
        gain = unit.static_features[0].value
        gains.append(gain)
        ungained = [o.value for o in plain.read_observations(unit.key)]
        assert [o.value for o in gained.read_observations(unit.key)] == pytest.approx(
            [value * gain for value in ungained], abs=1e-5
        )

    assert any(gain != 1.0 for gain in gains)


def test_a_channel_cannot_respond_to_more_factors_than_there_are() -> None:
    greedy = SMALL.with_dials(factors_per_channel=CONTROL_PROCESS.factors + 1)

    with pytest.raises(ValueError, match="cannot respond to"):
        reader(greedy)


def test_a_unit_that_reports_nothing_is_still_a_unit() -> None:
    corpus = SyntheticCorpusReader(PROCESS, HOSTILE)

    counts = [sum(1 for _ in corpus.read_observations(unit.key)) for unit in corpus.read_units()]

    assert 0 in counts
    assert corpus.describe().content.observation_count == sum(counts)


def test_a_key_of_another_layout_names_no_unit() -> None:
    corpus = reader()

    keys = (
        "other/0",
        str(SMALL.units),
        f"{SMALL.name}/{SMALL.units}",
        f"{SMALL.name}/x",
        # A real index in a form this layout never writes: it names no unit rather than the one
        # it would parse to, so two keys cannot stand for one unit.
        f"{SMALL.name}/00",
    )
    for key in keys:
        with pytest.raises(UnknownUnitError, match="is not a unit of layout"):
            corpus.read_observations(UnitKey(key))
