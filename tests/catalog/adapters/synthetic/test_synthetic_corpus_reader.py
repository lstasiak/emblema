from collections import Counter

import pytest

from emblema.catalog.adapters.synthetic.latent_factor_process import LatentFactorProcess
from emblema.catalog.adapters.synthetic.layouts import CONTROL_A, CONTROL_B, CONTROL_PROCESS
from emblema.catalog.adapters.synthetic.sensor_layout import SensorLayout
from emblema.catalog.adapters.synthetic.synthetic_corpus_reader import SyntheticCorpusReader
from emblema.catalog.domain.channels.channel_schema import Channel
from emblema.catalog.domain.exceptions import UnknownUnitError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.shared.kernel.sampling import SamplingRegime
from tests.support.synthetic import HOSTILE, PROCESS, miniature, uncoupled

SMALL = miniature(CONTROL_A)

# Pinned, one per code path the generator has — a synchronous layout on every step, and one that
# skips steps, reports its channels apart and leaves a unit empty. The corpus is a function of its
# specification and of the arithmetic that turns it into values, so a machine that produces other
# bytes would freeze a different corpus version under the same name, and a change to either has to
# be a deliberate change here.
PINNED = {
    "control-a": "sha256:72e8f4b86906b0270751bf16681eb1698a018ad17a5cf0fa49ed185c15ee926a",
    "control-b": "sha256:fc1933bd4a4e9bfa4642a04af767a3c0f23848963e7a7c3abd954fc3e0abc8d7",
    "hostile": "sha256:696cba9b5a7d709cc996e6b533f83145b64dfada7b9d4c2af2a2cbfb3f41aa08",
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
    turned = SensorLayout.model_validate({**SMALL.model_dump(), dial: turned_to})

    assert reader(turned).describe().content.checksum != reader().describe().content.checksum


def test_the_schema_carries_a_timeless_channel_beside_the_sensors() -> None:
    schema = reader().describe().channel_schema

    assert len(schema) == SMALL.channels + 1
    assert Channel(SensorLayout.GAIN, timeless=True) in schema.channels
    assert all(not channel.timeless for channel in schema if channel.name != SensorLayout.GAIN)


def test_the_gain_of_a_unit_is_its_only_static_feature() -> None:
    for unit in reader().read_units():
        assert [feature.channel for feature in unit.static_features] == [SensorLayout.GAIN]
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
    apart = SensorLayout.model_validate({**SMALL.model_dump(), "cadence": 3, "synchronous": False})
    corpus = reader(apart)
    unit = next(corpus.read_units())

    by_channel: dict[str, set[float]] = {}
    for observation in corpus.read_observations(unit.key):
        by_channel.setdefault(observation.channel, set()).add(observation.time)

    assert corpus.describe().sampling_regime is SamplingRegime.IRREGULAR
    assert len({frozenset(instants) for instants in by_channel.values()}) > 1


def test_switching_off_the_coupling_changes_the_values_and_nothing_else() -> None:
    coupled, null = observations(reader()), observations(reader(uncoupled(SMALL)))

    assert [reading[:3] for reading in coupled] == [reading[:3] for reading in null]
    assert [reading[3] for reading in coupled] != [reading[3] for reading in null]


def test_units_keep_their_extents_when_the_coupling_goes() -> None:
    coupled = [(unit.key, unit.extent) for unit in reader().read_units()]
    null = [(unit.key, unit.extent) for unit in reader(uncoupled(SMALL)).read_units()]

    assert coupled == null


def test_a_channel_cannot_respond_to_more_factors_than_there_are() -> None:
    greedy = SensorLayout.model_validate(
        {**SMALL.model_dump(), "factors_per_channel": CONTROL_PROCESS.factors + 1}
    )

    with pytest.raises(ValueError, match="cannot respond to"):
        reader(greedy)


def test_a_unit_that_reports_nothing_is_still_a_unit() -> None:
    corpus = SyntheticCorpusReader(PROCESS, HOSTILE)

    counts = [sum(1 for _ in corpus.read_observations(unit.key)) for unit in corpus.read_units()]

    assert 0 in counts
    assert corpus.describe().content.observation_count == sum(counts)


def test_a_key_of_another_layout_names_no_unit() -> None:
    corpus = reader()

    for key in ("other/0", str(SMALL.units), f"{SMALL.name}/{SMALL.units}", f"{SMALL.name}/x"):
        with pytest.raises(UnknownUnitError, match="is not a unit of layout"):
            corpus.read_observations(UnitKey(key))
