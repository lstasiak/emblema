from pathlib import Path

import pytest
from pydantic import ValidationError

from emblema.config.compute_tiers import ComputeTiers
from emblema.pretraining.adapters.encoder.tier_architecture import architecture_of
from emblema.pretraining.adapters.experiments.experiment_file import ExperimentFile
from emblema.pretraining.domain.exceptions import (
    InvalidCorpusFractionError,
    InvalidCorpusPassesError,
    InvalidMaskingStrategyError,
    InvalidObjectiveLossError,
    InvalidTrainingBudgetError,
)
from emblema.pretraining.domain.training.objective_loss import LossKind, ObjectiveLoss
from emblema.pretraining.domain.training.precision import Precision
from emblema.shared.kernel.compute import ComputeTier

# Every experiment the repository states, which a run is reproduced by naming.
EXPERIMENTS = Path(__file__).resolve().parents[4] / "experiments"

STATED = """
name = "probe-s"
tier = "S"
corpora = ["control-a"]
precision = "bf16"
dropout = 0.1
decoder_layers = 2

[masking]
channel_rate = 0.15
block_rate = 0.6
block_span = 0.5
token_rate = 0.1

[budget]
epochs = 8
batch_size = 16
accumulation_steps = 2
learning_rate = 1e-3
warmup_epochs = 1
final_lr_fraction = 0.01
seed = 3

[checkpoint]
every_steps = 50
"""


def written(text: str, directory: Path) -> Path:
    path = directory / "experiment.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_file_states_a_whole_configuration(tmp_path: Path) -> None:
    stated = ExperimentFile.load(written(STATED, tmp_path)).configuration()

    assert stated.name == "probe-s"
    assert stated.tier is ComputeTier.S
    assert stated.precision is Precision.BF16
    assert (stated.dropout, stated.decoder_layers) == (0.1, 2)
    assert stated.budget.effective_batch_size == 32
    assert stated.checkpoint.every_steps == 50


def test_the_shape_comes_from_the_tier_unless_the_file_says_otherwise(tmp_path: Path) -> None:
    tiers = ComputeTiers.load()

    from_tier = ExperimentFile.load(written(STATED, tmp_path)).configuration(tiers)
    cut = ExperimentFile.load(
        written(STATED + "\n[shape]\nwidth = 96\nlayers = 2\n", tmp_path)
    ).configuration(tiers)

    assert from_tier.architecture == architecture_of(tiers.profile(ComputeTier.S))
    assert (cut.architecture.width, cut.architecture.layers) == (96, 2)
    # What the file does not override stays the tier's, so the cut is the stated difference.
    assert cut.architecture.feedforward_width == from_tier.architecture.feedforward_width


def test_the_share_of_the_corpus_comes_from_the_tier_unless_the_file_says_otherwise(
    tmp_path: Path,
) -> None:
    tiers = ComputeTiers.load()

    from_tier = ExperimentFile.load(written(STATED, tmp_path)).configuration(tiers)
    stated = ExperimentFile.load(
        written(
            STATED.replace("decoder_layers = 2\n", "decoder_layers = 2\ncorpus_fraction = 0.5\n"),
            tmp_path,
        )
    ).configuration(tiers)

    assert from_tier.corpus_fraction == tiers.profile(ComputeTier.S).corpus_fraction
    assert stated.corpus_fraction == 0.5


def test_a_key_nobody_reads_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="learning_rate_decay"):
        ExperimentFile.load(written(STATED + "\nlearning_rate_decay = 0.5\n", tmp_path))


def test_a_section_left_out_is_refused(tmp_path: Path) -> None:
    without_budget = STATED[: STATED.index("[budget]")] + STATED[STATED.index("[checkpoint]") :]

    with pytest.raises(ValidationError, match="budget"):
        ExperimentFile.load(written(without_budget, tmp_path))


def test_a_tier_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="tier"):
        ExperimentFile.load(written(STATED.replace('tier = "S"', 'tier = "XL"'), tmp_path))


def test_the_value_objects_say_what_is_out_of_range(tmp_path: Path) -> None:
    hiding_nothing = STATED.replace("channel_rate = 0.15", "channel_rate = 0.0")
    hiding_nothing = hiding_nothing.replace("block_rate = 0.6", "block_rate = 0.0")
    hiding_nothing = hiding_nothing.replace("token_rate = 0.1", "token_rate = 0.0")

    with pytest.raises(InvalidMaskingStrategyError):
        ExperimentFile.load(written(hiding_nothing, tmp_path)).configuration()

    with pytest.raises(InvalidTrainingBudgetError):
        ExperimentFile.load(
            written(STATED.replace("warmup_epochs = 1", "warmup_epochs = 8"), tmp_path)
        ).configuration()


@pytest.mark.parametrize(
    "path", sorted(EXPERIMENTS.glob("*.toml")), ids=lambda path: str(path.stem)
)
def test_every_experiment_in_the_repository_states_a_configuration(path: Path) -> None:
    stated = ExperimentFile.load(path).configuration()

    assert stated.name == path.stem


def test_a_file_that_states_no_objective_is_scored_by_the_square(tmp_path: Path) -> None:
    stated = ExperimentFile.load(written(STATED, tmp_path)).configuration()

    assert stated.loss == ObjectiveLoss(kind=LossKind.MSE)


def test_a_file_states_the_reading_its_hidden_tokens_are_scored_by(tmp_path: Path) -> None:
    text = STATED + '\n[objective]\nkind = "huber"\nhuber_delta = 1.5\n'

    loss = ExperimentFile.load(written(text, tmp_path)).configuration().loss

    assert loss == ObjectiveLoss(kind=LossKind.HUBER, huber_delta=1.5)
    assert loss.is_bounded


def test_a_knee_the_stated_reading_does_not_read_is_refused(tmp_path: Path) -> None:
    text = STATED + "\n[objective]\nhuber_delta = 1.0\n"

    with pytest.raises(InvalidObjectiveLossError, match="no knee"):
        ExperimentFile.load(written(text, tmp_path)).configuration()


def test_a_file_states_how_many_times_an_epoch_reads_a_corpus(tmp_path: Path) -> None:
    weighted = STATED.replace('corpora = ["control-a"]', 'corpora = ["control-a", "control-b"]')
    weighted += "\n[passes]\ncontrol-b = 4\n"

    stated = ExperimentFile.load(written(weighted, tmp_path)).configuration()

    assert stated.passes_of("control-b") == 4
    assert stated.passes_of("control-a") == 1
    assert ExperimentFile.load(written(STATED, tmp_path)).configuration().passes == ()


def test_passes_for_a_corpus_the_run_does_not_read_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="does not read"):
        ExperimentFile.load(written(STATED + "\n[passes]\ncontrol-b = 4\n", tmp_path))


def test_a_corpus_stated_to_be_read_once_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InvalidCorpusPassesError):
        ExperimentFile.load(
            written(STATED + "\n[passes]\ncontrol-a = 1\n", tmp_path)
        ).configuration()


def test_a_file_states_the_share_a_corpus_of_its_own_is_read_at(tmp_path: Path) -> None:
    mixed = STATED.replace('corpora = ["control-a"]', 'corpora = ["control-a", "control-b"]')
    mixed = mixed.replace("decoder_layers = 2", "decoder_layers = 2\ncorpus_fraction = 0.5")
    mixed += "\n[fraction]\ncontrol-b = 0.1\n"

    stated = ExperimentFile.load(written(mixed, tmp_path)).configuration()

    assert stated.corpus_share_of("control-b").fraction == 0.1
    assert stated.corpus_share_of("control-a").fraction == 0.5
    assert ExperimentFile.load(written(STATED, tmp_path)).configuration().fractions == ()


def test_a_fraction_for_a_corpus_the_run_does_not_read_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="fraction is stated for corpora the run does not"):
        ExperimentFile.load(written(STATED + "\n[fraction]\ncontrol-b = 0.1\n", tmp_path))


def test_a_fraction_no_run_could_read_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InvalidCorpusFractionError):
        ExperimentFile.load(
            written(STATED + "\n[fraction]\ncontrol-a = 1.5\n", tmp_path)
        ).configuration()


def test_a_file_states_the_tail_and_one_that_does_not_hides_none(tmp_path: Path) -> None:
    forecasting = STATED.replace(
        "token_rate = 0.1",
        "token_rate = 0.1\nhorizon_rate = 0.5\nhorizon_min_span = 0.15\nhorizon_max_span = 0.5",
    )

    stated = ExperimentFile.load(written(forecasting, tmp_path)).configuration().masking

    assert (stated.horizon_rate, stated.horizon_min_span, stated.horizon_max_span) == (
        0.5,
        0.15,
        0.5,
    )
    assert not ExperimentFile.load(written(STATED, tmp_path)).configuration().masking.has_horizon


def test_a_tail_with_spans_out_of_order_is_refused(tmp_path: Path) -> None:
    disordered = STATED.replace(
        "token_rate = 0.1",
        "token_rate = 0.1\nhorizon_rate = 0.5\nhorizon_min_span = 0.5\nhorizon_max_span = 0.2",
    )

    with pytest.raises(InvalidMaskingStrategyError, match="horizon spans"):
        ExperimentFile.load(written(disordered, tmp_path)).configuration()
