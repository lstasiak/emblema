from pathlib import Path

import pytest
from pydantic import ValidationError

from emblema.config.compute_tiers import ComputeTiers
from emblema.pretraining.adapters.encoder.tier_architecture import architecture_of
from emblema.pretraining.adapters.experiments.experiment_file import ExperimentFile
from emblema.pretraining.domain.exceptions import (
    InvalidMaskingStrategyError,
    InvalidTrainingBudgetError,
)
from emblema.pretraining.domain.training.precision import Precision
from emblema.shared.kernel.compute import ComputeTier

# Every experiment the repository states, which a run is reproduced by naming.
EXPERIMENTS = Path(__file__).resolve().parents[4] / "experiments"

STATED = """
name = "probe-s"
tier = "S"
corpus = "control-a"
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
