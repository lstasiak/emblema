"""The file a campaign is declared in, read back into the values a design is stated with.

Closed to keys nobody reads, because a campaign is a registration: a line somebody added and
nothing acts on would read as though it had been in force.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from emblema.evaluation.adapters.campaigns.campaign_file import CampaignFile
from emblema.evaluation.domain.exceptions import InvalidLabelBudgetError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.kernel.compute import ComputeTier

DECLARED = """
name = "baselines-fd001"
tier = "S"

[candidates]
competing = ["from_scratch", "boosted_trees_per_channel"]
control = "from_scratch"
endpoint = "boosted_trees_per_channel"

[budgets]
labels = ["50", "all"]
endpoint = "all"
seeds = [1, 2]

[rules]
minimum_relative_reduction = 0.1
floor_share = 0.02
secondary_family_size = 8
"""


def written(text: str, tmp_path: Path) -> CampaignFile:
    path = tmp_path / "campaign.toml"
    path.write_text(text, encoding="utf-8")
    return CampaignFile.load(path)


def test_a_declared_campaign_reads_back_as_the_values_a_design_is_stated_with(
    tmp_path: Path,
) -> None:
    declared = written(DECLARED, tmp_path)

    assert declared.tier is ComputeTier.S
    assert [str(ref) for ref in declared.competing()] == [
        "from_scratch",
        "boosted_trees_per_channel",
    ]
    assert declared.label_budgets() == (LabelBudget.of(50), LabelBudget.everything())
    assert declared.endpoint_budget() == LabelBudget.everything()


def test_a_campaign_that_says_nothing_about_its_purpose_is_a_tuning_one(tmp_path: Path) -> None:
    # The one purpose that may not be defaulted into is the final run, and it is not this one.
    assert written(DECLARED, tmp_path).purpose is RunPurpose.TUNING


def test_the_interval_a_campaign_states_nothing_about_is_the_registered_one(
    tmp_path: Path,
) -> None:
    drawn = written(DECLARED, tmp_path).paired_bootstrap()

    assert (drawn.resamples, drawn.seed, drawn.level) == (10_000, 1, 0.95)


def test_the_rules_carry_the_error_rate_the_correction_is_applied_at(tmp_path: Path) -> None:
    stricter = written(DECLARED + "alpha = 0.01\n", tmp_path)

    assert stricter.comparison_rules().holm.alpha == 0.01
    assert written(DECLARED, tmp_path).comparison_rules().holm.alpha == 0.05


def test_a_key_nobody_reads_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        written(DECLARED + "\n[bootstrap]\nresamples = 100\nwarmup = 3\n", tmp_path)


def test_a_budget_that_is_neither_a_count_nor_every_window_is_refused(tmp_path: Path) -> None:
    declared = written(DECLARED.replace('"50", "all"', '"most of them", "all"'), tmp_path)

    with pytest.raises(InvalidLabelBudgetError):
        declared.label_budgets()


def test_a_campaign_missing_a_section_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="budgets"):
        written(DECLARED.split("[budgets]")[0], tmp_path)
