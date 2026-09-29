"""The file a campaign is declared in, read back into the values a design is stated with.

Closed to keys nobody reads, because a campaign is a registration: a line somebody added and
nothing acts on would read as though it had been in force.
"""

from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from emblema.evaluation.adapters.campaigns.campaign_file import CampaignFile
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.exceptions import (
    InvalidComparisonRulesError,
    InvalidLabelBudgetError,
    InvalidTunedChoiceError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from emblema.evaluation.domain.statistics.benjamini_hochberg_correction import (
    BenjaminiHochbergCorrection,
)
from emblema.evaluation.domain.statistics.holm_correction import HolmCorrection
from emblema.evaluation.domain.statistics.threshold_kind import ThresholdKind
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.domain.tuning.tuned_choice import TunedChoice
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

    assert stricter.comparison_rules().correction.alpha == 0.01
    assert written(DECLARED, tmp_path).comparison_rules().correction.alpha == 0.05


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


def test_a_selection_declares_how_it_divides_the_tuning_side(tmp_path: Path) -> None:
    declared = written(
        DECLARED.replace('name = "', 'purpose = "selection"\nname = "', 1)
        + "\n[selection]\none_in = 5\n",
        tmp_path,
    )

    assert declared.purpose is RunPurpose.SELECTION
    assert declared.inner_holdout() == InnerHoldout(one_in=5)


def test_a_selection_may_hold_the_same_units_out_whatever_the_seed(tmp_path: Path) -> None:
    declared = written(
        DECLARED.replace('name = "', 'purpose = "selection"\nname = "', 1)
        + "\n[selection]\none_in = 5\ndivision_seed = 101\n",
        tmp_path,
    )

    assert declared.inner_holdout() == InnerHoldout(one_in=5, division_seed=101)


def test_a_comparison_names_each_tuned_pairing_and_the_selection_behind_it(
    tmp_path: Path,
) -> None:
    declared = written(
        DECLARED
        + """
[[tuned]]
candidate = "minirocket"
budget = "200"
variant = "minirocket@grid_resolution=2"
selected_by = "00000000-0000-0000-0000-000000000003"
""",
        tmp_path,
    )

    assert declared.inner_holdout() is None
    assert declared.tuned_choices() == (
        TunedChoice(
            candidate=CandidateRef("minirocket"),
            budget=LabelBudget.of(200),
            variant=CandidateRef("minirocket@grid_resolution=2"),
            selected_by=CampaignId(UUID(int=3)),
        ),
    )


def test_a_tuned_pairing_naming_a_variant_of_another_candidate_is_refused(
    tmp_path: Path,
) -> None:
    declared = written(
        DECLARED
        + """
[[tuned]]
candidate = "minirocket"
budget = "200"
variant = "boosted_trees_spectral@max_depth=3"
selected_by = "00000000-0000-0000-0000-000000000003"
""",
        tmp_path,
    )

    with pytest.raises(InvalidTunedChoiceError, match="not a variant"):
        declared.tuned_choices()


def test_the_family_is_read_under_holm_unless_the_file_names_another_correction(
    tmp_path: Path,
) -> None:
    assert isinstance(written(DECLARED, tmp_path).comparison_rules().correction, HolmCorrection)
    declared = written(DECLARED + 'correction = "benjamini_hochberg"\nalpha = 0.1\n', tmp_path)

    correction = declared.comparison_rules().correction

    assert isinstance(correction, BenjaminiHochbergCorrection)
    assert correction.alpha == 0.1


def test_a_correction_nobody_knows_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InvalidComparisonRulesError, match="no family correction is called"):
        written(DECLARED + 'correction = "bonferroni"\n', tmp_path).comparison_rules()


ABSOLUTE = DECLARED.replace(
    "minimum_relative_reduction = 0.1\nfloor_share = 0.02\n",
    "minimum_absolute_reduction = 0.02\nabsolute_floor = 0.01\n",
)


def test_thresholds_stated_as_shares_are_read_as_shares_of_the_controls_error(
    tmp_path: Path,
) -> None:
    rules = written(DECLARED, tmp_path).comparison_rules()

    assert (rules.threshold, rules.minimum_reduction, rules.floor_part) == (
        ThresholdKind.RELATIVE,
        0.1,
        0.02,
    )


def test_thresholds_stated_in_the_errors_unit_are_read_in_it(tmp_path: Path) -> None:
    rules = written(ABSOLUTE, tmp_path).comparison_rules()

    assert (rules.threshold, rules.minimum_reduction, rules.floor_part) == (
        ThresholdKind.ABSOLUTE,
        0.02,
        0.01,
    )


@pytest.mark.parametrize(
    "rules",
    [
        "minimum_relative_reduction = 0.1\nabsolute_floor = 0.01\n",
        "minimum_relative_reduction = 0.1\nfloor_share = 0.02\nabsolute_floor = 0.01\n",
        "minimum_absolute_reduction = 0.02\n",
        "",
    ],
    ids=["mixed", "both", "half", "neither"],
)
def test_thresholds_stated_other_than_as_one_whole_pair_are_refused(
    tmp_path: Path, rules: str
) -> None:
    stated = DECLARED.replace("minimum_relative_reduction = 0.1\nfloor_share = 0.02\n", rules)

    with pytest.raises(ValidationError, match="both as shares"):
        written(stated, tmp_path)


def test_a_campaign_reads_by_the_measure_it_names_or_leaves_it_to_the_task(
    tmp_path: Path,
) -> None:
    named = written('measure = "auroc_shortfall"\n' + ABSOLUTE, tmp_path)

    assert named.measure is ErrorMeasure.AUROC_SHORTFALL
    assert written(DECLARED, tmp_path).measure is None
