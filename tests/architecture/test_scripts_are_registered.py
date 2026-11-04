"""Every script stands in the table of ``scripts/README.md``, and a new report says when it goes.

A report written for one reading tends to stay for good. The rule costs one row when a script is
added, and the lists below hold what is older than the rule: they can only shrink.
"""

import re
from datetime import date
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
KINDS_THAT_GO = frozenset({"measurement", "transitional"})
REVIEW = re.compile(r"review by (\d{4}-\d{2}-\d{2})")
NAME = re.compile(r"`([\w.-]+\.(?:py|sh))`")
ROW = re.compile(
    r"^\| (?P<names>.+?) \| (?P<kind>\w+) \| .* \| (?P<retired_by>.+?) \|$", re.MULTILINE
)

NOT_IN_THE_TABLE = frozenset(
    {
        "channel_identity_report.py",
        "channel_pooling_report.py",
        "channel_set_pooling_report.py",
        "convolution_penalty_report.py",
        "detection_states.py",
        "esa_channel_packets.py",
        "label_curve_figures.py",
        "latent_prediction_report.py",
        "magnitude_response_report.py",
        "mex_power_audit.py",
        "probe_penalty_report.py",
        "ranking_calibration_report.py",
        "raw_regime_audit.py",
        "regime_coverage_report.py",
        "seed_spread_report.py",
        "state_geometry_report.py",
        "state_spread_report.py",
        "stay_summary_report.py",
        "stay_summary_states.py",
        "strats_reference_answers.py",
        "strats_reference_data.py",
        "strats_reference_run.py",
        "summary_target_audit.py",
        "taught_class_report.py",
        "vocabulary.py",
        "wqp_fetch.py",
        "wqp_result_table.py",
        "ztf_light_curve_table.py",
    }
)
NO_REVIEW_DATE = frozenset(
    {
        "bootstrap_calibration_report.py",
        "campaign_curve_figures.py",
        "campaign_pairs_report.py",
        "campaign_report.py",
        "cell_pretext_audit.py",
        "classical_baselines_report.py",
        "comparison_power_report.py",
        "corpus_budget_report.py",
        "corpus_facts.py",
        "corpus_saturation_figures.py",
        "corpus_saturation_report.py",
        "detection_power_report.py",
        "detection_side_readings.py",
        "encoder_budget_report.py",
        "excursion_figures.py",
        "excursion_report.py",
        "frozen_representations.py",
        "head_and_representation_report.py",
        "held_out_units.py",
        "loader_throughput_report.py",
        "masked_reconstruction_assessment.py",
        "masked_reconstruction_figures.py",
        "masked_reconstruction_report.py",
        "pretext_triviality_report.py",
        "pretraining_budget_report.py",
        "pretraining_curve_figures.py",
        "pretraining_curve_report.py",
        "pretraining_screen_report.py",
        "published_corpus_report.py",
        "rate_choice_report.py",
        "spectral_probe.py",
        "stage_reader_figures.py",
        "synthetic_control_report.py",
        "token_view_figures.py",
        "token_view_report.py",
        "training_loop_report.py",
        "transduction_report.py",
        "window_sanity_report.py",
    }
)


def rows_of(text: str) -> dict[str, tuple[str, str]]:
    """Each script the table names, with its kind and what retires it."""
    return {
        name: (row["kind"], row["retired_by"])
        for row in ROW.finditer(text)
        for name in NAME.findall(row["names"])
    }


def faults_of(
    text: str, scripts: set[str], *, unlisted: frozenset[str], undated: frozenset[str]
) -> list[str]:
    """What breaks the rule: the first thing a new script must do, and exceptions gone stale."""
    rows = rows_of(text)
    faults = [f"{name} has no row" for name in sorted(scripts - rows.keys() - unlisted)]
    faults += [
        f"{name} has a row: strike it from the exceptions"
        for name in sorted(unlisted & rows.keys())
    ]
    faults += [f"{name} no longer exists" for name in sorted((unlisted | undated) - scripts)]
    for name, (kind, retired_by) in sorted(rows.items()):
        found = REVIEW.search(retired_by)
        if name in undated:
            if found or kind not in KINDS_THAT_GO:
                faults.append(f"{name} needs no exception: strike it")
        elif kind in KINDS_THAT_GO:
            if not found:
                faults.append(f"{name} says neither what retires it nor when to review it")
            elif not retired_by.removesuffix(found.group(0)).strip(" ;"):
                faults.append(f"{name} has a date but nothing that retires it")
            else:
                date.fromisoformat(found.group(1))
    return faults


def test_every_script_is_registered_and_every_new_report_says_when_it_goes() -> None:
    scripts = {
        path.name
        for path in SCRIPTS.iterdir()
        if path.suffix in {".py", ".sh"} and path.name != "__init__.py"
    }
    text = (SCRIPTS / "README.md").read_text(encoding="utf-8")

    assert faults_of(text, scripts, unlisted=NOT_IN_THE_TABLE, undated=NO_REVIEW_DATE) == []


GOOD = "| `a.py` | measurement | a note | the harness; review by 2027-02-01 |\n"
NONE: frozenset[str] = frozenset()


def test_a_report_with_what_retires_it_and_a_date_is_in_order() -> None:
    assert faults_of(GOOD, {"a.py"}, unlisted=NONE, undated=NONE) == []


def test_tooling_stays_without_a_date() -> None:
    assert (
        faults_of("| `a.py` | tooling | runs it | — |\n", {"a.py"}, unlisted=NONE, undated=NONE)
        == []
    )


@pytest.mark.parametrize(
    ("text", "scripts", "fault"),
    [
        ("", {"a.py"}, "a.py has no row"),
        (
            "| `a.py` | measurement | a note | — |\n",
            {"a.py"},
            "a.py says neither what retires it nor when to review it",
        ),
        (
            "| `a.py` | transitional | a note | the harness |\n",
            {"a.py"},
            "a.py says neither what retires it nor when to review it",
        ),
        (
            "| `a.py` | measurement | a note | review by 2027-02-01 |\n",
            {"a.py"},
            "a.py has a date but nothing that retires it",
        ),
    ],
)
def test_a_new_script_that_breaks_the_rule_is_named(
    text: str, scripts: set[str], fault: str
) -> None:
    assert faults_of(text, scripts, unlisted=NONE, undated=NONE) == [fault]


def test_a_date_that_is_no_date_is_refused() -> None:
    text = "| `a.py` | measurement | a note | the harness; review by 2027-13-01 |\n"

    with pytest.raises(ValueError, match="month"):
        faults_of(text, {"a.py"}, unlisted=NONE, undated=NONE)


def test_every_script_a_row_names_counts() -> None:
    text = (
        "| `a.py`, `b.sh`, `c.toml` | measurement | a note | the harness; review by 2027-02-01 |\n"
    )

    assert rows_of(text).keys() == {"a.py", "b.sh"}


def test_an_exception_gone_stale_must_be_struck() -> None:
    assert faults_of(GOOD, {"a.py"}, unlisted=frozenset({"a.py"}), undated=NONE) == [
        "a.py has a row: strike it from the exceptions"
    ]
    assert faults_of(GOOD, {"a.py"}, unlisted=NONE, undated=frozenset({"a.py"})) == [
        "a.py needs no exception: strike it"
    ]
    assert faults_of("", set(), unlisted=frozenset({"gone.py"}), undated=NONE) == [
        "gone.py no longer exists"
    ]
