import tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from emblema.config.compute_tiers import TIERS_FILE, ComputeTiers
from emblema.shared.kernel.compute import ComputeTier


def shipped() -> dict[str, Any]:
    with TIERS_FILE.open("rb") as handle:
        return tomllib.load(handle)


def test_the_shipped_file_states_every_tier_in_order() -> None:
    tiers = ComputeTiers.load()

    assert [tier.name for tier in tiers.tiers] == [ComputeTier.S, ComputeTier.M, ComputeTier.L]
    assert tiers.profile(ComputeTier.M).device == "T4, fp16"


def test_a_file_elsewhere_is_read_the_same_way(tmp_path: Path) -> None:
    copy = tmp_path / "tiers.toml"
    copy.write_bytes(TIERS_FILE.read_bytes())

    assert ComputeTiers.load(copy) == ComputeTiers.load()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda tiers: tiers.pop(), "exactly once"),
        (lambda tiers: tiers.append(dict(tiers[0])), "exactly once"),
        (lambda tiers: tiers[0].update(name="XL"), "name"),
        (lambda tiers: tiers[0].update(corpus_fraction=1.5), "corpus_fraction"),
        (lambda tiers: tiers[0].update(width=0), "width"),
        (lambda tiers: tiers[0].update(window="shortest"), "window"),
        (lambda tiers: tiers[0].update(dropout=0.1), "dropout"),
    ],
)
def test_a_tier_file_breaking_its_rules_is_refused(change: Any, message: str) -> None:
    raw = shipped()
    change(raw["tiers"])

    with pytest.raises(ValidationError, match=message):
        ComputeTiers.model_validate(raw)
