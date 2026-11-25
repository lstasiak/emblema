"""A farm's turbines as the cold-start readings load them."""

from pathlib import Path

import numpy as np
import pytest

from emblema.catalog.adapters.readers.greenbyte import GreenbyteCorpusReader
from scripts.fleet_hours import CHANNELS, HELD_UNREAD, another_farm, loaded, loaded_farm
from scripts.wind_fleet_audit import held_out


def test_the_turbines_refused_are_the_audits_draw_and_no_name_is_shared_by_the_farms() -> None:
    farms = (GreenbyteCorpusReader.KELMARSH, GreenbyteCorpusReader.PENMANSHIEL)
    counts = (2, 4)
    drawn = {
        t for f, n in zip(farms, counts, strict=True) for t in held_out(f.name, f.turbines, n, 1)
    }

    assert drawn == HELD_UNREAD
    assert not set(farms[0].turbines) & set(farms[1].turbines)


def written(directory: Path, name: str) -> None:
    np.savez(
        directory / f"hourly-{name}.npz",
        hours=np.arange(5),
        channels=np.asarray(CHANNELS),
        means=np.zeros((5, len(CHANNELS))),
        counts=np.ones((5, len(CHANNELS))),
    )


def test_a_turbine_held_unread_is_refused_before_its_file_is_opened(tmp_path: Path) -> None:
    written(tmp_path, "05")
    written(tmp_path, "07")

    with pytest.raises(ValueError, match=r"\['05'\] are held unread"):
        loaded_farm(tmp_path)
    with pytest.raises(ValueError, match="held unread"):
        loaded(tmp_path, ["6"])
    assert [t.name for t in loaded(tmp_path, ["07"])] == ["07"]


def test_the_farm_answered_is_refused_as_another_farm_under_any_path(tmp_path: Path) -> None:
    own, other = tmp_path / "kelmarsh", tmp_path / "penmanshiel"
    own.mkdir()
    other.mkdir()

    assert another_farm(other, own) == []
    with pytest.raises(ValueError, match="is the farm answered"):
        another_farm(tmp_path / "penmanshiel" / ".." / "kelmarsh", own)
